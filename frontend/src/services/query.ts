import type { QueryResponse, QuerySource } from "../types";
import { ApiError, extractErrorDetail, requestJson } from "./apiClient";

export { ApiError as QueryApiError } from "./apiClient";

export async function queryDocuments(
  question: string,
  topK: number,
): Promise<QueryResponse> {
  const normalizedQuestion = question.trim();
  if (!normalizedQuestion) {
    throw new ApiError("Enter a question before submitting.");
  }
  if (!Number.isInteger(topK) || topK <= 0) {
    throw new ApiError("Top k must be a positive whole number.");
  }

  const { response, payload } = await requestJson("/query", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question: normalizedQuestion, top_k: topK }),
  });

  if (!response.ok) {
    throw new ApiError(
      queryMessageForStatus(response.status, extractErrorDetail(payload)),
      response.status,
    );
  }
  if (!isQueryResponse(payload)) {
    throw new ApiError(
      "The API returned an unexpected query response. Please try again.",
      response.status,
    );
  }
  return payload;
}

function queryMessageForStatus(status: number, detail: string | null): string {
  if (status === 422) {
    return detail ?? "No usable source context was found for this question.";
  }
  if (status >= 500) {
    return "The server could not answer this question. Please try again.";
  }
  return detail ?? "The question could not be completed. Please try again.";
}

function isQueryResponse(payload: unknown): payload is QueryResponse {
  if (!isRecord(payload)) {
    return false;
  }
  return (
    typeof payload.answer === "string" &&
    typeof payload.context === "string" &&
    typeof payload.prompt === "string" &&
    Array.isArray(payload.sources) &&
    payload.sources.every(isQuerySource)
  );
}

function isQuerySource(payload: unknown): payload is QuerySource {
  if (!isRecord(payload)) {
    return false;
  }
  return (
    typeof payload.chunk_id === "string" &&
    typeof payload.document_id === "string" &&
    typeof payload.source_name === "string" &&
    Number.isInteger(payload.chunk_index) &&
    (payload.chunk_index as number) >= 0 &&
    typeof payload.text === "string" &&
    typeof payload.score === "number" &&
    Number.isFinite(payload.score)
  );
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}
