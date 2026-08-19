import type {
  DocumentDetail,
  DocumentSummary,
  DocumentUploadResponse,
} from "../types";
import { ApiError, extractErrorDetail, requestJson } from "./apiClient";
import { isNonNegativeInteger, isRecord, isStringRecord } from "./validation";

export { ApiError as DocumentApiError } from "./apiClient";

export async function uploadDocument(
  file: File,
): Promise<DocumentUploadResponse> {
  const formData = new FormData();
  formData.append("file", file);

  const { response, payload } = await requestJson("/documents", {
    method: "POST",
    body: formData,
  });
  if (!response.ok) {
    throw new ApiError(
      uploadMessageForStatus(response.status, extractErrorDetail(payload)),
      response.status,
    );
  }

  if (!isDocumentUploadResponse(payload)) {
    throw new ApiError(
      "The API returned an unexpected upload response. Please try again.",
      response.status,
    );
  }
  return payload;
}

export async function listDocuments(): Promise<DocumentSummary[]> {
  const { response, payload } = await requestJson("/documents");
  if (!response.ok) {
    throw new ApiError(
      readMessageForStatus(response.status, extractErrorDetail(payload)),
      response.status,
    );
  }
  if (!Array.isArray(payload) || !payload.every(isDocumentSummary)) {
    throw new ApiError(
      "The API returned an unexpected document list. Please try again.",
      response.status,
    );
  }
  return payload;
}

export async function getDocument(
  documentId: string,
): Promise<DocumentDetail> {
  const { response, payload } = await requestJson(
    `/documents/${encodeURIComponent(documentId)}`,
  );
  if (!response.ok) {
    throw new ApiError(
      readMessageForStatus(response.status, extractErrorDetail(payload)),
      response.status,
    );
  }
  if (!isDocumentDetail(payload)) {
    throw new ApiError(
      "The API returned unexpected document details. Please try again.",
      response.status,
    );
  }
  return payload;
}

export async function deleteDocument(documentId: string): Promise<void> {
  const { response, payload } = await requestJson(
    `/documents/${encodeURIComponent(documentId)}`,
    { method: "DELETE" },
  );
  if (response.status === 204) {
    return;
  }
  if (!response.ok) {
    throw new ApiError(
      deleteMessageForStatus(response.status, extractErrorDetail(payload)),
      response.status,
    );
  }
  throw new ApiError(
    "The API returned an unexpected deletion response. Please try again.",
    response.status,
  );
}

function uploadMessageForStatus(status: number, detail: string | null): string {
  if (status === 413) {
    return "This file is larger than the 1 MB upload limit.";
  }
  if (status === 415) {
    return "This file type is not supported. Choose TXT, Markdown, PDF, or DOCX.";
  }
  if (status === 422) {
    return detail ?? "The API could not read usable content from this document.";
  }
  if (status >= 500) {
    return "The server could not upload this document. Please try again.";
  }
  return detail ?? "The document could not be uploaded. Please try again.";
}

function readMessageForStatus(status: number, detail: string | null): string {
  if (status === 404) {
    return "This document could not be found. It may no longer be available.";
  }
  if (status >= 500) {
    return "The server could not load the document catalog. Please try again.";
  }
  return detail ?? "The document catalog could not be loaded. Please try again.";
}

function deleteMessageForStatus(status: number, detail: string | null): string {
  if (status === 404) {
    return "This document no longer exists. The catalog will be updated.";
  }
  if (status >= 500) {
    return "The server could not delete this document. Please try again.";
  }
  return detail ?? "The document could not be deleted. Please try again.";
}

function isDocumentUploadResponse(
  payload: unknown,
): payload is DocumentUploadResponse {
  if (!isRecord(payload)) {
    return false;
  }
  return (
    typeof payload.document_id === "string" &&
    typeof payload.filename === "string" &&
    typeof payload.chunks_created === "number" &&
    typeof payload.embedding_dimension === "number" &&
    typeof payload.vector_store_name === "string"
  );
}

function isDocumentSummary(payload: unknown): payload is DocumentSummary {
  if (!isRecord(payload)) {
    return false;
  }
  return (
    typeof payload.document_id === "string" &&
    typeof payload.filename === "string" &&
    isNonNegativeInteger(payload.chunk_count)
  );
}

function isDocumentDetail(payload: unknown): payload is DocumentDetail {
  const chunks = isRecord(payload) ? payload.chunks : null;
  return (
    isDocumentSummary(payload) &&
    Array.isArray(chunks) &&
    chunks.every(isDocumentChunk)
  );
}

function isDocumentChunk(payload: unknown): payload is DocumentDetail["chunks"][number] {
  if (!isRecord(payload)) {
    return false;
  }
  return (
    typeof payload.chunk_id === "string" &&
    isNonNegativeInteger(payload.chunk_index) &&
    typeof payload.text === "string" &&
    isStringRecord(payload.metadata)
  );
}
