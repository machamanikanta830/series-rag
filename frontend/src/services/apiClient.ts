import type { FastApiErrorResponse } from "../types";

const configuredApiBaseUrl = import.meta.env.VITE_API_BASE_URL?.trim();
const apiBaseUrl = configuredApiBaseUrl?.replace(/\/+$/, "") ?? "";

export class ApiError extends Error {
  readonly status: number | null;

  constructor(message: string, status: number | null = null) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

export interface JsonResponse {
  response: Response;
  payload: unknown;
}

export async function requestJson(
  path: string,
  init?: RequestInit,
): Promise<JsonResponse> {
  let response: Response;
  try {
    response = await fetch(`${apiBaseUrl}${path}`, init);
  } catch {
    throw new ApiError(
      "The SeriesRAG API is unreachable. Confirm the backend is running and try again.",
    );
  }
  return {
    response,
    payload: response.status === 204 ? null : await readJson(response),
  };
}

export function extractErrorDetail(payload: unknown): string | null {
  if (!isRecord(payload)) {
    return null;
  }

  const { detail } = payload as FastApiErrorResponse;
  if (typeof detail === "string" && detail.trim()) {
    return detail;
  }
  if (Array.isArray(detail)) {
    const messages = detail
      .map((issue) => issue.msg)
      .filter((message): message is string => Boolean(message?.trim()));
    return messages.length > 0 ? messages.join(" ") : null;
  }
  return null;
}

async function readJson(response: Response): Promise<unknown> {
  try {
    return await response.json();
  } catch {
    return null;
  }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}
