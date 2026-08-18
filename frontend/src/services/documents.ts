import type {
  DocumentDetail,
  DocumentSummary,
  DocumentUploadResponse,
  FastApiErrorResponse,
} from "../types";

const configuredApiBaseUrl = import.meta.env.VITE_API_BASE_URL?.trim();
const apiBaseUrl = configuredApiBaseUrl?.replace(/\/+$/, "") ?? "";

export class DocumentApiError extends Error {
  readonly status: number | null;

  constructor(message: string, status: number | null = null) {
    super(message);
    this.name = "DocumentApiError";
    this.status = status;
  }
}

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
    throw new DocumentApiError(
      uploadMessageForStatus(response.status, extractErrorDetail(payload)),
      response.status,
    );
  }

  if (!isDocumentUploadResponse(payload)) {
    throw new DocumentApiError(
      "The API returned an unexpected upload response. Please try again.",
      response.status,
    );
  }
  return payload;
}

export async function listDocuments(): Promise<DocumentSummary[]> {
  const { response, payload } = await requestJson("/documents");
  if (!response.ok) {
    throw new DocumentApiError(
      readMessageForStatus(response.status, extractErrorDetail(payload)),
      response.status,
    );
  }
  if (!Array.isArray(payload) || !payload.every(isDocumentSummary)) {
    throw new DocumentApiError(
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
    throw new DocumentApiError(
      readMessageForStatus(response.status, extractErrorDetail(payload)),
      response.status,
    );
  }
  if (!isDocumentDetail(payload)) {
    throw new DocumentApiError(
      "The API returned unexpected document details. Please try again.",
      response.status,
    );
  }
  return payload;
}

interface JsonResponse {
  response: Response;
  payload: unknown;
}

async function requestJson(path: string, init?: RequestInit): Promise<JsonResponse> {
  let response: Response;
  try {
    response = await fetch(`${apiBaseUrl}${path}`, init);
  } catch {
    throw new DocumentApiError(
      "The SeriesRAG API is unreachable. Confirm the backend is running and try again.",
    );
  }
  return { response, payload: await readJson(response) };
}

async function readJson(response: Response): Promise<unknown> {
  try {
    return await response.json();
  } catch {
    return null;
  }
}

function extractErrorDetail(payload: unknown): string | null {
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

function isNonNegativeInteger(value: unknown): value is number {
  return typeof value === "number" && Number.isInteger(value) && value >= 0;
}

function isStringRecord(value: unknown): value is Record<string, string> {
  return (
    isRecord(value) &&
    Object.entries(value).every(
      ([key, entryValue]) => key.length > 0 && typeof entryValue === "string",
    )
  );
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}
