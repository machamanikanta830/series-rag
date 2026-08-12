import type {
  DocumentUploadResponse,
  FastApiErrorResponse,
} from "../types";

const configuredApiBaseUrl = import.meta.env.VITE_API_BASE_URL?.trim();
const apiBaseUrl = configuredApiBaseUrl?.replace(/\/+$/, "") ?? "";

export class DocumentUploadError extends Error {
  readonly status: number | null;

  constructor(message: string, status: number | null = null) {
    super(message);
    this.name = "DocumentUploadError";
    this.status = status;
  }
}

export async function uploadDocument(
  file: File,
): Promise<DocumentUploadResponse> {
  const formData = new FormData();
  formData.append("file", file);

  let response: Response;
  try {
    response = await fetch(`${apiBaseUrl}/documents`, {
      method: "POST",
      body: formData,
    });
  } catch {
    throw new DocumentUploadError(
      "The SeriesRAG API is unreachable. Confirm the backend is running and try again.",
    );
  }

  const payload = await readJson(response);
  if (!response.ok) {
    throw new DocumentUploadError(
      messageForStatus(response.status, extractErrorDetail(payload)),
      response.status,
    );
  }

  if (!isDocumentUploadResponse(payload)) {
    throw new DocumentUploadError(
      "The API returned an unexpected upload response. Please try again.",
      response.status,
    );
  }
  return payload;
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

function messageForStatus(status: number, detail: string | null): string {
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

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}
