export interface DocumentUploadResponse {
  document_id: string;
  filename: string;
  chunks_created: number;
  embedding_dimension: number;
  vector_store_name: string;
}

export interface FastApiValidationIssue {
  msg?: string;
}

export interface FastApiErrorResponse {
  detail?: string | FastApiValidationIssue[];
}
