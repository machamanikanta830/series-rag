export interface DocumentUploadResponse {
  document_id: string;
  filename: string;
  chunks_created: number;
  embedding_dimension: number;
  vector_store_name: string;
}

export interface DocumentSummary {
  document_id: string;
  filename: string;
  chunk_count: number;
}

export interface DocumentChunk {
  chunk_id: string;
  chunk_index: number;
  text: string;
  metadata: Record<string, string>;
}

export interface DocumentDetail extends DocumentSummary {
  chunks: DocumentChunk[];
}

export interface QuerySource {
  chunk_id: string;
  document_id: string;
  source_name: string;
  chunk_index: number;
  text: string;
  score: number;
}

export interface QueryResponse {
  answer: string;
  context: string;
  prompt: string;
  sources: QuerySource[];
}

export interface FastApiValidationIssue {
  msg?: string;
}

export interface FastApiErrorResponse {
  detail?: string | FastApiValidationIssue[];
}
