import { type ChangeEvent, type FormEvent, useRef, useState } from "react";

import { PageIntro } from "../../components/PageIntro";
import { DocumentApiError, uploadDocument } from "../../services";
import type { DocumentUploadResponse } from "../../types";

const MAX_UPLOAD_BYTES = 1_048_576;
const SUPPORTED_EXTENSIONS = new Set(["txt", "md", "markdown", "pdf", "docx"]);
const FILE_ACCEPT = ".txt,.md,.markdown,.pdf,.docx";

export function UploadPage() {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [validationError, setValidationError] = useState<string | null>(null);
  const [submitError, setSubmitError] = useState<string | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [result, setResult] = useState<DocumentUploadResponse | null>(null);

  function handleFileChange(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0] ?? null;
    setSelectedFile(file);
    setResult(null);
    setSubmitError(null);
    setValidationError(file === null ? null : validateFile(file));
  }

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setSubmitError(null);

    if (selectedFile === null) {
      setValidationError("Choose one document before uploading.");
      fileInputRef.current?.focus();
      return;
    }

    const fileError = validateFile(selectedFile);
    if (fileError !== null) {
      setValidationError(fileError);
      fileInputRef.current?.focus();
      return;
    }

    setValidationError(null);
    setIsSubmitting(true);
    try {
      const uploadResult = await uploadDocument(selectedFile);
      setResult(uploadResult);
    } catch (error) {
      setSubmitError(
        error instanceof DocumentApiError
          ? error.message
          : "The document could not be uploaded. Please try again.",
      );
    } finally {
      setIsSubmitting(false);
    }
  }

  function resetUpload() {
    setSelectedFile(null);
    setValidationError(null);
    setSubmitError(null);
    setResult(null);
    if (fileInputRef.current !== null) {
      fileInputRef.current.value = "";
      fileInputRef.current.focus();
    }
  }

  return (
    <div>
      <PageIntro
        eyebrow="Sources"
        title="Upload"
        description="Add one source document to the current collection and inspect the ingestion result."
      />

      {result === null ? (
        <form
          className="mt-12 max-w-3xl border-y border-slate-200 py-10 sm:mt-16 sm:py-14"
          onSubmit={handleSubmit}
          noValidate
        >
          <div>
            <label
              htmlFor="document-file"
              className="block text-sm font-semibold text-slate-950"
            >
              Source document
            </label>
            <p id="document-file-hint" className="mt-2 text-sm leading-6 text-slate-500">
              TXT, MD, Markdown, native-text PDF, or DOCX. Maximum size: 1 MB.
            </p>
            <input
              ref={fileInputRef}
              id="document-file"
              name="document-file"
              type="file"
              accept={FILE_ACCEPT}
              aria-describedby="document-file-hint document-file-status"
              aria-invalid={validationError !== null}
              disabled={isSubmitting}
              onChange={handleFileChange}
              className="mt-5 block w-full cursor-pointer rounded-xl border border-slate-300 bg-white text-sm text-slate-600 file:mr-5 file:border-0 file:border-r file:border-slate-200 file:bg-slate-50 file:px-5 file:py-3.5 file:font-semibold file:text-slate-800 hover:border-slate-400 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-brand-600 disabled:cursor-not-allowed disabled:opacity-60"
            />
          </div>

          <div id="document-file-status" className="mt-6" aria-live="polite">
            {selectedFile !== null ? (
              <dl className="grid gap-4 rounded-xl bg-slate-100 px-5 py-4 sm:grid-cols-2">
                <div>
                  <dt className="text-xs font-bold uppercase tracking-[0.14em] text-slate-400">
                    Selected file
                  </dt>
                  <dd className="mt-1 break-all text-sm font-semibold text-slate-900">
                    {selectedFile.name}
                  </dd>
                </div>
                <div>
                  <dt className="text-xs font-bold uppercase tracking-[0.14em] text-slate-400">
                    File size
                  </dt>
                  <dd className="mt-1 text-sm font-semibold text-slate-900">
                    {formatFileSize(selectedFile.size)}
                  </dd>
                </div>
              </dl>
            ) : (
              <p className="text-sm text-slate-500">No document selected.</p>
            )}
          </div>

          {validationError !== null ? (
            <p
              className="mt-5 rounded-xl border border-amber-200 bg-amber-50 px-4 py-3 text-sm leading-6 text-amber-900"
              role="alert"
            >
              {validationError}
            </p>
          ) : null}

          {submitError !== null ? (
            <p
              className="mt-5 rounded-xl border border-red-200 bg-red-50 px-4 py-3 text-sm leading-6 text-red-800"
              role="alert"
            >
              {submitError}
            </p>
          ) : null}

          <button
            type="submit"
            disabled={
              selectedFile === null || validationError !== null || isSubmitting
            }
            className="mt-8 inline-flex min-h-11 items-center justify-center rounded-xl bg-brand-600 px-6 py-3 text-sm font-semibold text-white shadow-sm shadow-brand-900/10 hover:bg-brand-700 focus-visible:outline-2 focus-visible:outline-offset-3 focus-visible:outline-brand-600 disabled:cursor-not-allowed disabled:bg-slate-300 disabled:shadow-none"
          >
            {isSubmitting ? "Uploading document…" : "Upload document"}
          </button>

          {isSubmitting ? (
            <p className="mt-4 text-sm text-slate-500" role="status" aria-live="polite">
              The document is being validated, chunked, embedded, and stored.
            </p>
          ) : null}
        </form>
      ) : (
        <UploadSuccess result={result} onReset={resetUpload} />
      )}
    </div>
  );
}

interface UploadSuccessProps {
  result: DocumentUploadResponse;
  onReset: () => void;
}

function UploadSuccess({ result, onReset }: UploadSuccessProps) {
  const details = [
    ["Filename", result.filename],
    ["Document ID", result.document_id],
    ["Chunks created", String(result.chunks_created)],
    ["Embedding dimension", String(result.embedding_dimension)],
    ["Vector store", result.vector_store_name],
  ] as const;

  return (
    <section
      className="mt-12 max-w-3xl border-y border-slate-200 py-10 sm:mt-16 sm:py-14"
      aria-labelledby="upload-success-title"
      role="status"
    >
      <p className="text-xs font-bold uppercase tracking-[0.18em] text-brand-700">
        Upload complete
      </p>
      <h2
        id="upload-success-title"
        className="mt-3 text-2xl font-semibold tracking-tight text-slate-950"
      >
        Document ready for retrieval
      </h2>
      <p className="mt-3 text-sm leading-6 text-slate-600">
        The backend finished ingestion and returned the following result.
      </p>

      <dl className="mt-8 divide-y divide-slate-200 border-y border-slate-200">
        {details.map(([label, value]) => (
          <div
            key={label}
            className="grid gap-1 py-4 sm:grid-cols-[11rem_minmax(0,1fr)] sm:gap-6"
          >
            <dt className="text-xs font-bold uppercase tracking-[0.14em] text-slate-400">
              {label}
            </dt>
            <dd className="break-all text-sm font-medium text-slate-900">{value}</dd>
          </div>
        ))}
      </dl>

      <button
        type="button"
        onClick={onReset}
        className="mt-8 inline-flex min-h-11 items-center justify-center rounded-xl border border-brand-600 px-6 py-3 text-sm font-semibold text-brand-700 hover:bg-brand-50 focus-visible:outline-2 focus-visible:outline-offset-3 focus-visible:outline-brand-600"
      >
        Upload another file
      </button>
    </section>
  );
}

function validateFile(file: File): string | null {
  const extension = file.name.split(".").pop()?.toLowerCase() ?? "";
  if (!SUPPORTED_EXTENSIONS.has(extension)) {
    return "Choose a TXT, Markdown, PDF, or DOCX file.";
  }
  if (file.size > MAX_UPLOAD_BYTES) {
    return "This file is larger than the 1 MB upload limit.";
  }
  return null;
}

function formatFileSize(sizeInBytes: number): string {
  if (sizeInBytes < 1024) {
    return `${sizeInBytes} B`;
  }
  const sizeInKilobytes = sizeInBytes / 1024;
  if (sizeInKilobytes < 1024) {
    return `${sizeInKilobytes.toFixed(1)} KB`;
  }
  return `${(sizeInKilobytes / 1024).toFixed(2)} MB`;
}
