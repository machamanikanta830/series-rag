import { useEffect, useRef, useState } from "react";
import { Link } from "react-router-dom";

import { PageIntro } from "../../components/PageIntro";
import {
  deleteDocument,
  DocumentApiError,
  getDocument,
  listDocuments,
} from "../../services";
import type {
  DocumentChunk,
  DocumentDetail,
  DocumentSummary,
} from "../../types";

export function DocumentsPage() {
  const [documents, setDocuments] = useState<DocumentSummary[] | null>(null);
  const [listError, setListError] = useState<string | null>(null);
  const [listAttempt, setListAttempt] = useState(0);
  const [selectedDocumentId, setSelectedDocumentId] = useState<string | null>(null);
  const [selectedDocument, setSelectedDocument] = useState<DocumentDetail | null>(
    null,
  );
  const [detailError, setDetailError] = useState<string | null>(null);
  const [detailAttempt, setDetailAttempt] = useState(0);
  const [deletionTarget, setDeletionTarget] = useState<DocumentDetail | null>(null);
  const [deletionError, setDeletionError] = useState<string | null>(null);
  const [isDeleting, setIsDeleting] = useState(false);
  const [deletionFeedback, setDeletionFeedback] = useState<string | null>(null);
  const deleteTriggerRef = useRef<HTMLButtonElement | null>(null);
  const feedbackRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    let isCurrent = true;
    setDocuments(null);
    setListError(null);

    void listDocuments()
      .then((loadedDocuments) => {
        if (!isCurrent) return;
        setDocuments(loadedDocuments);
        setSelectedDocumentId((currentId) =>
          currentId !== null &&
          loadedDocuments.some((document) => document.document_id === currentId)
            ? currentId
            : null,
        );
      })
      .catch((error: unknown) => {
        if (isCurrent) setListError(toFriendlyMessage(error));
      });

    return () => {
      isCurrent = false;
    };
  }, [listAttempt]);

  useEffect(() => {
    if (selectedDocumentId === null) {
      setSelectedDocument(null);
      setDetailError(null);
      return;
    }

    let isCurrent = true;
    setSelectedDocument(null);
    setDetailError(null);

    void getDocument(selectedDocumentId)
      .then((document) => {
        if (isCurrent) setSelectedDocument(document);
      })
      .catch((error: unknown) => {
        if (isCurrent) setDetailError(toFriendlyMessage(error));
      });

    return () => {
      isCurrent = false;
    };
  }, [detailAttempt, selectedDocumentId]);

  function selectDocument(documentId: string) {
    if (isDeleting) return;
    setDeletionTarget(null);
    setDeletionError(null);
    setSelectedDocumentId(documentId);
    setDetailAttempt(0);
  }

  function requestDeletion(
    document: DocumentDetail,
    trigger: HTMLButtonElement,
  ) {
    deleteTriggerRef.current = trigger;
    setDeletionError(null);
    setDeletionTarget(document);
  }

  function cancelDeletion() {
    setDeletionTarget(null);
    setDeletionError(null);
    requestAnimationFrame(() => deleteTriggerRef.current?.focus());
  }

  function removeDeletedDocument(document: DocumentDetail, message: string) {
    setDocuments((currentDocuments) =>
      currentDocuments?.filter(
        (candidate) => candidate.document_id !== document.document_id,
      ) ?? null,
    );
    setSelectedDocumentId(null);
    setSelectedDocument(null);
    setDetailError(null);
    setDeletionTarget(null);
    setDeletionError(null);
    setDeletionFeedback(message);
    requestAnimationFrame(() => feedbackRef.current?.focus());
  }

  async function confirmDeletion() {
    if (deletionTarget === null || isDeleting) return;

    const document = deletionTarget;
    setIsDeleting(true);
    setDeletionError(null);
    setDeletionFeedback(null);
    try {
      await deleteDocument(document.document_id);
      removeDeletedDocument(document, `${document.filename} was deleted.`);
    } catch (error) {
      if (error instanceof DocumentApiError && error.status === 404) {
        removeDeletedDocument(
          document,
          `${document.filename} was already removed. The catalog is now up to date.`,
        );
      } else {
        setDeletionError(toDeletionMessage(error));
      }
    } finally {
      setIsDeleting(false);
    }
  }

  return (
    <div>
      <PageIntro
        eyebrow="Collection"
        title="Documents"
        description="Inspect ingested sources, their chunk boundaries, and the provenance retained for retrieval."
      />

      <div className="mt-12 sm:mt-16">
        {deletionFeedback !== null ? (
          <div
            ref={feedbackRef}
            tabIndex={-1}
            className="mb-7 rounded-2xl border border-brand-100 bg-brand-50 px-5 py-4 text-sm font-medium text-brand-900 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-brand-600"
            role="status"
            aria-live="polite"
          >
            {deletionFeedback}
          </div>
        ) : null}
        {documents === null && listError === null ? <ListLoading /> : null}
        {listError !== null ? (
          <ListError
            message={listError}
            onRetry={() => setListAttempt((attempt) => attempt + 1)}
          />
        ) : null}
        {documents?.length === 0 ? <EmptyCatalog /> : null}
        {documents !== null && documents.length > 0 ? (
          <div className="grid items-start gap-8 lg:grid-cols-[minmax(17rem,0.8fr)_minmax(0,1.6fr)]">
            <DocumentList
              documents={documents}
              selectedDocumentId={selectedDocumentId}
              disabled={isDeleting}
              onSelect={selectDocument}
            />
            <DocumentInspector
              selectedDocumentId={selectedDocumentId}
              document={selectedDocument}
              error={detailError}
              deletionTarget={deletionTarget}
              deletionError={deletionError}
              isDeleting={isDeleting}
              onRetry={() => setDetailAttempt((attempt) => attempt + 1)}
              onRequestDeletion={requestDeletion}
              onCancelDeletion={cancelDeletion}
              onConfirmDeletion={() => void confirmDeletion()}
            />
          </div>
        ) : null}
      </div>
    </div>
  );
}

interface DocumentListProps {
  documents: DocumentSummary[];
  selectedDocumentId: string | null;
  disabled: boolean;
  onSelect: (documentId: string) => void;
}

function DocumentList({
  documents,
  selectedDocumentId,
  disabled,
  onSelect,
}: DocumentListProps) {
  return (
    <section aria-labelledby="document-list-title">
      <div className="mb-4 flex items-end justify-between gap-4">
        <div>
          <p className="text-xs font-bold uppercase tracking-[0.16em] text-brand-700">
            Source inventory
          </p>
          <h2 id="document-list-title" className="mt-2 text-xl font-semibold">
            {documents.length} {documents.length === 1 ? "document" : "documents"}
          </h2>
        </div>
        <Link
          to="/upload"
          className="rounded-lg px-2 py-1 text-sm font-semibold text-brand-700 hover:bg-brand-50 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-brand-600"
        >
          Upload
        </Link>
      </div>

      <ul className="space-y-3">
        {documents.map((document) => {
          const isSelected = document.document_id === selectedDocumentId;
          return (
            <li key={document.document_id}>
              <button
                type="button"
                aria-pressed={isSelected}
                aria-controls="document-inspector"
                disabled={disabled}
                onClick={() => onSelect(document.document_id)}
                className={`w-full rounded-2xl border px-5 py-4 text-left transition-colors focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-brand-600 disabled:cursor-wait disabled:opacity-60 ${
                  isSelected
                    ? "border-brand-500 bg-brand-50 shadow-sm"
                    : "border-slate-200 bg-white hover:border-slate-300 hover:bg-slate-50"
                }`}
              >
                <span className="block break-words text-sm font-semibold text-slate-950">
                  {document.filename}
                </span>
                <span className="mt-3 flex flex-wrap items-center gap-x-4 gap-y-1 text-xs text-slate-500">
                  <span>
                    {document.chunk_count} {document.chunk_count === 1 ? "chunk" : "chunks"}
                  </span>
                  <span className="font-mono" title={document.document_id}>
                    {shortenId(document.document_id)}
                  </span>
                </span>
              </button>
            </li>
          );
        })}
      </ul>
    </section>
  );
}

interface DocumentInspectorProps {
  selectedDocumentId: string | null;
  document: DocumentDetail | null;
  error: string | null;
  deletionTarget: DocumentDetail | null;
  deletionError: string | null;
  isDeleting: boolean;
  onRetry: () => void;
  onRequestDeletion: (
    document: DocumentDetail,
    trigger: HTMLButtonElement,
  ) => void;
  onCancelDeletion: () => void;
  onConfirmDeletion: () => void;
}

function DocumentInspector({
  selectedDocumentId,
  document,
  error,
  deletionTarget,
  deletionError,
  isDeleting,
  onRetry,
  onRequestDeletion,
  onCancelDeletion,
  onConfirmDeletion,
}: DocumentInspectorProps) {
  return (
    <section
      id="document-inspector"
      aria-label="Document details"
      className="min-w-0 rounded-2xl border border-slate-200 bg-white p-5 sm:p-7"
    >
      {selectedDocumentId === null ? (
        <div className="py-12 text-center sm:py-20">
          <p className="text-sm font-semibold text-slate-900">Select a document</p>
          <p className="mx-auto mt-2 max-w-sm text-sm leading-6 text-slate-500">
            Choose a source from the inventory to inspect its chunks and provenance.
          </p>
        </div>
      ) : null}

      {selectedDocumentId !== null && document === null && error === null ? (
        <div className="py-12 text-center" role="status" aria-live="polite">
          <p className="text-sm font-semibold text-slate-900">Loading document…</p>
          <p className="mt-2 text-sm text-slate-500">
            Fetching chunk text and provenance.
          </p>
        </div>
      ) : null}

      {error !== null ? (
        <div className="py-8" role="alert">
          <p className="text-sm font-semibold text-red-800">Document unavailable</p>
          <p className="mt-2 text-sm leading-6 text-slate-600">{error}</p>
          <button
            type="button"
            onClick={onRetry}
            className="mt-5 rounded-xl border border-slate-300 px-4 py-2.5 text-sm font-semibold text-slate-800 hover:bg-slate-50 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-brand-600"
          >
            Try again
          </button>
        </div>
      ) : null}

      {document !== null ? (
        <DocumentContents
          document={document}
          deletionTarget={deletionTarget}
          deletionError={deletionError}
          isDeleting={isDeleting}
          onRequestDeletion={onRequestDeletion}
          onCancelDeletion={onCancelDeletion}
          onConfirmDeletion={onConfirmDeletion}
        />
      ) : null}
    </section>
  );
}

function DocumentContents({
  document,
  deletionTarget,
  deletionError,
  isDeleting,
  onRequestDeletion,
  onCancelDeletion,
  onConfirmDeletion,
}: {
  document: DocumentDetail;
  deletionTarget: DocumentDetail | null;
  deletionError: string | null;
  isDeleting: boolean;
  onRequestDeletion: (
    document: DocumentDetail,
    trigger: HTMLButtonElement,
  ) => void;
  onCancelDeletion: () => void;
  onConfirmDeletion: () => void;
}) {
  return (
    <div>
      <div className="flex flex-col gap-5 sm:flex-row sm:items-start sm:justify-between">
        <div className="min-w-0">
          <p className="text-xs font-bold uppercase tracking-[0.16em] text-brand-700">
            Selected source
          </p>
          <h2
            id="document-detail-title"
            className="mt-2 break-words text-2xl font-semibold tracking-tight text-slate-950"
          >
            {document.filename}
          </h2>
        </div>
        <button
          type="button"
          disabled={deletionTarget !== null}
          onClick={(event) => onRequestDeletion(document, event.currentTarget)}
          className="shrink-0 self-start rounded-xl border border-red-200 px-4 py-2.5 text-sm font-semibold text-red-700 hover:border-red-300 hover:bg-red-50 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-red-700 disabled:cursor-default disabled:opacity-50"
        >
          Delete document
        </button>
      </div>
      <div className="mt-3 flex flex-wrap gap-x-5 gap-y-1 text-xs text-slate-500">
        <span>
          {document.chunk_count} {document.chunk_count === 1 ? "chunk" : "chunks"}
        </span>
        <span className="break-all font-mono">{document.document_id}</span>
      </div>

      {deletionTarget !== null ? (
        <DeleteConfirmation
          document={deletionTarget}
          error={deletionError}
          isDeleting={isDeleting}
          onCancel={onCancelDeletion}
          onConfirm={onConfirmDeletion}
        />
      ) : null}

      {document.chunks.length === 0 ? (
        <p className="mt-8 rounded-xl bg-slate-50 px-5 py-6 text-sm text-slate-600">
          This document has no stored chunks.
        </p>
      ) : (
        <ol className="mt-8 space-y-5">
          {document.chunks.map((chunk) => (
            <ChunkCard
              key={chunk.chunk_id}
              chunk={chunk}
              filename={document.filename}
            />
          ))}
        </ol>
      )}
    </div>
  );
}

interface DeleteConfirmationProps {
  document: DocumentDetail;
  error: string | null;
  isDeleting: boolean;
  onCancel: () => void;
  onConfirm: () => void;
}

function DeleteConfirmation({
  document,
  error,
  isDeleting,
  onCancel,
  onConfirm,
}: DeleteConfirmationProps) {
  const cancelButtonRef = useRef<HTMLButtonElement>(null);

  useEffect(() => {
    cancelButtonRef.current?.focus();
  }, []);

  return (
    <section
      className="mt-8 rounded-2xl border border-red-200 bg-red-50 p-5 sm:p-6"
      role="alertdialog"
      aria-modal="false"
      aria-labelledby="delete-confirmation-title"
      aria-describedby="delete-confirmation-description"
    >
      <p className="text-xs font-bold uppercase tracking-[0.16em] text-red-700">
        Confirm deletion
      </p>
      <h3
        id="delete-confirmation-title"
        className="mt-2 text-lg font-semibold text-slate-950"
      >
        Delete {document.filename}?
      </h3>
      <p
        id="delete-confirmation-description"
        className="mt-2 text-sm leading-6 text-slate-700"
      >
        This removes the document and all of its stored chunks. This action cannot
        be undone.
      </p>

      {error !== null ? (
        <p
          className="mt-4 rounded-xl border border-red-200 bg-white px-4 py-3 text-sm text-red-800"
          role="alert"
        >
          {error}
        </p>
      ) : null}

      {isDeleting ? (
        <p className="mt-4 text-sm text-slate-600" role="status" aria-live="polite">
          Deleting document…
        </p>
      ) : null}

      <div className="mt-6 flex flex-wrap gap-3">
        <button
          ref={cancelButtonRef}
          type="button"
          disabled={isDeleting}
          onClick={onCancel}
          className="rounded-xl border border-slate-300 bg-white px-4 py-2.5 text-sm font-semibold text-slate-800 hover:bg-slate-50 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-brand-600 disabled:cursor-wait disabled:opacity-60"
        >
          Keep document
        </button>
        <button
          type="button"
          disabled={isDeleting}
          onClick={onConfirm}
          className="rounded-xl bg-red-700 px-4 py-2.5 text-sm font-semibold text-white hover:bg-red-800 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-red-700 disabled:cursor-wait disabled:bg-red-300"
        >
          {isDeleting ? "Deleting…" : "Delete permanently"}
        </button>
      </div>
    </section>
  );
}

function ChunkCard({ chunk, filename }: { chunk: DocumentChunk; filename: string }) {
  const provenance = Object.entries(chunk.metadata).filter(
    ([key]) => key !== "filename",
  );

  return (
    <li className="rounded-2xl border border-slate-200 bg-slate-50/70 p-5">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <p className="text-xs font-bold uppercase tracking-[0.14em] text-slate-500">
          Chunk {chunk.chunk_index}
        </p>
        <p className="break-all text-xs text-slate-500">Source: {filename}</p>
      </div>

      {provenance.length > 0 ? (
        <dl className="mt-4 flex flex-wrap gap-2" aria-label="Chunk provenance">
          {provenance.map(([key, value]) => (
            <div
              key={key}
              className="inline-flex rounded-full border border-slate-200 bg-white px-3 py-1.5 text-xs"
            >
              <dt className="font-semibold text-slate-600">{metadataLabel(key)}:</dt>
              <dd className="ml-1 text-slate-800">{value}</dd>
            </div>
          ))}
        </dl>
      ) : null}

      <p className="mt-5 whitespace-pre-wrap text-sm leading-7 text-slate-800">
        {chunk.text}
      </p>
    </li>
  );
}

function ListLoading() {
  return (
    <div
      className="rounded-2xl border border-slate-200 bg-white px-6 py-14 text-center"
      role="status"
      aria-live="polite"
    >
      <p className="text-sm font-semibold text-slate-900">Loading documents…</p>
      <p className="mt-2 text-sm text-slate-500">Reading the source catalog.</p>
    </div>
  );
}

function ListError({ message, onRetry }: { message: string; onRetry: () => void }) {
  return (
    <div className="rounded-2xl border border-red-200 bg-red-50 px-6 py-8" role="alert">
      <p className="text-sm font-semibold text-red-800">Documents could not be loaded</p>
      <p className="mt-2 text-sm leading-6 text-slate-700">{message}</p>
      <button
        type="button"
        onClick={onRetry}
        className="mt-5 rounded-xl border border-red-300 bg-white px-4 py-2.5 text-sm font-semibold text-red-800 hover:bg-red-100 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-red-700"
      >
        Refresh documents
      </button>
    </div>
  );
}

function EmptyCatalog() {
  return (
    <div className="rounded-2xl border border-dashed border-slate-300 bg-white px-6 py-14 text-center">
      <p className="text-sm font-semibold text-slate-900">No documents yet</p>
      <p className="mx-auto mt-2 max-w-md text-sm leading-6 text-slate-500">
        Upload a source document first, then return here to inspect its chunks and
        provenance.
      </p>
      <Link
        to="/upload"
        className="mt-6 inline-flex min-h-11 items-center rounded-xl bg-brand-600 px-5 py-3 text-sm font-semibold text-white hover:bg-brand-700 focus-visible:outline-2 focus-visible:outline-offset-3 focus-visible:outline-brand-600"
      >
        Upload a document
      </Link>
    </div>
  );
}

function shortenId(documentId: string): string {
  return documentId.length <= 16
    ? documentId
    : `${documentId.slice(0, 9)}…${documentId.slice(-5)}`;
}

function metadataLabel(key: string): string {
  const knownLabels: Record<string, string> = {
    page_number: "Page",
    section_type: "Section",
    heading_style: "Heading style",
  };
  return knownLabels[key] ?? key.replaceAll("_", " ");
}

function toFriendlyMessage(error: unknown): string {
  return error instanceof DocumentApiError
    ? error.message
    : "The document catalog could not be loaded. Please try again.";
}

function toDeletionMessage(error: unknown): string {
  return error instanceof DocumentApiError
    ? error.message
    : "The document could not be deleted. Please try again.";
}
