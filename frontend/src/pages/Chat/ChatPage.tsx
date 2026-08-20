import {
  type FormEvent,
  type RefObject,
  useEffect,
  useRef,
  useState,
} from "react";

import { PageIntro } from "../../components/PageIntro";
import { queryDocuments, QueryApiError } from "../../services";
import type { ConversationItem, QueryResponse } from "../../types";

const DEFAULT_TOP_K = "5";

export function ChatPage() {
  const [question, setQuestion] = useState("");
  const [topK, setTopK] = useState(DEFAULT_TOP_K);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [conversations, setConversations] = useState<ConversationItem[]>([]);
  const [clearConfirmationOpen, setClearConfirmationOpen] = useState(false);
  const [conversationNotice, setConversationNotice] = useState<string | null>(null);
  const questionInputRef = useRef<HTMLTextAreaElement>(null);
  const conversationEndRef = useRef<HTMLDivElement>(null);
  const clearTriggerRef = useRef<HTMLButtonElement | null>(null);
  const conversationNoticeRef = useRef<HTMLDivElement>(null);
  const shouldScrollToNewest = useRef(false);

  useEffect(() => {
    if (!shouldScrollToNewest.current) return;
    conversationEndRef.current?.scrollIntoView({ block: "nearest" });
    shouldScrollToNewest.current = false;
  }, [conversations]);

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (isSubmitting) return;

    const normalizedQuestion = question.trim();
    if (!normalizedQuestion) {
      setError("Enter a question before submitting.");
      return;
    }

    const parsedTopK = Number(topK);
    if (!Number.isInteger(parsedTopK) || parsedTopK <= 0) {
      setError("Top k must be a positive whole number.");
      return;
    }

    setIsSubmitting(true);
    setError(null);
    setConversationNotice(null);
    setClearConfirmationOpen(false);
    try {
      const response = await queryDocuments(normalizedQuestion, parsedTopK);
      shouldScrollToNewest.current = true;
      setConversations((currentItems) => [
        ...currentItems,
        createConversationItem(normalizedQuestion, response),
      ]);
      setQuestion("");
      requestAnimationFrame(() =>
        questionInputRef.current?.focus({ preventScroll: true }),
      );
    } catch (requestError) {
      setError(
        requestError instanceof QueryApiError
          ? requestError.message
          : "The question could not be completed. Please try again.",
      );
    } finally {
      setIsSubmitting(false);
    }
  }

  function requestClearConversation(trigger: HTMLButtonElement) {
    clearTriggerRef.current = trigger;
    setConversationNotice(null);
    setClearConfirmationOpen(true);
  }

  function cancelClearConversation() {
    setClearConfirmationOpen(false);
    requestAnimationFrame(() => clearTriggerRef.current?.focus());
  }

  function clearConversation() {
    setConversations([]);
    setClearConfirmationOpen(false);
    setConversationNotice(
      "Conversation cleared. Your uploaded documents were not changed.",
    );
    requestAnimationFrame(() => conversationNoticeRef.current?.focus());
  }

  return (
    <div>
      <PageIntro
        eyebrow="Ask with evidence"
        title="Chat"
        description="Ask follow-up questions during this browser session and inspect the evidence behind every grounded response."
      />

      <div className="mt-12 grid items-start gap-8 lg:mt-16 lg:grid-cols-[minmax(18rem,0.78fr)_minmax(0,1.5fr)]">
        <QuestionForm
          inputRef={questionInputRef}
          question={question}
          topK={topK}
          isSubmitting={isSubmitting}
          error={error}
          onQuestionChange={setQuestion}
          onTopKChange={setTopK}
          onSubmit={handleSubmit}
        />

        <ConversationPanel
          conversations={conversations}
          pendingQuestion={isSubmitting ? question.trim() : null}
          isSubmitting={isSubmitting}
          clearConfirmationOpen={clearConfirmationOpen}
          conversationNotice={conversationNotice}
          endRef={conversationEndRef}
          noticeRef={conversationNoticeRef}
          onRequestClear={requestClearConversation}
          onCancelClear={cancelClearConversation}
          onConfirmClear={clearConversation}
        />
      </div>
    </div>
  );
}

interface QuestionFormProps {
  inputRef: RefObject<HTMLTextAreaElement | null>;
  question: string;
  topK: string;
  isSubmitting: boolean;
  error: string | null;
  onQuestionChange: (question: string) => void;
  onTopKChange: (topK: string) => void;
  onSubmit: (event: FormEvent<HTMLFormElement>) => void;
}

function QuestionForm({
  inputRef,
  question,
  topK,
  isSubmitting,
  error,
  onQuestionChange,
  onTopKChange,
  onSubmit,
}: QuestionFormProps) {
  const numericTopK = Number(topK);
  const topKIsInvalid =
    error !== null && (!Number.isInteger(numericTopK) || numericTopK <= 0);

  return (
    <form
      className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm sm:p-7 lg:sticky lg:top-8"
      onSubmit={onSubmit}
      noValidate
    >
      <div>
        <label htmlFor="question" className="block text-sm font-semibold text-slate-950">
          Your question
        </label>
        <p id="question-hint" className="mt-2 text-sm leading-6 text-slate-500">
          Ask about information contained in the documents currently available to
          SeriesRAG.
        </p>
        <textarea
          ref={inputRef}
          id="question"
          name="question"
          rows={6}
          value={question}
          disabled={isSubmitting}
          aria-describedby="question-hint"
          aria-invalid={error !== null && !question.trim()}
          onChange={(event) => onQuestionChange(event.target.value)}
          placeholder="What is the shared responsibility model?"
          className="mt-4 block w-full resize-y rounded-xl border border-slate-300 bg-white px-4 py-3 text-sm leading-6 text-slate-950 placeholder:text-slate-400 focus:border-brand-500 focus:outline-2 focus:outline-offset-2 focus:outline-brand-600 disabled:cursor-wait disabled:bg-slate-50 disabled:text-slate-500"
        />
      </div>

      <div className="mt-6 max-w-40">
        <label htmlFor="top-k" className="block text-sm font-semibold text-slate-950">
          Results to retrieve
        </label>
        <input
          id="top-k"
          name="top-k"
          type="number"
          min="1"
          step="1"
          inputMode="numeric"
          value={topK}
          disabled={isSubmitting}
          aria-invalid={topKIsInvalid}
          onChange={(event) => onTopKChange(event.target.value)}
          className="mt-3 block w-full rounded-xl border border-slate-300 bg-white px-4 py-2.5 text-sm text-slate-950 focus:border-brand-500 focus:outline-2 focus:outline-offset-2 focus:outline-brand-600 disabled:cursor-wait disabled:bg-slate-50"
        />
      </div>

      {error !== null ? (
        <p
          className="mt-6 rounded-xl border border-red-200 bg-red-50 px-4 py-3 text-sm leading-6 text-red-800"
          role="alert"
        >
          {error}
        </p>
      ) : null}

      <button
        type="submit"
        disabled={isSubmitting}
        className="mt-7 inline-flex min-h-11 w-full items-center justify-center rounded-xl bg-brand-600 px-6 py-3 text-sm font-semibold text-white shadow-sm shadow-brand-900/10 hover:bg-brand-700 focus-visible:outline-2 focus-visible:outline-offset-3 focus-visible:outline-brand-600 disabled:cursor-wait disabled:bg-slate-300 disabled:shadow-none sm:w-auto"
      >
        {isSubmitting ? "Finding an answer…" : "Ask SeriesRAG"}
      </button>
    </form>
  );
}

interface ConversationPanelProps {
  conversations: ConversationItem[];
  pendingQuestion: string | null;
  isSubmitting: boolean;
  clearConfirmationOpen: boolean;
  conversationNotice: string | null;
  endRef: RefObject<HTMLDivElement | null>;
  noticeRef: RefObject<HTMLDivElement | null>;
  onRequestClear: (trigger: HTMLButtonElement) => void;
  onCancelClear: () => void;
  onConfirmClear: () => void;
}

function ConversationPanel({
  conversations,
  pendingQuestion,
  isSubmitting,
  clearConfirmationOpen,
  conversationNotice,
  endRef,
  noticeRef,
  onRequestClear,
  onCancelClear,
  onConfirmClear,
}: ConversationPanelProps) {
  return (
    <section className="min-w-0" aria-label="Current conversation" aria-busy={isSubmitting}>
      <header className="flex flex-wrap items-end justify-between gap-4 border-b border-slate-200 pb-5">
        <div>
          <p className="text-xs font-bold uppercase tracking-[0.16em] text-brand-700">
            Current session
          </p>
          <h2 className="mt-2 text-xl font-semibold text-slate-950">
            {conversations.length} {conversations.length === 1 ? "answer" : "answers"}
          </h2>
        </div>
        {conversations.length > 0 ? (
          <button
            type="button"
            disabled={isSubmitting || clearConfirmationOpen}
            onClick={(event) => onRequestClear(event.currentTarget)}
            className="rounded-xl border border-slate-300 bg-white px-4 py-2.5 text-sm font-semibold text-slate-700 hover:border-red-200 hover:bg-red-50 hover:text-red-700 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-red-700 disabled:cursor-not-allowed disabled:opacity-50"
          >
            Clear conversation
          </button>
        ) : null}
      </header>

      {conversationNotice !== null ? (
        <div
          ref={noticeRef}
          tabIndex={-1}
          className="mt-5 rounded-xl border border-brand-100 bg-brand-50 px-4 py-3 text-sm font-medium text-brand-900 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-brand-600"
          role="status"
          aria-live="polite"
        >
          {conversationNotice}
        </div>
      ) : null}

      {clearConfirmationOpen ? (
        <ClearConversationConfirmation
          onCancel={onCancelClear}
          onConfirm={onConfirmClear}
        />
      ) : null}

      {conversations.length === 0 && !isSubmitting ? <ConversationEmptyState /> : null}

      {conversations.length > 0 ? (
        <ol className="mt-6 space-y-7">
          {conversations.map((item, index) => (
            <ConversationCard key={item.id} item={item} index={index} />
          ))}
        </ol>
      ) : null}

      {isSubmitting && pendingQuestion !== null ? (
        <PendingConversation question={pendingQuestion} />
      ) : null}

      <div ref={endRef} aria-hidden="true" />
    </section>
  );
}

function ConversationEmptyState() {
  return (
    <div className="mt-6 rounded-2xl border border-dashed border-slate-300 bg-white px-6 py-16 text-center">
      <p className="text-sm font-semibold text-slate-900">Start a grounded conversation</p>
      <p className="mx-auto mt-2 max-w-sm text-sm leading-6 text-slate-500">
        Completed questions will remain here until you clear this browser session or
        refresh the page.
      </p>
    </div>
  );
}

function PendingConversation({ question }: { question: string }) {
  return (
    <article
      className="mt-7 rounded-2xl border border-brand-100 bg-brand-50/60 p-5 sm:p-7"
      role="status"
      aria-live="polite"
    >
      <p className="text-xs font-bold uppercase tracking-[0.16em] text-brand-700">
        Question in progress
      </p>
      <p className="mt-3 break-words text-sm font-medium leading-6 text-slate-700">
        {question}
      </p>
      <p className="mt-6 text-sm font-semibold text-slate-900">
        Retrieving relevant chunks…
      </p>
      <p className="mt-2 text-sm leading-6 text-slate-500">
        SeriesRAG is building grounded context for this question.
      </p>
    </article>
  );
}

function ConversationCard({ item, index }: { item: ConversationItem; index: number }) {
  const [copyFeedback, setCopyFeedback] = useState<{
    message: string;
    isError: boolean;
  } | null>(null);
  const questionTitleId = `conversation-question-${item.id}`;

  async function copyAnswer() {
    try {
      if (navigator.clipboard === undefined) {
        throw new Error("Clipboard API is unavailable");
      }
      await navigator.clipboard.writeText(item.answer);
      setCopyFeedback({ message: "Copied", isError: false });
    } catch {
      setCopyFeedback({
        message: "The answer could not be copied. Select the text and copy it manually.",
        isError: true,
      });
    }
  }

  return (
    <li>
      <article
        className="min-w-0 overflow-hidden rounded-2xl border border-slate-200 bg-white shadow-sm"
        aria-labelledby={questionTitleId}
      >
        <section className="border-b border-brand-100 bg-brand-50 px-5 py-5 sm:px-7">
          <p className="text-xs font-bold uppercase tracking-[0.16em] text-brand-700">
            Question {index + 1}
          </p>
          <h3
            id={questionTitleId}
            className="mt-3 break-words text-base font-semibold leading-7 text-slate-950"
          >
            {item.question}
          </h3>
        </section>

        <section className="px-5 py-6 sm:px-7 sm:py-7" aria-label={`Answer ${index + 1}`}>
          <div className="flex flex-wrap items-center justify-between gap-3">
            <p className="text-xs font-bold uppercase tracking-[0.16em] text-slate-400">
              Grounded answer
            </p>
            <div className="flex flex-wrap items-center justify-end gap-3">
              {copyFeedback !== null ? (
                <span
                  className={`text-xs font-medium ${
                    copyFeedback.isError ? "text-red-700" : "text-brand-700"
                  }`}
                  role={copyFeedback.isError ? "alert" : "status"}
                  aria-live="polite"
                >
                  {copyFeedback.message}
                </span>
              ) : null}
              <button
                type="button"
                onClick={() => void copyAnswer()}
                aria-label={`Copy answer ${index + 1}`}
                className="rounded-lg border border-slate-300 px-3 py-2 text-xs font-semibold text-slate-700 hover:bg-slate-50 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-brand-600"
              >
                Copy answer
              </button>
            </div>
          </div>

          <p className="mt-5 whitespace-pre-wrap break-words text-base leading-8 text-slate-900">
            {item.answer}
          </p>

          <SourceInspector sources={item.sources} conversationId={item.id} />
        </section>
      </article>
    </li>
  );
}

interface ClearConversationConfirmationProps {
  onCancel: () => void;
  onConfirm: () => void;
}

function ClearConversationConfirmation({
  onCancel,
  onConfirm,
}: ClearConversationConfirmationProps) {
  const cancelButtonRef = useRef<HTMLButtonElement>(null);

  useEffect(() => {
    cancelButtonRef.current?.focus();
  }, []);

  return (
    <section
      className="mt-5 rounded-2xl border border-red-200 bg-red-50 p-5 sm:p-6"
      role="alertdialog"
      aria-modal="false"
      aria-labelledby="clear-conversation-title"
      aria-describedby="clear-conversation-description"
    >
      <p className="text-xs font-bold uppercase tracking-[0.16em] text-red-700">
        Confirm clear
      </p>
      <h3 id="clear-conversation-title" className="mt-2 text-lg font-semibold text-slate-950">
        Clear this conversation?
      </h3>
      <p id="clear-conversation-description" className="mt-2 text-sm leading-6 text-slate-700">
        Questions and answers from this browser session will be removed. Uploaded
        documents will not be deleted.
      </p>
      <div className="mt-6 flex flex-wrap gap-3">
        <button
          ref={cancelButtonRef}
          type="button"
          onClick={onCancel}
          className="rounded-xl border border-slate-300 bg-white px-4 py-2.5 text-sm font-semibold text-slate-800 hover:bg-slate-50 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-brand-600"
        >
          Keep conversation
        </button>
        <button
          type="button"
          onClick={onConfirm}
          className="rounded-xl bg-red-700 px-4 py-2.5 text-sm font-semibold text-white hover:bg-red-800 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-red-700"
        >
          Clear conversation
        </button>
      </div>
    </section>
  );
}

function SourceInspector({
  sources,
  conversationId,
}: {
  sources: ConversationItem["sources"];
  conversationId: string;
}) {
  const [expandedChunkId, setExpandedChunkId] = useState<string | null>(null);
  const sourcesTitleId = `sources-title-${conversationId}`;

  return (
    <section
      className="mt-10 border-t border-slate-200 pt-6"
      aria-labelledby={sourcesTitleId}
    >
      <div className="flex flex-wrap items-baseline justify-between gap-3">
        <h4 id={sourcesTitleId} className="text-sm font-semibold text-slate-950">
          Supporting sources
        </h4>
        <span className="text-xs text-slate-500">
          {sources.length} {sources.length === 1 ? "source" : "sources"}
        </span>
      </div>

      {sources.length === 0 ? (
        <p className="mt-4 text-sm text-slate-500">No source chunks were returned.</p>
      ) : (
        <ol className="mt-4 space-y-3">
          {sources.map((source, index) => (
            <SourceEvidence
              key={source.chunk_id}
              source={source}
              panelId={`${conversationId}-source-evidence-${index}`}
              expanded={expandedChunkId === source.chunk_id}
              onToggle={() =>
                setExpandedChunkId((currentId) =>
                  currentId === source.chunk_id ? null : source.chunk_id,
                )
              }
            />
          ))}
        </ol>
      )}
    </section>
  );
}

interface SourceEvidenceProps {
  source: ConversationItem["sources"][number];
  panelId: string;
  expanded: boolean;
  onToggle: () => void;
}

function SourceEvidence({
  source,
  panelId,
  expanded,
  onToggle,
}: SourceEvidenceProps) {
  const metadata = Object.entries(source.metadata);

  return (
    <li className="overflow-hidden rounded-xl border border-slate-200 bg-slate-50">
      <button
        type="button"
        aria-expanded={expanded}
        aria-controls={panelId}
        onClick={onToggle}
        className="w-full px-4 py-3 text-left hover:bg-slate-100 focus-visible:outline-2 focus-visible:outline-offset-[-2px] focus-visible:outline-brand-600 sm:px-5"
      >
        <span className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
          <span className="min-w-0">
            <span className="block break-words text-sm font-semibold text-slate-900">
              {source.source_name}
            </span>
            <span className="mt-1 block text-xs text-slate-500">
              Chunk {source.chunk_index}
            </span>
          </span>
          <span className="flex min-w-0 flex-wrap items-center justify-between gap-3 sm:shrink-0 sm:justify-end">
            <span className="break-all font-mono text-xs text-slate-500">
              Score {String(source.score)}
            </span>
            <span className="text-xs font-semibold text-brand-700">
              {expanded ? "Hide evidence" : "Inspect evidence"}
            </span>
          </span>
        </span>
      </button>

      {expanded ? (
        <div
          id={panelId}
          className="border-t border-slate-200 bg-white px-4 py-5 sm:px-5 sm:py-6"
        >
          <dl className="grid gap-4 text-xs sm:grid-cols-2">
            <div className="min-w-0">
              <dt className="font-bold uppercase tracking-[0.12em] text-slate-400">
                Document ID
              </dt>
              <dd
                className="mt-1 break-all font-mono text-slate-700"
                title={source.document_id}
              >
                {shortenId(source.document_id)}
              </dd>
            </div>
            <div className="min-w-0">
              <dt className="font-bold uppercase tracking-[0.12em] text-slate-400">
                Chunk ID
              </dt>
              <dd className="mt-1 break-all font-mono text-slate-700" title={source.chunk_id}>
                {shortenId(source.chunk_id)}
              </dd>
            </div>
          </dl>

          {metadata.length > 0 ? (
            <dl className="mt-5 flex flex-wrap gap-2" aria-label="Source provenance">
              {metadata.map(([key, value]) => (
                <div
                  key={key}
                  className="inline-flex min-w-0 max-w-full rounded-full border border-slate-200 bg-slate-50 px-3 py-1.5 text-xs"
                >
                  <dt className="shrink-0 font-semibold text-slate-600">
                    {metadataLabel(key)}:
                  </dt>
                  <dd className="ml-1 min-w-0 break-all text-slate-800">{value}</dd>
                </div>
              ))}
            </dl>
          ) : null}

          <div className="mt-6">
            <h5 className="text-xs font-bold uppercase tracking-[0.12em] text-slate-400">
              Full chunk text
            </h5>
            <p className="mt-3 whitespace-pre-wrap break-words text-sm leading-7 text-slate-800">
              {source.text}
            </p>
          </div>
        </div>
      ) : null}
    </li>
  );
}

function createConversationItem(
  question: string,
  response: QueryResponse,
): ConversationItem {
  return {
    id: crypto.randomUUID(),
    question,
    answer: response.answer,
    sources: response.sources,
  };
}

function shortenId(value: string): string {
  return value.length <= 20 ? value : `${value.slice(0, 11)}…${value.slice(-6)}`;
}

function metadataLabel(key: string): string {
  const knownLabels: Record<string, string> = {
    filename: "Filename",
    page_number: "Page",
    section_type: "Section",
    heading_style: "Heading style",
  };
  return knownLabels[key] ?? key.replaceAll("_", " ");
}
