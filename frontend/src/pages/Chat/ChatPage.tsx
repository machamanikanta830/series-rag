import { type FormEvent, useState } from "react";

import { PageIntro } from "../../components/PageIntro";
import { queryDocuments, QueryApiError } from "../../services";
import type { QueryResponse } from "../../types";

const DEFAULT_TOP_K = "5";

export function ChatPage() {
  const [question, setQuestion] = useState("");
  const [topK, setTopK] = useState(DEFAULT_TOP_K);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<QueryResponse | null>(null);
  const [submittedQuestion, setSubmittedQuestion] = useState<string | null>(null);

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
    setResult(null);
    setSubmittedQuestion(null);
    try {
      const response = await queryDocuments(normalizedQuestion, parsedTopK);
      setResult(response);
      setSubmittedQuestion(normalizedQuestion);
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

  return (
    <div>
      <PageIntro
        eyebrow="Ask with evidence"
        title="Chat"
        description="Ask one question at a time and inspect the source chunks behind the grounded response."
      />

      <div className="mt-12 grid items-start gap-8 lg:mt-16 lg:grid-cols-[minmax(0,0.9fr)_minmax(0,1.35fr)]">
        <QuestionForm
          question={question}
          topK={topK}
          isSubmitting={isSubmitting}
          error={error}
          onQuestionChange={setQuestion}
          onTopKChange={setTopK}
          onSubmit={handleSubmit}
        />

        <AnswerPanel
          isSubmitting={isSubmitting}
          result={result}
          submittedQuestion={submittedQuestion}
        />
      </div>
    </div>
  );
}

interface QuestionFormProps {
  question: string;
  topK: string;
  isSubmitting: boolean;
  error: string | null;
  onQuestionChange: (question: string) => void;
  onTopKChange: (topK: string) => void;
  onSubmit: (event: FormEvent<HTMLFormElement>) => void;
}

function QuestionForm({
  question,
  topK,
  isSubmitting,
  error,
  onQuestionChange,
  onTopKChange,
  onSubmit,
}: QuestionFormProps) {
  return (
    <form
      className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm sm:p-7"
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

      <div className="mt-6 max-w-36">
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
        className="mt-7 inline-flex min-h-11 items-center justify-center rounded-xl bg-brand-600 px-6 py-3 text-sm font-semibold text-white shadow-sm shadow-brand-900/10 hover:bg-brand-700 focus-visible:outline-2 focus-visible:outline-offset-3 focus-visible:outline-brand-600 disabled:cursor-wait disabled:bg-slate-300 disabled:shadow-none"
      >
        {isSubmitting ? "Finding an answer…" : "Ask SeriesRAG"}
      </button>
    </form>
  );
}

interface AnswerPanelProps {
  isSubmitting: boolean;
  result: QueryResponse | null;
  submittedQuestion: string | null;
}

function AnswerPanel({
  isSubmitting,
  result,
  submittedQuestion,
}: AnswerPanelProps) {
  return (
    <section
      className="min-h-80 rounded-2xl border border-slate-200 bg-white p-5 sm:p-7"
      aria-label="Grounded answer"
    >
      {isSubmitting ? (
        <div className="py-16 text-center" role="status" aria-live="polite">
          <p className="text-sm font-semibold text-slate-900">
            Retrieving relevant chunks…
          </p>
          <p className="mt-2 text-sm leading-6 text-slate-500">
            SeriesRAG is building grounded context for this question.
          </p>
        </div>
      ) : null}

      {!isSubmitting && result === null ? <AnswerEmptyState /> : null}

      {!isSubmitting && result !== null && submittedQuestion !== null ? (
        <div role="status" aria-live="polite">
          <p className="text-xs font-bold uppercase tracking-[0.16em] text-brand-700">
            Grounded answer
          </p>
          <p className="mt-3 text-sm font-medium leading-6 text-slate-500">
            {submittedQuestion}
          </p>
          <p className="mt-6 whitespace-pre-wrap text-base leading-8 text-slate-900">
            {result.answer}
          </p>

          <BasicSourceList sources={result.sources} />
        </div>
      ) : null}
    </section>
  );
}

function AnswerEmptyState() {
  return (
    <div className="py-16 text-center">
      <p className="text-sm font-semibold text-slate-900">Your answer will appear here</p>
      <p className="mx-auto mt-2 max-w-sm text-sm leading-6 text-slate-500">
        The response will include a basic list of the retrieved source chunks.
      </p>
    </div>
  );
}

function BasicSourceList({ sources }: { sources: QueryResponse["sources"] }) {
  return (
    <section
      className="mt-10 border-t border-slate-200 pt-6"
      aria-labelledby="sources-title"
    >
      <div className="flex items-baseline justify-between gap-4">
        <h2 id="sources-title" className="text-sm font-semibold text-slate-950">
          Retrieved sources
        </h2>
        <span className="text-xs text-slate-500">
          {sources.length} {sources.length === 1 ? "source" : "sources"}
        </span>
      </div>

      {sources.length === 0 ? (
        <p className="mt-4 text-sm text-slate-500">No source chunks were returned.</p>
      ) : (
        <ol className="mt-4 space-y-3">
          {sources.map((source, index) => (
            <li
              key={`${source.chunk_id}-${index}`}
              className="rounded-xl border border-slate-200 bg-slate-50 px-4 py-3"
            >
              <div className="flex flex-wrap items-center justify-between gap-2">
                <p className="break-words text-sm font-semibold text-slate-900">
                  {source.source_name}
                </p>
                <p className="font-mono text-xs text-slate-500">
                  Score {source.score.toFixed(4)}
                </p>
              </div>
              <p className="mt-2 text-xs text-slate-500">
                Chunk {source.chunk_index}
              </p>
            </li>
          ))}
        </ol>
      )}
    </section>
  );
}
