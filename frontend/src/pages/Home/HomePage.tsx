import { PageIntro } from "../../components/PageIntro";

const stages = [
  {
    number: "01",
    title: "Ingest",
    description: "Bring source documents into one traceable collection.",
  },
  {
    number: "02",
    title: "Retrieve",
    description: "Find semantically related chunks with visible source context.",
  },
  {
    number: "03",
    title: "Ground",
    description: "Build answers from retrieved evidence instead of hidden knowledge.",
  },
] as const;

export function HomePage() {
  return (
    <div>
      <PageIntro
        eyebrow="Learning-focused RAG"
        title="Understand every step from source to answer."
        description="SeriesRAG is a transparent workspace for semantic document search powered by embeddings and vector databases."
      />

      <section className="mt-14 grid border-y border-slate-200 sm:mt-20 sm:grid-cols-3">
        {stages.map((stage, index) => (
          <article
            key={stage.number}
            className={`py-8 sm:px-6 sm:py-10 ${index > 0 ? "border-t border-slate-200 sm:border-l sm:border-t-0" : ""}`}
          >
            <p className="text-xs font-bold tracking-[0.18em] text-brand-600">
              {stage.number}
            </p>
            <h2 className="mt-5 text-xl font-semibold tracking-tight text-slate-950">
              {stage.title}
            </h2>
            <p className="mt-3 text-sm leading-6 text-slate-600">
              {stage.description}
            </p>
          </article>
        ))}
      </section>

      <section className="mt-12 flex flex-col justify-between gap-6 bg-brand-900 px-6 py-8 text-white sm:mt-16 sm:flex-row sm:items-end sm:px-8 sm:py-9">
        <div>
          <p className="text-xs font-bold uppercase tracking-[0.18em] text-brand-100">
            Current milestone
          </p>
          <h2 className="mt-3 text-2xl font-semibold tracking-tight">
            Frontend foundation
          </h2>
        </div>
        <p className="max-w-md text-sm leading-6 text-emerald-50/80">
          Navigation and page structure are ready. Upload, document browsing, and
          grounded chat will be connected in later milestones.
        </p>
      </section>
    </div>
  );
}
