interface ComingSoonPanelProps {
  step: string;
  title: string;
  description: string;
}

export function ComingSoonPanel({
  step,
  title,
  description,
}: ComingSoonPanelProps) {
  return (
    <section className="mt-12 border-y border-slate-200 py-10 sm:mt-16 sm:py-14">
      <div className="grid gap-6 sm:grid-cols-[8rem_1fr] sm:items-start">
        <p className="text-xs font-bold uppercase tracking-[0.18em] text-slate-400">
          {step}
        </p>
        <div className="max-w-2xl">
          <h2 className="text-2xl font-semibold tracking-tight text-slate-950">
            {title}
          </h2>
          <p className="mt-3 leading-7 text-slate-600">{description}</p>
          <p className="mt-6 inline-flex items-center gap-2 text-sm font-semibold text-brand-700">
            <span className="size-2 rounded-full bg-brand-500" aria-hidden="true" />
            Foundation ready
          </p>
        </div>
      </div>
    </section>
  );
}
