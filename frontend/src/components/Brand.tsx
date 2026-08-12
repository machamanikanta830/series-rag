import { Link } from "react-router-dom";

export function Brand() {
  return (
    <Link
      to="/"
      className="group flex items-center gap-3 rounded-lg focus-visible:outline-2 focus-visible:outline-offset-4 focus-visible:outline-brand-600"
      aria-label="SeriesRAG home"
    >
      <span className="grid size-10 place-items-center rounded-xl bg-brand-600 text-sm font-bold tracking-tight text-white shadow-sm shadow-brand-900/15 group-hover:bg-brand-700">
        SR
      </span>
      <span>
        <span className="block text-base font-semibold tracking-tight text-slate-950">
          SeriesRAG
        </span>
        <span className="block text-xs text-slate-500">Grounded by design</span>
      </span>
    </Link>
  );
}
