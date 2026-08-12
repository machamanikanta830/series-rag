import { Outlet } from "react-router-dom";

import { Brand } from "../components/Brand";
import { Navigation } from "../components/Navigation";

export function AppLayout() {
  return (
    <div className="min-h-screen bg-slate-50 text-slate-950 md:grid md:grid-cols-[17rem_minmax(0,1fr)]">
      <aside className="hidden min-h-screen border-r border-slate-200 bg-white px-6 py-7 md:sticky md:top-0 md:flex md:h-screen md:flex-col">
        <Brand />
        <div className="mt-12">
          <p className="mb-3 px-3 text-[0.65rem] font-bold uppercase tracking-[0.18em] text-slate-400">
            Workspace
          </p>
          <Navigation />
        </div>
        <div className="mt-auto border-t border-slate-200 pt-5">
          <p className="text-xs leading-5 text-slate-500">
            A learning-first interface for inspectable, source-grounded retrieval.
          </p>
        </div>
      </aside>

      <div className="min-w-0">
        <header className="sticky top-0 z-10 border-b border-slate-200 bg-white/95 backdrop-blur md:hidden">
          <div className="px-4 py-4 sm:px-6">
            <Brand />
          </div>
          <Navigation compact />
        </header>
        <main className="mx-auto w-full max-w-6xl px-5 py-12 sm:px-8 sm:py-16 lg:px-12 lg:py-20">
          <Outlet />
        </main>
      </div>
    </div>
  );
}
