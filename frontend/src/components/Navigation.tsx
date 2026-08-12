import { NavLink } from "react-router-dom";

const navigationItems = [
  { to: "/", label: "Home", index: "01" },
  { to: "/upload", label: "Upload", index: "02" },
  { to: "/chat", label: "Chat", index: "03" },
  { to: "/documents", label: "Documents", index: "04" },
] as const;

interface NavigationProps {
  compact?: boolean;
}

export function Navigation({ compact = false }: NavigationProps) {
  return (
    <nav aria-label="Primary navigation">
      <ul
        className={
          compact
            ? "flex gap-1 overflow-x-auto px-4 pb-3"
            : "flex flex-col gap-1.5"
        }
      >
        {navigationItems.map((item) => (
          <li key={item.to} className={compact ? "shrink-0" : undefined}>
            <NavLink
              to={item.to}
              end={item.to === "/"}
              className={({ isActive }) => {
                const layout = compact
                  ? "inline-flex min-h-10 items-center gap-2 rounded-lg px-3 text-sm"
                  : "flex min-h-11 items-center gap-3 rounded-xl px-3 text-sm";
                const state = isActive
                  ? "bg-brand-50 font-semibold text-brand-700"
                  : "font-medium text-slate-600 hover:bg-slate-100 hover:text-slate-950";
                return `${layout} ${state} focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-brand-600`;
              }}
            >
              <span className="text-[0.65rem] font-bold tracking-[0.14em] opacity-60">
                {item.index}
              </span>
              <span>{item.label}</span>
            </NavLink>
          </li>
        ))}
      </ul>
    </nav>
  );
}
