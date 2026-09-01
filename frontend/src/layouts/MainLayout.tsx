import type { ReactNode } from "react";
import { Flame } from "lucide-react";

interface MainLayoutProps {
  children: ReactNode;
}

/**
 * Application shell. TODO: add navigation, alerts indicator, and
 * a persistent sidebar once the dashboard pages are built.
 */
export function MainLayout({ children }: MainLayoutProps) {
  return (
    <div className="flex min-h-screen flex-col bg-slate-950 text-slate-100">
      <header className="flex items-center gap-2 border-b border-slate-800 px-6 py-4">
        <Flame className="h-6 w-6 text-orange-500" aria-hidden="true" />
        <h1 className="text-lg font-semibold tracking-tight">
          Industrial Fire Intelligence Platform
        </h1>
      </header>
      <main className="flex-1 px-6 py-8">{children}</main>
    </div>
  );
}
