import type { ReactNode } from "react";
import { Sidebar } from "./Sidebar";

export function AppShell({ children }: { children: ReactNode }) {
  return (
    <div className="flex h-screen flex-col md:flex-row">
      <Sidebar />
      <main className="flex min-w-0 flex-1 flex-col overflow-auto">
        <div className="my-auto w-full">{children}</div>
      </main>
    </div>
  );
}
