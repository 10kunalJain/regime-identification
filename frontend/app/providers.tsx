"use client";

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { useState } from "react";

/**
 * Top-level client-side providers. TanStack Query is wired in now even
 * though the live panel renders entirely from server-side data, so that
 * the future historical-explorer, method-comparison, and detection-lag
 * panels (which will hit `/regime/path?from=...&to=...` and similar)
 * can drop in without a layout-level refactor.
 */
export function Providers({ children }: { children: React.ReactNode }) {
  const [client] = useState(
    () =>
      new QueryClient({
        defaultOptions: {
          queries: {
            staleTime: 60_000,
            refetchOnWindowFocus: false,
            retry: 1,
          },
        },
      }),
  );
  return <QueryClientProvider client={client}>{children}</QueryClientProvider>;
}
