import { useCallback, useEffect, useRef, useState } from "react";
import { fetchServerStats } from "../api/index.ts";
import type { ServerStats } from "../api/index.ts";

export function useServerStats(intervalMs = 2000) {
  const [stats, setStats] = useState<ServerStats | null>(null);
  const timer = useRef<ReturnType<typeof setInterval>>(undefined);

  const refresh = useCallback(async () => {
    try {
      setStats(await fetchServerStats());
    } catch {
      // server offline
    }
  }, []);

  useEffect(() => {
    refresh();
    timer.current = setInterval(refresh, intervalMs);
    return () => clearInterval(timer.current);
  }, [refresh, intervalMs]);

  return stats;
}
