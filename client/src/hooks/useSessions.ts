import { useCallback, useEffect, useRef, useState } from "react";
import { fetchSessions } from "../api/index.ts";
import type { Session } from "../api/index.ts";

export function useSessions(intervalMs = 3000) {
  const [sessions, setSessions] = useState<Session[]>([]);
  const timer = useRef<ReturnType<typeof setInterval>>(undefined);

  const refresh = useCallback(async () => {
    try {
      setSessions(await fetchSessions());
    } catch {
      // server offline
    }
  }, []);

  useEffect(() => {
    refresh();
    timer.current = setInterval(refresh, intervalMs);
    return () => clearInterval(timer.current);
  }, [refresh, intervalMs]);

  return { sessions, refresh };
}
