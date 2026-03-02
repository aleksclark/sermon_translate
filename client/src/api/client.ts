import type {
  PipelineInfo,
  ServerStats,
  Session,
  SessionCreate,
} from "./types.gen.ts";

const BASE = "/api";

async function json<T>(res: Response): Promise<T> {
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`${res.status}: ${text}`);
  }
  return res.json() as Promise<T>;
}

export async function fetchServerStats(): Promise<ServerStats> {
  return json<ServerStats>(await fetch(`${BASE}/stats`));
}

export async function fetchPipelines(): Promise<PipelineInfo[]> {
  return json<PipelineInfo[]>(await fetch(`${BASE}/pipelines`));
}

export async function fetchSessions(): Promise<Session[]> {
  return json<Session[]>(await fetch(`${BASE}/sessions`));
}

export async function fetchSession(id: string): Promise<Session> {
  return json<Session>(await fetch(`${BASE}/sessions/${id}`));
}

export async function createSession(req: SessionCreate): Promise<Session> {
  return json<Session>(
    await fetch(`${BASE}/sessions`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(req),
    }),
  );
}

export async function updateSession(
  id: string,
  body: { label?: string; status?: string },
): Promise<Session> {
  return json<Session>(
    await fetch(`${BASE}/sessions/${id}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }),
  );
}

export async function deleteSession(id: string): Promise<void> {
  const res = await fetch(`${BASE}/sessions/${id}`, { method: "DELETE" });
  if (!res.ok) throw new Error(`${res.status}`);
}
