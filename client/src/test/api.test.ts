import { describe, it, expect, vi, beforeEach } from "vitest";
import {
  fetchServerStats,
  fetchPipelines,
  fetchSessions,
  fetchSession,
  createSession,
  updateSession,
  deleteSession,
} from "../api/client.ts";

function mockFetch(body: unknown, status = 200) {
  return vi.fn().mockResolvedValue({
    ok: status >= 200 && status < 300,
    status,
    json: () => Promise.resolve(body),
    text: () => Promise.resolve(JSON.stringify(body)),
  });
}

describe("api client", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  it("fetchServerStats calls /api/stats", async () => {
    const data = { uptime_seconds: 10, active_sessions: 0, total_sessions: 0, total_bytes_processed: 0, available_pipelines: 1 };
    globalThis.fetch = mockFetch(data);
    const result = await fetchServerStats();
    expect(result).toEqual(data);
    expect(fetch).toHaveBeenCalledWith("/api/stats");
  });

  it("fetchPipelines calls /api/pipelines", async () => {
    const data = [{ id: "echo", name: "Echo", description: "test" }];
    globalThis.fetch = mockFetch(data);
    const result = await fetchPipelines();
    expect(result).toEqual(data);
  });

  it("fetchSessions calls /api/sessions", async () => {
    globalThis.fetch = mockFetch([]);
    const result = await fetchSessions();
    expect(result).toEqual([]);
  });

  it("fetchSession calls /api/sessions/:id", async () => {
    const session = { id: "abc", pipeline_id: "echo", label: "", status: "created", sample_rate: 48000, channels: 1, created_at: 0, stats: {} };
    globalThis.fetch = mockFetch(session);
    const result = await fetchSession("abc");
    expect(result.id).toBe("abc");
  });

  it("createSession posts to /api/sessions", async () => {
    const session = { id: "new", pipeline_id: "echo", label: "", status: "created", sample_rate: 48000, channels: 1, created_at: 0, stats: {} };
    globalThis.fetch = mockFetch(session, 201);
    const result = await createSession({ pipeline_id: "echo" });
    expect(result.id).toBe("new");
    expect(fetch).toHaveBeenCalledWith(
      "/api/sessions",
      expect.objectContaining({ method: "POST" }),
    );
  });

  it("updateSession patches /api/sessions/:id", async () => {
    const session = { id: "abc", pipeline_id: "echo", label: "renamed", status: "created", sample_rate: 48000, channels: 1, created_at: 0, stats: {} };
    globalThis.fetch = mockFetch(session);
    const result = await updateSession("abc", { label: "renamed" });
    expect(result.label).toBe("renamed");
  });

  it("deleteSession deletes /api/sessions/:id", async () => {
    globalThis.fetch = mockFetch(null, 204);
    await deleteSession("abc");
    expect(fetch).toHaveBeenCalledWith(
      "/api/sessions/abc",
      expect.objectContaining({ method: "DELETE" }),
    );
  });

  it("throws on error response", async () => {
    globalThis.fetch = mockFetch("Not Found", 404);
    await expect(fetchSession("nope")).rejects.toThrow("404");
  });
});
