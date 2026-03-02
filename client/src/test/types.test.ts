import { describe, it, expect } from "vitest";
import type {
  Session,
  SessionCreate,
  ServerStats,
  PipelineInfo,
  SessionStatus,
  SessionUpdate,
} from "../api/types.gen.ts";

describe("generated types", () => {
  it("Session has required fields at runtime shape", () => {
    const session: Session = {
      id: "abc",
      pipeline_id: "echo",
      label: "",
      status: "created",
      sample_rate: 48000,
      channels: 1,
      created_at: Date.now(),
      stats: {
        bytes_received: 0,
        bytes_sent: 0,
        chunks_received: 0,
        chunks_sent: 0,
        duration_seconds: 0,
        pipeline_latency_ms: 0,
      },
    };
    expect(session.id).toBe("abc");
    expect(session.status).toBe("created");
  });

  it("SessionCreate allows optional fields", () => {
    const req: SessionCreate = { pipeline_id: "echo" };
    expect(req.pipeline_id).toBe("echo");
    expect(req.sample_rate).toBeUndefined();
  });

  it("SessionUpdate allows all optional", () => {
    const upd: SessionUpdate = {};
    expect(upd.label).toBeUndefined();
    expect(upd.status).toBeUndefined();
  });

  it("ServerStats has required numeric fields", () => {
    const stats: ServerStats = {
      uptime_seconds: 10,
      active_sessions: 1,
      total_sessions: 5,
      total_bytes_processed: 1024,
      available_pipelines: 2,
    };
    expect(stats.active_sessions).toBe(1);
  });

  it("PipelineInfo has required fields", () => {
    const p: PipelineInfo = { id: "echo", name: "Echo", description: "test" };
    expect(p.id).toBe("echo");
  });

  it("SessionStatus union accepts valid values", () => {
    const statuses: SessionStatus[] = ["created", "active", "paused", "closed"];
    expect(statuses).toHaveLength(4);
  });
});
