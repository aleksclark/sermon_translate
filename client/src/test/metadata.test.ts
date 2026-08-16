import { describe, it, expect } from "vitest";
import { parseMetadataEvent } from "../hooks/useAudioStream.ts";
import type { TransportEvent } from "../transport/index.ts";

describe("parseMetadataEvent", () => {
  it("parses a metadata pipeline.event into a typed envelope", () => {
    const evt: TransportEvent = {
      type: "pipeline.event",
      session_id: "abc",
      payload: {
        kind: "metadata",
        stream: "prosody",
        metadata: {
          schema_version: 1,
          stream: "prosody",
          kind: "prosody",
          sequence: 3,
          source_utterance_id: null,
          target_utterance_id: null,
          start_ms: 100,
          end_ms: 200,
          prosody: {
            f0_hz: 120,
            pitch_confidence: null,
            energy: 0.4,
            speaking_rate: null,
            is_pause: false,
            boundary: null,
            emphasis: null,
            confidence: 1,
            features: {},
          },
          instructions: null,
          payload: {},
        },
      },
    };

    const update = parseMetadataEvent(evt);
    expect(update).not.toBeNull();
    expect(update?.stream).toBe("prosody");
    expect(update?.envelope.kind).toBe("prosody");
    expect(update?.envelope.sequence).toBe(3);
    expect(update?.envelope.prosody?.energy).toBe(0.4);
  });

  it("ignores transcript pipeline.events", () => {
    const evt: TransportEvent = {
      type: "pipeline.event",
      session_id: "abc",
      payload: { kind: "transcript", stream: "transcript", text: "hi" },
    };
    expect(parseMetadataEvent(evt)).toBeNull();
  });

  it("ignores non-pipeline events", () => {
    const evt: TransportEvent = {
      type: "session.stats",
      session_id: "abc",
      payload: {},
    };
    expect(parseMetadataEvent(evt)).toBeNull();
  });

  it("returns null when metadata envelope is missing", () => {
    const evt: TransportEvent = {
      type: "pipeline.event",
      session_id: "abc",
      payload: { kind: "metadata", stream: "prosody" },
    };
    expect(parseMetadataEvent(evt)).toBeNull();
  });
});
