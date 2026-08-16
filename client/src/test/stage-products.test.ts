import { describe, it, expect } from "vitest";
import { parseStageProductEvent } from "../hooks/useAudioStream.ts";
import type { TransportEvent } from "../transport/index.ts";

describe("parseStageProductEvent", () => {
  it("parses a listen stage.product event", () => {
    const evt: TransportEvent = {
      type: "pipeline.event",
      session_id: "abc",
      payload: {
        kind: "stage.product",
        stage: "listen",
        product: {
          sequence: 0,
          utterance_id: "utt-1",
          text: "hello",
          is_final: true,
          words: [{ text: "hello", start_ms: 0, end_ms: 200, conf: 1, prosody: null }],
          language: "en",
        },
      },
    };

    const update = parseStageProductEvent(evt);
    expect(update).not.toBeNull();
    expect(update?.stage).toBe("listen");
    expect((update?.product as { text: string }).text).toBe("hello");
  });

  it("parses a translate stage.product with instructions", () => {
    const evt: TransportEvent = {
      type: "pipeline.event",
      session_id: "abc",
      payload: {
        kind: "stage.product",
        stage: "translate",
        product: {
          sequence: 1,
          source_utterance_id: "utt-1",
          target_utterance_id: "tgt-1",
          text: "hola",
          is_final: true,
          words: [],
          instructions: { hints: {}, markers: [{ word: "hola" }] },
        },
      },
    };

    const update = parseStageProductEvent(evt);
    expect(update?.stage).toBe("translate");
    expect((update?.product as { text: string }).text).toBe("hola");
  });

  it("ignores transcript events", () => {
    const evt: TransportEvent = {
      type: "pipeline.event",
      session_id: "abc",
      payload: { kind: "transcript", stream: "listen", text: "hi" },
    };
    expect(parseStageProductEvent(evt)).toBeNull();
  });

  it("ignores malformed products", () => {
    expect(
      parseStageProductEvent({
        type: "pipeline.event",
        session_id: "abc",
        payload: { kind: "stage.product", stage: "listen" },
      }),
    ).toBeNull();

    expect(
      parseStageProductEvent({
        type: "pipeline.event",
        session_id: "abc",
        payload: { kind: "stage.product", stage: "unknown", product: { text: "x" } },
      }),
    ).toBeNull();

    expect(
      parseStageProductEvent({
        type: "session.stats",
        session_id: "abc",
        payload: {},
      }),
    ).toBeNull();
  });
});
