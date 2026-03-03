# Flutter Client — Implementation Plan

Native cross-platform client for the Sermon Translate platform, replacing the React web client with Flutter targeting desktop (macOS, Linux, Windows) and mobile (iOS, Android).

## Scope

The Flutter client replaces only `client/`. The Python server, binary WebSocket protocol, Docker-based e2e infrastructure, and all server-side pipelines remain unchanged.

## Quick Reference

| Existing layer | Web implementation | Flutter replacement |
|---|---|---|
| REST API (6 endpoints) | `fetch` + hand-typed TS | `dio` + generated Freezed classes |
| WebSocket transport | `WebSocket` + `Uint8Array` tagging | `web_socket_channel` + `Uint8List` tagging |
| Audio capture (mic) | `getUserMedia` + `ScriptProcessor` | `record` package (native APIs) |
| Audio capture (file) | `AudioContext.decodeAudioData` | `just_audio` or `ffmpeg_kit` |
| Types | Pydantic → `codegen.py` → TypeScript | Pydantic → `codegen.py` → Dart Freezed |
| UI framework | Mantine 8 (Material-ish) | Material 3 (native) |
| State management | React hooks + polling | Riverpod providers + polling |
| Unit tests | Vitest (20 tests) | `flutter_test` + mocktail |
| E2E tests | Playwright in Docker | `integration_test` against Docker Compose |

## Plan Structure

Each document covers one phase. Read them in order or jump to what you need.

1. **[Project Structure & Codegen](./01-structure-and-codegen.md)** — scaffold, dependencies, Dart type generation from Pydantic models
2. **[API & Transport Layer](./02-api-and-transport.md)** — REST client, WebSocket binary protocol, transport abstraction
3. **[Audio Capture & Playback](./03-audio.md)** — mic recording, MP3 file decoding, PCM streaming, output playback
4. **[State Management](./04-state.md)** — Riverpod providers, mapping from React hooks, reactive data flow
5. **[UI Widgets](./05-ui.md)** — Material 3 theme, all screens/panels, responsive layout
6. **[Testing & CI](./06-testing-and-ci.md)** — unit tests, integration tests, GitHub Actions workflow
7. **[Risks & Decisions](./07-risks-and-decisions.md)** — key design choices, platform-specific concerns, fallback strategies

## Effort Estimate

| Phase | Effort |
|---|---|
| 1. Scaffold & codegen | 4h |
| 2. API & transport | 8h |
| 3. Audio capture | 8h |
| 4. State | 4h |
| 5. UI | 8–12h |
| 6. Testing & CI | 8h |
| 7. Polish | 4h |
| **Total** | **~6 working days** |

## Wire Protocol Compatibility

The server speaks a tagged binary WebSocket protocol. The Flutter client must produce byte-identical frames:

```
Audio frame:   0x01 ++ raw_pcm_bytes (Int16LE)
Event frame:   0x02 ++ utf8_json_bytes
End-of-audio:  0x01 (1-byte frame, empty payload)
```

No server changes are required.
