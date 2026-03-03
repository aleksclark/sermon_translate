# Phase 7 — Risks & Decisions

## Key Design Decisions

### D1: Codegen over OpenAPI

**Decision:** Extend `codegen.py` to emit Dart directly, rather than generating an OpenAPI spec and using a generic OpenAPI→Dart generator.

**Rationale:**
- The existing codegen is 170 lines and already walks Pydantic schemas. Adding Dart output is ~100 more lines.
- OpenAPI generators produce verbose, opinionated code that doesn't match our Freezed style.
- Single source of truth stays in one file, not an intermediate spec.
- The `--check` staleness flag works identically for both TS and Dart.

**Trade-off:** If a third client language is needed, the codegen grows. At that point, consider switching to OpenAPI as the intermediate format.

### D2: Riverpod over BLoC

**Decision:** Use Riverpod for state management.

**Rationale:**
- The web client uses React hooks. Riverpod providers map 1:1 to hooks (see [Phase 4](./04-state.md)).
- BLoC requires event classes, state classes, and `mapEventToState` — ceremony that doesn't match the existing architecture's simplicity.
- Riverpod's `autoDispose` handles cleanup automatically, matching React's `useEffect` return function.
- The team already thinks in "hooks" — Riverpod preserves that mental model.

**Trade-off:** Riverpod has a steeper initial learning curve than `setState` or BLoC for Flutter newcomers.

### D3: `record` for mic capture

**Decision:** Use the `record` package for microphone input.

**Rationale:**
- Supports raw PCM output on all 6 platforms (macOS, Linux, Windows, iOS, Android, web).
- Single API for device enumeration and stream recording.
- Actively maintained (v5, 1000+ pub.dev likes).
- No FFI or platform channel code needed.

**Trade-off:** Chunk size and sample rate control vary by platform. If a platform delivers chunks at a different rate, the server handles it gracefully (it concatenates all incoming audio).

### D4: FFmpeg Kit for MP3 decode

**Decision:** Use `ffmpeg_kit_flutter_audio` for MP3 → PCM decoding.

**Rationale:**
- Reliable codec support across all platforms, identical behavior to the `ffmpeg` CLI used for the test fixture.
- No custom FFI bindings needed.
- Audio-only variant keeps binary size reasonable (~15MB).

**Trade-off:** 15MB added to bundle size. If this is unacceptable, the fallback is a minimp3 FFI binding (~50KB) which requires writing and maintaining platform-specific build scripts.

### D5: Streams over callbacks

**Decision:** The Dart `StreamTransport` interface exposes `Stream<Uint8List>` and `Stream<TransportEvent>` instead of callback registration methods (`onAudio`, `onEvent`, `onClose`).

**Rationale:**
- Dart idiom: streams compose naturally with `async for`, `StreamSubscription`, and Riverpod.
- Avoids the callback registration pattern which is more JavaScript-idiomatic.
- `StreamController.broadcast()` supports multiple listeners (e.g., both the audio player and the state notifier listen to `audioStream`).

**Trade-off:** Slightly different interface from the web client and server-side `TransportConnection`. Acceptable since client and server transports are already different (browser `WebSocket` vs Starlette `WebSocket`).

---

## Risk Areas

### R1: MP3 decode on desktop Linux

**Risk:** `just_audio` uses GStreamer on Linux, which may not have MP3 codecs installed in minimal environments.

**Mitigation:**
- Primary path: `ffmpeg_kit_flutter_audio` bundles its own codecs, no system dependency.
- Fallback: If FFmpeg Kit doesn't support Linux desktop, shell out to `ffmpeg` CLI (which is available on most Linux dev machines).
- Document the requirement in README: `sudo apt install ffmpeg` for Linux.

**Likelihood:** Medium. **Impact:** Low (only affects one platform, easy workaround).

### R2: Raw PCM mic capture sample rate

**Risk:** The `record` package may not deliver PCM at exactly 48kHz on all platforms. Some platforms may resample to their native rate (44.1kHz on macOS, 16kHz on some Android devices).

**Mitigation:**
- Query the actual sample rate from the `RecordConfig` after starting.
- If it differs from the session's 48kHz, either:
  - Pass the actual rate in `SessionCreate.sample_rate` (simplest — server downsamples from any rate to 16kHz for Whisper), or
  - Add a client-side linear resampler matching the server's `_downsample` function.

**Likelihood:** Medium. **Impact:** Low (server's downsample handles any input rate).

### R3: Integration test file picker

**Risk:** `FilePicker` shows a native OS dialog that can't be automated in integration tests.

**Mitigation:**
- Inject a `FilePickerPlatform` mock in test setup that returns the fixture path.
- Alternatively, bypass the dialog entirely: the test creates a `FileSource` directly and injects it into the provider.
- The web client's e2e test solved the same problem with Playwright's `page.waitForEvent('filechooser')` — Flutter's `MethodChannel` mock is the equivalent.

**Likelihood:** High (this will definitely need handling). **Impact:** Low (well-known testing pattern).

### R4: WebSocket binary frame handling on web

**Risk:** If the Flutter client is ever compiled to web (`flutter build web`), `web_socket_channel` uses the browser's `WebSocket` which handles `Blob` vs `ArrayBuffer` differently.

**Mitigation:**
- Not a current target (we're building native specifically to avoid the browser).
- If web support is added later, `web_socket_channel` handles the platform difference internally.

**Likelihood:** Low. **Impact:** None (out of scope).

### R5: Audio playback latency

**Risk:** Playing received PCM through `just_audio` may introduce buffering latency, making the echo pipeline feel sluggish.

**Mitigation:**
- For the echo pipeline (testing tool), latency is acceptable — it already has a 5-second delay.
- For Spanish translation (production), the TTS synthesis + network round-trip dominates latency.
- If sub-100ms playback latency is needed later, use platform channels to write directly to CoreAudio / WASAPI / PulseAudio output buffers.

**Likelihood:** Medium. **Impact:** Low (not noticeable given existing pipeline latency).

### R6: First-run model download time

**Risk:** The Whisper and translation models download on first session start. This is a server-side concern, not client-side, but it affects the user experience when the Flutter client is used for the first time against a fresh server.

**Mitigation:**
- Already handled server-side (model loading is async, WebSocket stays connected during load).
- Client shows "Streaming" badge immediately on WS connect; transcript appears after model loads and processes audio.
- Could add a "Loading model..." indicator by watching for the first `session.stats` event (which starts arriving immediately from `stats_loop`).

**Likelihood:** High (first run). **Impact:** Low (one-time, ~30s delay).

---

## Open Questions

These can be resolved during implementation:

1. **Output device selection**: The web client has an output device dropdown. Should the Flutter client replicate this, or use system default? (Recommendation: system default for v1, add device selection later if requested.)

2. **Theme persistence**: Should dark/light preference be persisted across app restarts? (Recommendation: yes, use `shared_preferences`.)

3. **Server URL configuration**: Should there be a settings screen for the server URL, or is compile-time `--dart-define` sufficient? (Recommendation: settings screen for production, `--dart-define` for development.)

4. **Offline handling**: Should the app show a reconnection UI when the server goes down? (Recommendation: yes, the polling providers already handle this — show "Server offline" in the stats panel, same as web client.)
