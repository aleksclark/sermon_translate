# Phase 6 — Testing & CI

## Overview

Port the existing test suite (20 unit + 10 e2e) to Dart, then add a GitHub Actions workflow for the Flutter client.

## 6.1 Unit Tests

### Test mapping

The web client has 20 unit tests across 3 files. Each maps directly to a Dart test.

#### Model serialization tests (port of `types.test.ts` — 6 tests)

```
test/models/models_test.dart

1. SessionStatus enum serializes to/from JSON strings
2. Session round-trips through JSON (fromJson → toJson → fromJson)
3. SessionCreate with defaults produces correct JSON
4. SessionCreate with explicit values preserves all fields
5. ServerStats round-trips through JSON
6. PipelineInfo round-trips through JSON
```

Use Freezed's generated `fromJson`/`toJson` directly:

```dart
test('Session round-trips through JSON', () {
  final session = Session(
    id: 'abc123',
    pipelineId: 'echo',
    label: 'Test',
    status: SessionStatus.active,
    sampleRate: 48000,
    channels: 1,
    createdAt: 1700000000.0,
    stats: const SessionStats(),
  );
  final json = session.toJson();
  final restored = Session.fromJson(json);
  expect(restored, equals(session));
});
```

#### API client tests (port of `api.test.ts` — 8 tests)

```
test/api/client_test.dart

1. fetchServerStats returns typed ServerStats
2. fetchPipelines returns list of PipelineInfo
3. createSession sends POST with correct body, returns Session
4. fetchSession returns Session for valid ID
5. fetchSession throws ApiException(404) for missing ID
6. updateSession sends PATCH with correct body
7. deleteSession sends DELETE, no response body
8. createSession with unknown pipeline throws ApiException(400)
```

Mock strategy: use `mocktail` to mock Dio's `HttpClientAdapter`, returning canned JSON responses:

```dart
class MockHttpClientAdapter extends Mock implements HttpClientAdapter {}

setUp(() {
  mockAdapter = MockHttpClientAdapter();
  dio = Dio()..httpClientAdapter = mockAdapter;
  client = ApiClient.withDio(dio);
});
```

#### Transport tests (port of `transport.test.ts` — 6 tests)

```
test/transport/ws_transport_test.dart

1. sendAudio prepends 0x01 tag byte
2. sendEvent prepends 0x02 tag byte with JSON payload
3. Received audio frame (0x01 + bytes) emitted on audioStream
4. Received event frame (0x02 + JSON) parsed and emitted on eventStream
5. WebSocket close emitted on closeStream
6. sendAudio with empty Uint8List sends 1-byte EOF frame
```

Mock strategy: use `StreamChannel` from `package:stream_channel` to create an in-memory WebSocket pair:

```dart
late StreamChannel<List<int>> channel;
late WebSocketTransport transport;

setUp(() {
  final controller = StreamChannelController<List<int>>();
  channel = controller.foreign;
  transport = WebSocketTransport.fromChannel(controller.local);
});
```

This requires adding a `WebSocketTransport.fromChannel()` constructor for testing — the production constructor uses `WebSocketChannel.connect()`.

### Running unit tests

```bash
cd native-client
flutter test
```

Expected: 20 tests pass.

## 6.2 Integration Tests

### Test mapping

The web client has 10 Playwright e2e tests across 5 files. The Flutter integration tests cover the same scenarios.

```
integration_test/

app_shell_test.dart (3 tests — port of app-shell.spec.ts)
  1. loads and shows title
  2. shows server stats panel
  3. shows sessions panel

session_crud_test.dart (1 test — port of session-crud.spec.ts)
  1. create, read, update, delete session (API-level, via ApiClient)

session_dialog_test.dart (2 tests — port of session-modal.spec.ts)
  1. opens dialog from + button and shows form
  2. cancel closes dialog

api_stats_test.dart (2 tests — port of api-stats.spec.ts)
  1. GET /api/stats returns valid stats
  2. GET /api/pipelines lists pipelines

mp3_upload_test.dart (1 test — port of mp3-upload.spec.ts)
  1. select MP3 source, upload file, start session, verify transcript

app_theme_test.dart (1 test — port of app-shell.spec.ts dark mode)
  1. has dark mode toggle
```

### Integration test infrastructure

Flutter integration tests run a real app instance. They need a running server.

**Option A — Docker Compose (recommended for CI):**

Reuse the existing `e2e/docker-compose.yml` server + nginx stack. The Flutter test connects to `localhost:4173`:

```bash
# Start server stack
cd e2e && docker compose up -d server web

# Run integration tests against it
cd native-client && flutter test integration_test/ \
  --dart-define=API_BASE_URL=http://localhost:4173/api

# Tear down
cd e2e && docker compose down
```

**Option B — Local server (for development):**

```bash
cd server && uv run uvicorn src.main:app --port 8000 &
cd native-client && flutter test integration_test/
```

### MP3 upload test

The MP3 test is simpler in Flutter than in Playwright because:
- No headless Chrome `AudioContext` limitations
- `FileSource` uses native FFmpeg, which always works
- No browser mocking needed

```dart
testWidgets('MP3 upload streams and shows transcript', (tester) async {
  app.main();
  await tester.pumpAndSettle();

  // Open new session dialog
  await tester.tap(find.byIcon(Icons.add));
  await tester.pumpAndSettle();

  // Select Whisper TTS pipeline
  await tester.tap(find.byType(DropdownButtonFormField<String>));
  await tester.pumpAndSettle();
  await tester.tap(find.text('Whisper TTS').last);
  await tester.pumpAndSettle();

  // Switch to MP3 File source
  await tester.tap(find.text('MP3 File'));
  await tester.pumpAndSettle();

  // Pick file (need to mock FilePicker in integration tests)
  // Use a test fixture bundled as an asset
  await _selectTestFixture(tester, 'test-speech.mp3');

  // Start session
  await tester.tap(find.text('Start'));
  await tester.pumpAndSettle();

  // Wait for transcript
  await tester.pumpAndSettle(const Duration(seconds: 30));
  expect(find.textContaining('grace'), findsOneWidget);
});
```

**File picker mocking:** In integration tests, `FilePicker` can't show a native dialog. Use a test-specific `FileSource` that reads from a bundled asset, or set up a `MethodChannel` mock that returns the fixture path.

### Test fixture

Copy the existing `e2e/fixtures/test-speech.mp3` to `native-client/integration_test/fixtures/test-speech.mp3`. Reference it via `rootBundle` or direct file path depending on the platform.

## 6.3 CI Workflow

### `.github/workflows/flutter.yml`

```yaml
name: Flutter CI

on:
  push:
    branches: [master]
    paths: [native-client/**, server/src/codegen.py, server/src/models/**]
  pull_request:
    branches: [master]
    paths: [native-client/**, server/src/codegen.py, server/src/models/**]

defaults:
  run:
    working-directory: native-client

jobs:
  analyze-and-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - uses: subosito/flutter-action@v2
        with:
          channel: stable
          cache: true

      - name: Install dependencies
        run: flutter pub get

      - name: Generate code
        run: dart run build_runner build --delete-conflicting-outputs

      - name: Analyze
        run: flutter analyze

      - name: Unit tests
        run: flutter test

      # Codegen staleness check
      - uses: astral-sh/setup-uv@v6
        with:
          version: latest

      - name: Check Dart types are up to date
        working-directory: server
        run: uv run python -m src.codegen --dart --check

  integration:
    runs-on: ubuntu-latest
    needs: analyze-and-test
    steps:
      - uses: actions/checkout@v4

      - uses: subosito/flutter-action@v2
        with:
          channel: stable
          cache: true

      - name: Install dependencies
        run: flutter pub get

      - name: Generate code
        run: dart run build_runner build --delete-conflicting-outputs

      - name: Start server stack
        working-directory: e2e
        run: docker compose up -d server web

      - name: Wait for server
        run: |
          for i in $(seq 1 30); do
            curl -sf http://localhost:4173/api/stats && break
            sleep 2
          done

      - name: Integration tests
        run: |
          flutter test integration_test/ \
            --dart-define=API_BASE_URL=http://localhost:4173/api

      - name: Teardown
        if: always()
        working-directory: e2e
        run: docker compose down
```

### CI summary

| Job | What runs | Duration est. |
|---|---|---|
| `analyze-and-test` | `flutter analyze` + 20 unit tests + codegen check | ~2 min |
| `integration` | Docker server + 10 integration tests | ~5 min |

## 6.4 AGENTS.md Updates

Add to the existing `AGENTS.md`:

```markdown
### Flutter Client (`cd native-client`)

flutter analyze                    # lint + static analysis
flutter test                       # unit tests
flutter test integration_test/     # integration tests (needs server running)
dart run build_runner build        # regenerate freezed/json code
```

## 6.5 Verification

After this phase:

- `flutter test` — 20 unit tests pass
- `flutter test integration_test/` — 10 integration tests pass (with server running)
- GitHub Actions workflow runs on PR and push to master

Next: [Phase 7 — Risks & Decisions](./07-risks-and-decisions.md)
