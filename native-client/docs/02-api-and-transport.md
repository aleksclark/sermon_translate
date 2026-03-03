# Phase 2 — API & Transport Layer

## Overview

Port the REST client and WebSocket transport from TypeScript to Dart. These two modules are the entire interface between the Flutter client and the server.

## 2.1 REST Client

**Replaces:** `client/src/api/client.ts`

The web client has 6 functions. The Dart client mirrors them exactly using `dio`.

### Interface

```dart
// api/client.dart

class ApiClient {
  final Dio _dio;

  ApiClient({required String baseUrl})
      : _dio = Dio(BaseOptions(baseUrl: baseUrl));

  Future<ServerStats> fetchServerStats();
  Future<List<PipelineInfo>> fetchPipelines();
  Future<List<Session>> fetchSessions();
  Future<Session> fetchSession(String id);
  Future<Session> createSession(SessionCreate request);
  Future<Session> updateSession(String id, SessionUpdate request);
  Future<void> deleteSession(String id);
}
```

### Endpoint mapping

| Method | Path | Request body | Response |
|---|---|---|---|
| `GET` | `/api/stats` | — | `ServerStats` |
| `GET` | `/api/pipelines` | — | `List<PipelineInfo>` |
| `GET` | `/api/sessions` | — | `List<Session>` |
| `GET` | `/api/sessions/{id}` | — | `Session` |
| `POST` | `/api/sessions` | `SessionCreate` JSON | `Session` (201) |
| `PATCH` | `/api/sessions/{id}` | `SessionUpdate` JSON | `Session` |
| `DELETE` | `/api/sessions/{id}` | — | — (204) |

### Error handling

Throw typed exceptions for non-2xx responses:

```dart
class ApiException implements Exception {
  final int statusCode;
  final String message;
  ApiException(this.statusCode, this.message);
}
```

Check `response.statusCode` and throw `ApiException` from a Dio interceptor. Callers catch these at the provider level (see [Phase 4](./04-state.md)).

### Base URL configuration

Default to `/api` for same-origin deployments. Accept an override for development or when the server runs on a different host:

```dart
// Reads from environment or falls back to localhost
final baseUrl = const String.fromEnvironment(
  'API_BASE_URL',
  defaultValue: 'http://localhost:8000/api',
);
```

## 2.2 Transport Abstraction

**Replaces:** `client/src/transport/base.ts`

### Interface

```dart
// transport/transport.dart

class TransportEvent {
  final String type;
  final String sessionId;
  final Map<String, dynamic> payload;

  TransportEvent({
    required this.type,
    required this.sessionId,
    this.payload = const {},
  });

  factory TransportEvent.fromJson(Map<String, dynamic> json) => TransportEvent(
    type: json['type'] as String,
    sessionId: json['session_id'] as String? ?? '',
    payload: json['payload'] as Map<String, dynamic>? ?? {},
  );

  Map<String, dynamic> toJson() => {
    'type': type,
    'session_id': sessionId,
    'payload': payload,
  };
}

abstract class StreamTransport {
  Future<void> connect();
  void sendAudio(Uint8List data);
  void sendEvent(TransportEvent event);
  Stream<Uint8List> get audioStream;
  Stream<TransportEvent> get eventStream;
  Stream<void> get closeStream;
  void close();
}
```

Note: the web client uses callback registration (`onAudio`, `onEvent`, `onClose`). The Dart version uses `Stream` — idiomatic Dart and works naturally with Riverpod.

## 2.3 WebSocket Transport

**Replaces:** `client/src/transport/ws.ts`

### Wire protocol

Identical to the web client — tagged binary frames:

```
Audio:  [0x01] [pcm_bytes...]     (Int16LE mono)
Event:  [0x02] [utf8_json...]
EOF:    [0x01]                     (1-byte, empty payload = end-of-audio)
```

### Implementation

```dart
// transport/ws_transport.dart

class WebSocketTransport implements StreamTransport {
  final String url;
  WebSocketChannel? _channel;

  final _audioController = StreamController<Uint8List>.broadcast();
  final _eventController = StreamController<TransportEvent>.broadcast();
  final _closeController = StreamController<void>.broadcast();

  static const int _audioTag = 0x01;
  static const int _eventTag = 0x02;

  WebSocketTransport(this.url);

  @override
  Future<void> connect() async {
    _channel = WebSocketChannel.connect(Uri.parse(url));
    await _channel!.ready;

    _channel!.stream.listen(
      _onMessage,
      onDone: _onDone,
      onError: (_) => _onDone(),
    );
  }

  void _onMessage(dynamic data) {
    if (data is! List<int> || data.isEmpty) return;
    final bytes = data is Uint8List ? data : Uint8List.fromList(data);
    final tag = bytes[0];
    final body = bytes.sublist(1);

    if (tag == _audioTag) {
      _audioController.add(body);
    } else if (tag == _eventTag) {
      final json = jsonDecode(utf8.decode(body)) as Map<String, dynamic>;
      _eventController.add(TransportEvent.fromJson(json));
    }
  }

  void _onDone() {
    _closeController.add(null);
    _audioController.close();
    _eventController.close();
    _closeController.close();
  }

  @override
  void sendAudio(Uint8List data) {
    if (_channel == null) return;
    final tagged = Uint8List(1 + data.length);
    tagged[0] = _audioTag;
    tagged.setRange(1, tagged.length, data);
    _channel!.sink.add(tagged);
  }

  @override
  void sendEvent(TransportEvent event) {
    if (_channel == null) return;
    final json = utf8.encode(jsonEncode(event.toJson()));
    final tagged = Uint8List(1 + json.length);
    tagged[0] = _eventTag;
    tagged.setRange(1, tagged.length, json);
    _channel!.sink.add(tagged);
  }

  @override
  Stream<Uint8List> get audioStream => _audioController.stream;

  @override
  Stream<TransportEvent> get eventStream => _eventController.stream;

  @override
  Stream<void> get closeStream => _closeController.stream;

  @override
  void close() {
    _channel?.sink.close();
    _channel = null;
  }
}
```

### URL construction

Same pattern as the web client — derive from the API base URL:

```dart
String wsUrl(String baseUrl, String sessionId) {
  final uri = Uri.parse(baseUrl);
  final scheme = uri.scheme == 'https' ? 'wss' : 'ws';
  return '$scheme://${uri.host}:${uri.port}/ws/stream/$sessionId';
}
```

## 2.4 Unit Tests

### API client tests (port of `api.test.ts`)

Use `mocktail` to mock Dio's `HttpClientAdapter`:

```dart
// test/api/client_test.dart

// 1. fetchServerStats returns typed ServerStats
// 2. fetchPipelines returns list of PipelineInfo
// 3. createSession sends correct JSON, returns Session
// 4. fetchSession with valid ID returns Session
// 5. fetchSession with invalid ID throws ApiException(404)
// 6. updateSession sends PATCH with correct body
// 7. deleteSession sends DELETE
// 8. createSession with bad pipeline returns ApiException(400)
```

### Transport tests (port of `transport.test.ts`)

Use an in-memory `StreamChannel` pair:

```dart
// test/transport/ws_transport_test.dart

// 1. sendAudio prefixes 0x01 tag
// 2. sendEvent prefixes 0x02 tag with JSON payload
// 3. Received audio frame dispatched to audioStream
// 4. Received event frame parsed and dispatched to eventStream
// 5. Connection close dispatched to closeStream
// 6. sendAudio with empty Uint8List sends 1-byte EOF frame
```

## 2.5 Verification

After this phase:

- `flutter test test/api/` — 8 tests pass
- `flutter test test/transport/` — 6 tests pass
- Manual: create a session via `ApiClient`, connect via `WebSocketTransport`, verify server logs `connection open`

Next: [Phase 3 — Audio Capture & Playback](./03-audio.md)
