# Phase 4 — State Management

## Overview

Replace React hooks with Riverpod providers. The mapping is nearly 1:1 — each hook becomes a provider, `useState` becomes `StateNotifier`, polling `useEffect` becomes `Timer` inside a `Notifier`.

## 4.1 Hook → Provider Mapping

| React hook | Riverpod provider | Refresh strategy |
|---|---|---|
| `useServerStats(2000)` | `serverStatsProvider` (auto-dispose `AsyncNotifier`) | `Timer.periodic` every 2s |
| `useSessions(3000)` | `sessionsProvider` (auto-dispose `AsyncNotifier`) | `Timer.periodic` every 3s |
| `useState<ActiveSession>` | `activeSessionProvider` (`StateNotifier`) | Manual set/clear |
| `useAudioStream(options)` | `audioStreamProvider` (family `AsyncNotifier`) | Lifecycle-driven (connect on create, dispose on stop) |

## 4.2 Provider Definitions

### API Client Provider

Singleton — all other providers depend on this:

```dart
final apiClientProvider = Provider<ApiClient>((ref) {
  return ApiClient(baseUrl: const String.fromEnvironment(
    'API_BASE_URL',
    defaultValue: 'http://localhost:8000/api',
  ));
});
```

### Server Stats

Replaces `useServerStats.ts`:

```dart
final serverStatsProvider =
    AsyncNotifierProvider.autoDispose<ServerStatsNotifier, ServerStats?>(
  ServerStatsNotifier.new,
);

class ServerStatsNotifier extends AutoDisposeAsyncNotifier<ServerStats?> {
  Timer? _timer;

  @override
  Future<ServerStats?> build() async {
    ref.onDispose(() => _timer?.cancel());
    _timer = Timer.periodic(
      const Duration(seconds: 2),
      (_) => _refresh(),
    );
    return _fetch();
  }

  Future<ServerStats?> _fetch() async {
    try {
      return await ref.read(apiClientProvider).fetchServerStats();
    } catch (_) {
      return null;
    }
  }

  Future<void> _refresh() async {
    state = AsyncData(await _fetch());
  }
}
```

### Sessions

Replaces `useSessions.ts`:

```dart
final sessionsProvider =
    AsyncNotifierProvider.autoDispose<SessionsNotifier, List<Session>>(
  SessionsNotifier.new,
);

class SessionsNotifier extends AutoDisposeAsyncNotifier<List<Session>> {
  Timer? _timer;

  @override
  Future<List<Session>> build() async {
    ref.onDispose(() => _timer?.cancel());
    _timer = Timer.periodic(
      const Duration(seconds: 3),
      (_) => _refresh(),
    );
    return _fetch();
  }

  Future<List<Session>> _fetch() async {
    try {
      return await ref.read(apiClientProvider).fetchSessions();
    } catch (_) {
      return [];
    }
  }

  Future<void> _refresh() async {
    state = AsyncData(await _fetch());
  }

  void invalidate() => ref.invalidateSelf();
}
```

### Active Session

Replaces `useState<ActiveSession | null>` in `App.tsx`:

```dart
@freezed
class ActiveSession with _$ActiveSession {
  const factory ActiveSession({
    required Session session,
    required AudioSource audioSource,
  }) = _ActiveSession;
}

final activeSessionProvider =
    StateNotifierProvider<ActiveSessionNotifier, ActiveSession?>(
  (ref) => ActiveSessionNotifier(ref),
);

class ActiveSessionNotifier extends StateNotifier<ActiveSession?> {
  final Ref _ref;
  ActiveSessionNotifier(this._ref) : super(null);

  void start(Session session, AudioSource source) {
    state = ActiveSession(session: session, audioSource: source);
  }

  Future<void> stop() async {
    final id = state?.session.id;
    await state?.audioSource.dispose();
    state = null;
    if (id != null) {
      try {
        await _ref.read(apiClientProvider).updateSession(
          id,
          SessionUpdate(status: SessionStatus.closed),
        );
      } catch (_) {}
    }
    _ref.read(sessionsProvider.notifier).invalidate();
  }
}
```

### Audio Stream

Replaces `useAudioStream.ts`. This is the most complex provider — it manages the transport, audio streaming, and exposes reactive state.

```dart
final audioStreamProvider =
    StateNotifierProvider.autoDispose<AudioStreamNotifier, AudioStreamState>(
  (ref) {
    final notifier = AudioStreamNotifier(ref);
    final session = ref.watch(activeSessionProvider);
    if (session != null) {
      notifier.connect(session);
    }
    return notifier;
  },
);

@freezed
class AudioStreamState with _$AudioStreamState {
  const factory AudioStreamState({
    @Default(false) bool connected,
    SessionStats? liveStats,
    @Default([]) List<String> transcript,
  }) = _AudioStreamState;
}

class AudioStreamNotifier extends StateNotifier<AudioStreamState> {
  final Ref _ref;
  StreamTransport? _transport;
  StreamSubscription? _eventSub;
  StreamSubscription? _closeSub;
  StreamSubscription? _audioSub;

  AudioStreamNotifier(this._ref) : super(const AudioStreamState());

  Future<void> connect(ActiveSession active) async {
    final session = active.session;
    final baseUrl = _ref.read(apiClientProvider).baseUrl;
    final url = wsUrl(baseUrl, session.id);

    final transport = WebSocketTransport(url);
    try {
      await transport.connect();
    } catch (_) {
      return;
    }

    _transport = transport;
    state = state.copyWith(connected: true);

    // Start streaming audio
    _streamAudio(active.audioSource, transport, session.sampleRate);

    // Listen for events
    _eventSub = transport.eventStream.listen(_onEvent);
    _closeSub = transport.closeStream.listen((_) {
      state = state.copyWith(connected: false);
    });
  }

  Future<void> _streamAudio(
    AudioSource source,
    StreamTransport transport,
    int sampleRate,
  ) async {
    await for (final chunk in source.pcmStream(sampleRate)) {
      transport.sendAudio(chunk);
    }
  }

  void _onEvent(TransportEvent event) {
    if (event.type == 'session.stats') {
      final stats = SessionStats.fromJson(event.payload);
      state = state.copyWith(liveStats: stats);
    } else if (event.type == 'pipeline.event' &&
        event.payload['kind'] == 'transcript') {
      final text = event.payload['text'] as String;
      state = state.copyWith(
        transcript: [...state.transcript, text],
      );
    }
  }

  @override
  void dispose() {
    _eventSub?.cancel();
    _closeSub?.cancel();
    _audioSub?.cancel();
    _transport?.close();
    super.dispose();
  }
}
```

## 4.3 Data Flow Diagram

```
┌─────────────┐     polls      ┌──────────────────┐
│ Server REST  │◄──────────────│ serverStatsProvider│
│ /api/stats   │               │ sessionsProvider   │
└─────────────┘               └──────────────────┘
                                       │
                                       ▼ watches
                              ┌──────────────────┐
                              │ UI Widgets        │
                              │ (ref.watch)       │
                              └──────────────────┘
                                       │
                                       │ user action
                                       ▼
                              ┌──────────────────┐
                              │activeSessionProv. │──► AudioSource.pcmStream()
                              └──────────────────┘        │
                                       │                  │ Uint8List chunks
                                       ▼                  ▼
                              ┌──────────────────┐  ┌──────────┐
                              │audioStreamProvider│──│ WS Trans │──► Server
                              │ connected         │  └──────────┘
                              │ liveStats         │◄─── events ───┘
                              │ transcript        │
                              └──────────────────┘
```

## 4.4 Key Differences from React

| React pattern | Riverpod equivalent | Notes |
|---|---|---|
| `useState(x)` | `state = x` inside `StateNotifier` | Notifier auto-notifies listeners |
| `useEffect(() => {}, [dep])` | `ref.watch(dep)` in `build()` | Riverpod rebuilds on dependency changes |
| `useCallback` | Not needed | Dart closures don't have identity issues |
| `useRef` | Instance field on `Notifier` | `_transport`, `_timer`, etc. |
| `useMemo` | `ref.watch` + `select` | Riverpod deduplicates automatically |
| Cleanup in `useEffect` return | `ref.onDispose()` | Called when provider is disposed |

## 4.5 Verification

After this phase:

- Server stats update reactively every 2s in UI
- Session list refreshes every 3s
- Starting a session creates an `ActiveSession`, connects transport, streams audio
- Stopping a session disposes audio, closes transport, PATCHes server
- Transcript lines appear in state as pipeline events arrive

Next: [Phase 5 — UI Widgets](./05-ui.md)
