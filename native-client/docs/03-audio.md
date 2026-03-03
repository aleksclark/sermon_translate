# Phase 3 — Audio Capture & Playback

## Overview

Replace the browser's Web Audio API with native platform audio. Three concerns:

1. **Mic capture** → stream PCM to server (replaces `getUserMedia` + `ScriptProcessor`)
2. **File decode** → stream MP3-as-PCM to server (replaces `AudioContext.decodeAudioData`)
3. **Playback** → play received PCM from server (replaces `AudioContext.createBufferSource`)

All three produce or consume **Int16LE mono PCM** — the server's universal format.

## 3.1 Audio Source Abstraction

```dart
// audio/audio_source.dart

abstract class AudioSource {
  /// Yields PCM chunks (Int16LE mono) at approximately real-time pace.
  /// [sampleRate] is the session's sample rate (default 48000).
  Stream<Uint8List> pcmStream(int sampleRate);

  /// Release platform resources.
  Future<void> dispose();
}
```

This is the Dart equivalent of the web client's `audioSource.type` branching in `useAudioStream.ts`. The transport layer doesn't know or care which source is active — it just receives `Uint8List` chunks.

## 3.2 Mic Source

**Replaces:** `navigator.mediaDevices.getUserMedia` + `ScriptProcessor`

### Package

[`record`](https://pub.dev/packages/record) v5 — supports PCM output on all platforms.

### Platform audio backends

| Platform | Backend |
|---|---|
| macOS | CoreAudio |
| Linux | PulseAudio / ALSA |
| Windows | WASAPI |
| iOS | AVAudioSession |
| Android | AudioRecord |

### Implementation sketch

```dart
// audio/mic_source.dart

class MicSource implements AudioSource {
  final AudioRecorder _recorder = AudioRecorder();
  final String? deviceId;

  MicSource({this.deviceId});

  @override
  Stream<Uint8List> pcmStream(int sampleRate) async* {
    final stream = await _recorder.startStream(RecordConfig(
      encoder: AudioEncoder.pcm16bits,
      sampleRate: sampleRate,
      numChannels: 1,
      autoGain: false,
      echoCancel: false,
      noiseSuppress: false,
      device: deviceId != null ? InputDevice(id: deviceId!) : null,
    ));

    await for (final chunk in stream) {
      yield Uint8List.fromList(chunk);
    }
  }

  @override
  Future<void> dispose() async {
    await _recorder.stop();
    _recorder.dispose();
  }
}
```

### Device enumeration

```dart
Future<List<InputDevice>> listInputDevices() async {
  final recorder = AudioRecorder();
  final devices = await recorder.listInputDevices();
  recorder.dispose();
  return devices;
}
```

Equivalent to `useAudioDevices.ts` filtering for `audioinput`.

### Chunk size

The web client uses `ScriptProcessor` with `bufferSize=4096` samples. The `record` package streams chunks at its own native buffer size. If chunks arrive in a different size, they can be forwarded as-is — the server's `np.frombuffer` / `np.concatenate` handles arbitrary chunk sizes. No rebuffering needed.

## 3.3 File Source

**Replaces:** `streamFileAudio()` in `useAudioStream.ts`

### Decode strategy

The web client decodes MP3 via `AudioContext.decodeAudioData` + resamples via `OfflineAudioContext`. In Flutter, two options:

| Approach | Package | Pros | Cons |
|---|---|---|---|
| **`just_audio`** | `just_audio` | Already a dependency for playback, works on all platforms | No direct "decode to PCM buffer" API — would need to use its player internals |
| **`ffmpeg_kit`** | `ffmpeg_kit_flutter_audio` | Full codec support, decode to raw PCM file | Large binary (~15MB), slower startup |
| **Dart FFI + minimp3** | custom | Tiny (~50KB), pure decode, no system deps | Requires writing FFI bindings |

**Recommended:** Start with `ffmpeg_kit_flutter_audio` for reliable cross-platform MP3→PCM decode. If bundle size is a concern, migrate to minimp3 FFI later.

### Implementation sketch

```dart
// audio/file_source.dart

class FileSource implements AudioSource {
  final String filePath;
  bool _cancelled = false;

  FileSource(this.filePath);

  @override
  Stream<Uint8List> pcmStream(int sampleRate) async* {
    // Decode MP3 to raw PCM file using ffmpeg
    final pcmPath = '${filePath}.pcm';
    await FFmpegKit.execute(
      '-i "$filePath" -f s16le -acodec pcm_s16le '
      '-ar $sampleRate -ac 1 "$pcmPath" -y'
    );

    final file = File(pcmPath);
    final bytes = await file.readAsBytes();
    final chunkSize = 4096 * 2; // 4096 samples × 2 bytes per Int16
    final chunkDuration = Duration(
      microseconds: (4096 / sampleRate * 1e6).round(),
    );

    for (var offset = 0; offset < bytes.length; offset += chunkSize) {
      if (_cancelled) break;
      final end = (offset + chunkSize).clamp(0, bytes.length);
      yield bytes.sublist(offset, end);
      await Future.delayed(chunkDuration);
    }

    // Signal end-of-audio (empty frame)
    yield Uint8List(0);

    // Cleanup temp file
    await file.delete();
  }

  @override
  Future<void> dispose() async {
    _cancelled = true;
  }
}
```

### Real-time pacing

Same approach as the web client: `Future.delayed` with `chunkDuration = (CHUNK_SIZE / sampleRate)` seconds between chunks. This ensures the server's 3-second Whisper buffer fills at the expected rate.

### End-of-audio signal

After the last chunk, yield `Uint8List(0)`. The transport sends this as a 1-byte `0x01` frame (empty payload). The server interprets this as end-of-stream, processes the final Whisper buffer, and sends back the transcript.

## 3.4 Audio Playback

**Replaces:** `transport.onAudio` callback + `AudioContext.createBufferSource`

Used by the Echo and Spanish Translation pipelines, which send audio back to the client.

### Approach

Use `just_audio` with a custom `StreamAudioSource` that accepts PCM chunks from the transport:

```dart
// audio/pcm_player.dart

class PcmPlayer {
  final AudioPlayer _player = AudioPlayer();
  final StreamController<Uint8List> _pcmSink = StreamController();
  final int sampleRate;

  PcmPlayer({required this.sampleRate});

  void addPcm(Uint8List chunk) {
    _pcmSink.add(chunk);
  }

  Future<void> start() async {
    await _player.setAudioSource(
      _PcmStreamSource(_pcmSink.stream, sampleRate),
    );
    _player.play();
  }

  Future<void> dispose() async {
    await _pcmSink.close();
    await _player.dispose();
  }
}
```

The `_PcmStreamSource` wraps the PCM stream in a WAV header (44 bytes) so `just_audio` can play it. Alternatively, use platform channels to write directly to the audio output buffer — more efficient but more platform-specific code.

### Output device selection

`just_audio` uses the system default output. For explicit device selection (matching the web client's output device dropdown), use `just_audio`'s `AudioSession` configuration or platform-specific APIs. This is a nice-to-have — the web client's output device selector is rarely used.

## 3.5 Wiring Audio to Transport

The audio streaming orchestration lives in the state layer (see [Phase 4](./04-state.md)), but the core loop is:

```dart
Future<void> streamAudio(
  AudioSource source,
  StreamTransport transport,
  int sampleRate,
) async {
  await for (final chunk in source.pcmStream(sampleRate)) {
    transport.sendAudio(chunk);
  }
}
```

For mic: the stream runs until the user clicks Stop (which calls `source.dispose()`).
For file: the stream ends naturally after the last chunk + EOF signal. The transport stays open for receiving transcript events.

## 3.6 Platform Permissions

### iOS — `Info.plist`

```xml
<key>NSMicrophoneUsageDescription</key>
<string>Microphone access is needed for live sermon translation.</string>
```

### Android — `AndroidManifest.xml`

```xml
<uses-permission android:name="android.permission.RECORD_AUDIO" />
```

### macOS — `DebugProfile.entitlements` + `Release.entitlements`

```xml
<key>com.apple.security.device.audio-input</key>
<true/>
<key>com.apple.security.network.client</key>
<true/>
```

### Linux / Windows

No special permissions. Mic access is granted by the OS.

## 3.7 Verification

After this phase:

- `MicSource` streams PCM to the server, server logs show `bytes_received` increasing
- `FileSource` decodes the test fixture MP3, streams PCM, server returns transcript
- `PcmPlayer` plays audio received from the Echo pipeline
- Manual test on at least one desktop platform (macOS or Linux)

Next: [Phase 4 — State Management](./04-state.md)
