# Audio fixture NOTICE

## Purpose

Tiny synthetic English PCM fixtures for `stage.v1` protocol and real EN→ES
pre-EOS E2E gates (Wave 4 / G4). These prove **behavior**, not production speech
quality. No external ASR/MT corpora were downloaded.

## License / public-domain source text

| Fixture | Source text | Text status |
|---|---|---|
| `public-domain-en-01` | "In the beginning God created the heaven and the earth." | King James Version (KJV), public domain (US) |
| `public-domain-en-02` | "The Lord is my shepherd; I shall not want. He maketh me to lie down in green pastures." | KJV Psalm 23:1–2, public domain (US) |
| `public-domain-en-03` | "Blessed are the meek: for they shall inherit the earth." | KJV Matthew 5:5, public domain (US) |
| `silence-1s-16k` | (no speech) | Generated silence |
| `tone-440hz-500ms-16k` | (no speech) | Deterministic sine |

Synthetic speech audio is generated locally with **espeak-ng** (open-source TTS)
from the public-domain sentences above. The audio waveforms themselves are
machine-generated test signals, not third-party recordings.

## Human Spanish references (evaluation only — not a hard quality gate)

| Fixture | EN | ES reference (human) |
|---|---|---|
| `public-domain-en-01` | In the beginning God created the heaven and the earth. | En el principio creó Dios los cielos y la tierra. |
| `public-domain-en-02` | The Lord is my shepherd; I shall not want. He maketh me to lie down in green pastures. | Jehová es mi pastor; nada me faltará. En lugares de delicados pastos me hará descansar. |
| `public-domain-en-03` | Blessed are the meek: for they shall inherit the earth. | Bienaventurados los mansos, porque ellos recibirán la tierra por heredad. |

## Sample format

- Codec: `pcm_s16le`
- Sample rate: `16000` Hz
- Channels: `1` (mono)
- Container: RIFF WAVE (`.wav`) and raw little-endian PCM (`.pcm`)

## Generation toolchain

```text
espeak-ng 1.52.0
ffmpeg (libav)
```

### Regeneration commands

```bash
# Example: public-domain-en-01 (triple-utterance with short silences for pre-EOS)
TEXT='In the beginning God created the heaven and the earth.'
espeak-ng -v en-us -s 130 -w /tmp/a.wav "$TEXT"
ffmpeg -y -i /tmp/a.wav -ar 16000 -ac 1 -c:a pcm_s16le /tmp/a16.wav
ffmpeg -y -f lavfi -i anullsrc=r=16000:cl=mono -t 0.35 -acodec pcm_s16le /tmp/sil.wav
# concat a16 + sil + a16 + sil + a16 → public-domain-en-01.wav
ffmpeg -y -i public-domain-en-01.wav -f s16le -acodec pcm_s16le public-domain-en-01.pcm

# silence
ffmpeg -y -f lavfi -i anullsrc=r=16000:cl=mono -t 1.0 -acodec pcm_s16le silence-1s-16k.wav

# tone
ffmpeg -y -f lavfi -i 'sine=frequency=440:sample_rate=16000:duration=0.5' \
  -ac 1 -acodec pcm_s16le tone-440hz-500ms-16k.wav
```

Exact committed byte digests are in `MANIFEST.sha256.json`.

## SHA-256

See `MANIFEST.sha256.json` in this directory. Do not hand-edit digests; regenerate
fixtures and refresh the manifest together.
