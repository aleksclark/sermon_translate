"""Pipeline evaluation harness.

Run from the server directory::

    uv run python -m src.harness                          # all pipelines
    uv run python -m src.harness --pipeline whisper-tts   # one pipeline
    uv run python -m src.harness --audio path/to/file.mp3 # custom audio
"""
