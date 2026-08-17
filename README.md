# Speaches

`speaches` is an OpenAI API-compatible server supporting streaming transcription, translation, and speech generation. Speach-to-Text is powered by [faster-whisper](https://github.com/SYSTRAN/faster-whisper) and for Text-to-Speech [piper](https://github.com/rhasspy/piper) and [Kokoro](https://huggingface.co/hexgrad/Kokoro-82M) are used. This project aims to be Ollama, but for TTS/STT models.

See the documentation for installation instructions and usage: [speaches.ai](https://speaches.ai/)

## Features:

- OpenAI API compatible. All tools and SDKs that work with OpenAI's API should work with `speaches`.
- Audio generation (chat completions endpoint) | [OpenAI Documentation](https://platform.openai.com/docs/guides/realtime)
  - Generate a spoken audio summary of a body of text (text in, audio out)
  - Perform sentiment analysis on a recording (audio in, text out)
  - Async speech to speech interactions with a model (audio in, audio out)
- Streaming support (transcription is sent via SSE as the audio is transcribed. You don't need to wait for the audio to fully be transcribed before receiving it).
- Dynamic model loading / offloading. Just specify which model you want to use in the request and it will be loaded automatically. It will then be unloaded after a period of inactivity.
- Text-to-Speech via `kokoro`(Ranked #1 in the [TTS Arena](https://huggingface.co/spaces/Pendrokar/TTS-Spaces-Arena)) and `piper` models.
- GPU and CPU support.
- [Deployable via Docker Compose / Docker](https://speaches.ai/installation/)
- [Realtime API](https://speaches.ai/usage/realtime-api)
- [Highly configurable](https://speaches.ai/configuration/)

Please create an issue if you find a bug, have a question, or a feature suggestion.


## Jetson / Orin Nano (aarch64) GPU build

`Dockerfile.jetson` builds a CUDA-accelerated STT container for NVIDIA Jetson devices (Orin Nano, compute capability 8.7). The stock `Dockerfile` targets x86-64 CUDA; the Jetson variant fixes the following issues found when building on the Orin Nano:

### STAGE 1 — build a CUDA-enabled ctranslate2 wheel (aarch64)

- cuDNN dev headers are already bundled in `l4t-jetpack:r36.4.0`, so the redundant `libcudnn9-dev-cuda-12` install is dropped.
- CTranslate2 4.5.0 is compiled with `-DWITH_CUDA=ON -DWITH_CUDNN=ON -DCMAKE_CUDA_ARCHITECTURES=87` and `make -j2` (parallel CUDA compile on the Nano).
- The Python wheel is built with a **Python 3.12** venv (`uv venv --python 3.12`) so the compiled `cp312` ABI matches the runtime; `uv run pip install` is used instead of `pip` (avoids the PEP 668 externally-managed-environment error).
- The core build is verified for CUDA symbols (guard against a silent CPU fallback) before packaging.
- `setup.py` ignores `CTRANSLATE2_ROOT` and bundles its own CPU core, so the runtime **overrides the installed `libctranslate2.so` with the CUDA build** and registers it via `ldconfig` — guaranteeing GPU dispatch.

### STAGE 2 — runtime

- The Jetson apt repo (`repo.download.nvidia.com/jetson`) is enabled and `libcudnn9-cuda-12 libopenblas0` are installed (the CUDA core lib needs the cuDNN runtime).
- `uv sync` is pinned to Python 3.12 and uses buildkit cache/bind mounts.
- The CUDA wheel is installed with `uv pip install --force-reinstall --no-deps`, then the CUDA `libctranslate2.so` is forced into the installed package.
- Both `/usr/local/lib` (CUDA core) and the venv's bundled nvidia libs are registered with `ldconfig`.
- `EXPOSE 8000` + `CMD ["uvicorn", "--factory", "speaches.main:create_app"]`.

### Runtime env for keeping the STT model in memory

The app offloads the STT model after `stt_model_ttl` seconds (default 300). To keep the loaded model in memory and never auto-offload it until another model is loaded, run the container with:

```
-e STT_MODEL_TTL=-1
```

(`ttl < 0` disables the unload timer.) This is a **runtime** setting on the container, not baked into the image.

### Verify GPU (not CPU fallback)

```sh
docker exec <container> .venv/bin/python -c "import ctranslate2; print(ctranslate2.get_cuda_device_count())"
```

During inference `tegrastats` should show `GR3D_FREQ` busy (e.g. 26–99%) and GPU power (`VDD_IN`) spiking to ~9 W.

## Demos

### Realtime API

https://github.com/user-attachments/assets/457a736d-4c29-4b43-984b-05cc4d9995bc

(Excuse the breathing lol. Didn't have enough time to record a better demo)

### Streaming Transcription

TODO

### Speech Generation

https://github.com/user-attachments/assets/0021acd9-f480-4bc3-904d-831f54c4d45b
