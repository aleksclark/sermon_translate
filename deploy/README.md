# Deployment & GPU Infrastructure

Foundations for running the sermon-translate translation server with GPU
inference — and fine-tuning/training — on **node-6** and its two NVIDIA Tesla
V100 GPUs.

> These are **artifacts and docs only**. Nothing here submits, plans, or mutates
> any Nomad job. Building images and submitting jobs is an explicit operator
> action. **Read the [safety note](#safety--capacity) before running anything.**

## node-6 capabilities

| Property     | Value                                                  |
|--------------|--------------------------------------------------------|
| Name / IP    | `node-6` / `192.168.0.99`                              |
| Nomad        | 2.0.x, region `home`, datacenter `home`               |
| CPU          | 8-core Xeon E5-1620 v4                                  |
| RAM          | ~32 GB                                                  |
| GPUs         | **2 × Tesla V100-SXM2-16GB** (Volta, sm_70)            |
| **VRAM**     | **2 × 16 GB = 32 GB total**                            |
| Docker       | driver runtimes `io.containerd.runc.v2, nvidia, runc` |
| Node meta    | `gpu=true`, `gpu_count=2`, `gpu_type=v100-sxm2-16gb`, `ram_gb=32`, `compute=true` |
| Host volumes | `local-data` (`/data/disk0`), `moosefs` (`/mnt/moosefs`), `moosefs-configs`, `moosefs-family`, `moosefs-media` |

## Files

```
deploy/
  README.md                              this file
  nomad/
    sermon-translate-gpu.nomad.hcl       GPU inference service (1 or 2 GPUs)
    sermon-translate-train.nomad.hcl     GPU training/fine-tune BATCH template
server/
  Dockerfile.gpu                         CUDA/cuDNN server image (torch, faster-whisper, seamless)
  Dockerfile                             existing CPU image (unchanged)
```

## Operator prerequisites (host side — NOT configured here)

These must already be true on the node-6 Nomad client. This repo does **not**
configure the host:

1. **NVIDIA driver + `nvidia` container runtime** registered with Docker
   (already present per `docker` driver runtimes above).
2. **Nomad `nvidia` device plugin** enabled on the client so that the
   `device "nvidia/gpu"` stanza can fingerprint and assign GPUs. Without it the
   job will stay `pending` (unplaceable) — it will not error loudly.
3. The GPU image (`server/Dockerfile.gpu`) built and pushed to a registry the
   node can pull from.

## Building the GPU image

The base is `nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04`. CUDA 12.4 + cuDNN 9
supports Volta (V100, sm_70), PyTorch 2.x wheels, and ctranslate2's runtime
cuBLAS/cuDNN needs (faster-whisper).

### The seamless_communication local-dep problem

`server/pyproject.toml` declares:

```toml
[tool.uv.sources]
seamless-communication = { path = "../../seamless_communication" }
```

That path resolves to a **sibling checkout outside the `server/` build
context**, so a naive `docker build server/` cannot see it. The
`SeamlessStreaming` pipeline is already **conditional at import time**, so the
server still runs without it — but if you want that pipeline on GPU you must
make the source reachable at build time. `Dockerfile.gpu` handles this with a
`SEAMLESS_SRC` build arg and by staging the source where the relative path
resolves inside the image (`/seamless_communication`).

**Strategy A — vendored copy (recommended, used by default).** Build from the
repo root so both `server/` and the vendored seamless source are in context:

```bash
# from the repo root
cp -r ../seamless_communication server/vendor/seamless_communication   # adjust to your checkout
docker build -f server/Dockerfile.gpu \
  --build-arg SEAMLESS_SRC=server/vendor/seamless_communication \
  -t <registry>/sermon-translate-server:gpu .
docker push <registry>/sermon-translate-server:gpu
```

**Strategy B — skip seamless.** If you do not need the SeamlessStreaming
pipeline, build without it (maintain an optional extra and pass
`UV_SYNC_ARGS`/`--extra` accordingly). The server starts fine; only that one
pipeline is unavailable. This keeps the image smaller and the build simpler.

> The Dockerfile does **not** pretend seamless "just works" — you must vendor it
> (Strategy A) or opt out (Strategy B) before building.

## Running the inference job

```bash
# Verify node-6 capacity FIRST (see safety note).
nomad job run \
  -var 'image=<registry>/sermon-translate-server:gpu' \
  -var 'crosstalk_base_url=https://crosstalk.example' \
  deploy/nomad/sermon-translate-gpu.nomad.hcl
```

### GPU device & constraint stanzas (inference)

Pinned to node-6, one GPU requested (`gpu_count` variable → set `2` for both):

```hcl
constraint {
  attribute = "${node.unique.name}"
  value     = "node-6"
}

constraint {
  attribute = "${meta.gpu}"
  value     = "true"
}

resources {
  cpu    = 4000
  memory = 12288

  device "nvidia/gpu" {
    count = var.gpu_count        # 1 = single V100 (16 GB); 2 = both (32 GB)

    constraint {
      attribute = "${device.attr.memory}"
      operator  = ">="
      value     = "16000 MiB"
    }
  }
}
```

The docker task sets `runtime = "nvidia"`. Nomad injects
`NVIDIA_VISIBLE_DEVICES` for the assigned GPUs and the nvidia runtime exposes
exactly those (re-indexed from 0); the app selects the compute device via
`COMPUTE_DEVICE=cuda`. Do **not** set `CUDA_VISIBLE_DEVICES=all` — CUDA rejects
the literal `all` and would mask every GPU. Leave `CUDA_VISIBLE_DEVICES` unset
so CUDA sees all injected devices, or pin an ordinal
(`CUDA_VISIBLE_DEVICES=0`) only if you deliberately want one GPU out of the set.

### Two-GPU variant

Set `-var 'gpu_count=2'` to reserve both V100s (32 GB total). Only do this when
no other GPU workload (e.g. a training run) needs node-6.

## Running the training / fine-tune job

`deploy/nomad/sermon-translate-train.nomad.hcl` is a **`type = "batch"`
template**. Nomad runs it to completion and stops. It mounts a node-6 host
volume for datasets and checkpoints and claims 1 or 2 GPUs.

```bash
nomad job run \
  -var 'image=<registry>/sermon-translate-trainer:gpu' \
  -var 'gpu_count=1' \
  -var 'host_volume=local-data' \
  deploy/nomad/sermon-translate-train.nomad.hcl
```

**Where the real trainer goes:** the job ships a *placeholder* command
(`nvidia-smi` + workspace listing). Replace the `config { command / args }` in
the task with your entrypoint, e.g.:

```hcl
command = "uv"
args = ["run", "--no-sync", "python", "-m", "src.training.finetune",
        "--data", "/workspace/datasets",
        "--out",  "/workspace/checkpoints"]
```

Add that module to the training image. Datasets and checkpoints live under the
mounted `/workspace` (host volume) so they survive allocation restarts. No
training script is invented here — the scaffold only proves GPU + volume
plumbing.

### Host-volume mount (training)

```hcl
volume "workspace" {
  type      = "host"
  source    = var.host_volume    # "local-data" (/data/disk0) or "moosefs"
  read_only = false
}
# ...
volume_mount {
  volume      = "workspace"
  destination = "/workspace"
  read_only   = false
}
```

## VRAM budgeting (16 GB per GPU)

Each V100 has **16 GB**. Rough guidance for a single GPU:

| Workload                                    | Fits in 16 GB? | Notes |
|---------------------------------------------|----------------|-------|
| faster-whisper `base`/`small`/`medium` fp16 | Yes, easily    | <2 GB even at `large-v3` int8_float16 |
| faster-whisper `large-v3` fp16              | Yes            | ~3–5 GB |
| Opus-MT en→es (ctranslate2)                 | Yes            | small |
| SeamlessStreaming (s2tt, fp16)              | Yes            | fits on one V100 |
| **Inference server, all pipelines loaded**  | Yes on 1 GPU   | leaves headroom |
| **Fine-tuning a 7B+ model, full precision** | **No**         | shard across both GPUs (32 GB) or use LoRA/8-bit |

**Sharding across 2 GPUs:** for training that exceeds 16 GB, claim both V100s
(`gpu_count=2`, 32 GB total) and use your trainer's model/data parallelism
(FSDP, DeepSpeed ZeRO, `device_map="auto"`, etc.). Inference for these models
fits comfortably on one GPU; keep inference to a single GPU so the other stays
free.

## Compute-device configuration (server)

The server picks its inference device from settings (Phase 2 `server/src/config.py`):

| Env var          | Default | Effect |
|------------------|---------|--------|
| `COMPUTE_DEVICE` | `cpu`   | `cuda`, `cuda:0`, `cpu`, … — passed to faster-whisper / ctranslate2 and SeamlessStreaming |
| `COMPUTE_TYPE`   | *(auto)*| `float16`/`int8_float16`/`int8`; auto-selects `float16` on CUDA, `int8` on CPU |

The GPU Nomad job sets `COMPUTE_DEVICE=cuda`. Defaults remain CPU/int8, so
existing CPU deployments are unchanged. `SeamlessStreaming` honors an explicit
`COMPUTE_DEVICE` and otherwise auto-detects CUDA as before.

## Safety & capacity

Submitting these jobs **co-locates them on node-6 with the existing workloads**:

```
traefik, syncthing, coredns, emqx, otel, idrive, temp-exporter
```

node-6 has only **8 CPU cores and ~32 GB RAM**. The inference job reserves
4000 MHz CPU + 12 GB RAM; the training template reserves 4000 MHz + 16 GB.
**Before running either job, verify node-6 has free CPU, RAM, and GPU capacity**
and confirm you are not starving the services above. Do not run the inference
job and a 2-GPU training job on node-6 at the same time.

## Validation performed

- `nomad fmt` — both HCL files formatted (idempotent, no diff).
- `nomad job validate` — run against an **unreachable** agent
  (`NOMAD_ADDR=http://127.0.0.1:1`) so only the **local HCL parse/structural
  validation** ran; **no cluster contact, no plan, no submit**. Both files
  report "Job validation successful".
