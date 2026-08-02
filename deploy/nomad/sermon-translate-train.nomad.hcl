# Training / fine-tuning batch job for translation models on node-6.
#
# TEMPLATE. This is a scaffold operators complete with a real trainer entrypoint.
# It is `type = "batch"`: Nomad runs it to completion, then the allocation stops.
# It mounts a host volume for datasets and checkpoints and can claim 1 or 2 of
# node-6's Tesla V100 GPUs (16 GB each, 32 GB total across both).
#
# SAFETY: submitting this co-locates a training container on node-6 with the
# existing workloads (traefik, syncthing, coredns, emqx, otel, idrive,
# temp-exporter) and possibly the inference job. Training is memory- and
# GPU-hungry; verify node-6 has free CPU/RAM/VRAM before running, and prefer
# claiming both GPUs only when the inference job is NOT scheduled there.
#
# WHERE THE REAL TRAINER GOES: replace the placeholder `args` command below with
# your training entrypoint (e.g. `uv run python -m src.training.finetune ...`).
# Add that module to the training image; do not commit training scripts here
# unless they exist. Datasets and checkpoints belong under the mounted
# /workspace volume so they survive allocation restarts.

variable "image" {
  type        = string
  description = "Training image reference (a GPU image with the trainer baked in)."
  default     = "sermon-translate-trainer:gpu"
}

variable "gpu_count" {
  type        = number
  description = "GPUs to claim for training: 1 (16 GB) or 2 (32 GB, model/data parallel)."
  default     = 1
}

variable "host_volume" {
  type        = string
  description = "node-6 host volume for datasets/checkpoints (local-data or moosefs)."
  default     = "local-data"
}

job "sermon-translate-train" {
  datacenters = ["home"]
  region      = "home"
  type        = "batch"

  constraint {
    attribute = "${node.unique.name}"
    value     = "node-6"
  }

  constraint {
    attribute = "${meta.gpu}"
    value     = "true"
  }

  group "trainer" {
    count = 1

    # Datasets and checkpoints live on a node-6 host volume so they persist
    # across allocation restarts. local-data -> /data/disk0; moosefs -> shared.
    volume "workspace" {
      type      = "host"
      source    = var.host_volume
      read_only = false
    }

    restart {
      attempts = 0
      mode     = "fail"
    }

    task "train" {
      driver = "docker"

      volume_mount {
        volume      = "workspace"
        destination = "/workspace"
        read_only   = false
      }

      config {
        image   = var.image
        runtime = "nvidia"

        # PLACEHOLDER entrypoint. Replace with the real trainer, e.g.:
        #   command = "uv"
        #   args    = ["run", "--no-sync", "python", "-m", "src.training.finetune",
        #              "--data", "/workspace/datasets",
        #              "--out",  "/workspace/checkpoints"]
        command = "bash"
        args = [
          "-lc",
          "echo 'sermon-translate training placeholder — wire up the real trainer'; nvidia-smi || true; ls -la /workspace",
        ]
      }

      # Batch training on a single V100 (16 GB). Keep CPU/RAM modest so node-6's
      # services are not starved. Raise deliberately for large fine-tunes and
      # only when the node has headroom.
      resources {
        cpu    = 4000
        memory = 16384

        device "nvidia/gpu" {
          count = var.gpu_count

          constraint {
            attribute = "${device.attr.memory}"
            operator  = ">="
            value     = "16000 MiB"
          }
        }
      }

      env {
        # NVIDIA_VISIBLE_DEVICES is set by the nvidia device plugin to the
        # assigned GPUs; the runtime exposes exactly those, re-indexed from 0.
        # Do NOT set CUDA_VISIBLE_DEVICES=all (CUDA rejects "all" and masks
        # every GPU). Leave it unset, or pin an ordinal for single-GPU runs.
        COMPUTE_DEVICE = "cuda"
        HF_HOME        = "/workspace/hf"
        DATA_DIR       = "/workspace/datasets"
        CHECKPOINT_DIR = "/workspace/checkpoints"
      }
    }
  }
}
