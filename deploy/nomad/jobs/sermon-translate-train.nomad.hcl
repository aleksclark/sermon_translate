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
# PREREQUISITE: node-6 currently advertises neither the docker `nvidia` runtime
# nor any Nomad `nvidia/gpu` devices. Run deploy/scripts/preflight-gpu.sh and
# choose a `gpu_mode` matching what the node actually advertises.
#
# WHERE THE REAL TRAINER GOES: replace the placeholder `args` command below with
# your training entrypoint (e.g. `uv run python -m src.training.finetune ...`).
# Add that module to the training image; do not commit training scripts here
# unless they exist. Datasets and checkpoints belong under the mounted
# /workspace volume so they survive allocation restarts.

variable "image" {
  type        = string
  description = "Training image reference (a GPU image with the trainer baked in)."
  default     = ""
}

variable "image_digest" {
  type        = string
  description = "Immutable digest (sha256:...). Required for reconciliation authority; empty only for local fixture skip."
  default     = ""
}

locals {
  resolved_image = var.image_digest != "" ? "${var.image}@${var.image_digest}" : var.image
}



variable "gpu_count" {
  type        = number
  description = "GPUs to claim for training: 1 (16 GB) or 2 (32 GB, model/data parallel)."
  default     = 1
}

variable "gpu_model" {
  type        = string
  description = <<-EOT
    Exact GPU model the device plugin must assign. node-6 also advertises a
    4 GB M2000 display card; pinning the model (and VRAM floor below) keeps
    the scheduler from handing that card to training.
  EOT
  default     = "Tesla V100-SXM2-16GB"
}

variable "gpu_mode" {
  type        = string
  description = <<-EOT
    How GPUs are attached.

    "device"  - request GPUs from the Nomad nvidia device plugin. Correct and
                preferred, but the job stays pending forever if the plugin is
                not running on the client.
    "runtime" - no device stanza; rely on the docker `nvidia` runtime and
                NVIDIA_VISIBLE_DEVICES. Use only when the device plugin is
                unavailable. Nomad does not track GPU usage in this mode, so a
                training run can silently contend with inference for a GPU.
                Pin visible_devices to V100 ordinals — never leave it as "all"
                while the M2000 is present.
  EOT
  default     = "device"

  validation {
    condition     = contains(["device", "runtime"], var.gpu_mode)
    error_message = "The gpu_mode variable must be either \"device\" or \"runtime\"."
  }
}

variable "visible_devices" {
  type        = string
  description = <<-EOT
    NVIDIA_VISIBLE_DEVICES when gpu_mode is "runtime". Prefer explicit V100
    UUIDs or ordinals (e.g. "GPU-..." or "1,2"), never "all", so the 4 GB
    M2000 display card is never injected into the container.
  EOT
  default     = "0"
}

variable "host_volume" {
  type        = string
  description = "node-6 host volume for datasets/checkpoints (local-data or moosefs)."
  default     = "moosefs"
}

job "sermon-translate-train" {
  meta {
    managed_by           = "fleet-pull-reconciler"
    deployment_owner     = "aleks-clark"
    source_repo          = "aleksclark/sermon_translate"
    contract_state       = "normalized-disabled"
    recon_never_dispatch = "true"
    job_class            = "batch-manual"
  }

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
        image   = local.resolved_image
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

        dynamic "device" {
          for_each = var.gpu_mode == "device" ? [var.gpu_model] : []
          iterator = gpu
          labels   = ["nvidia/gpu/${gpu.value}"]

          content {
            count = var.gpu_count

            constraint {
              attribute = "${device.model}"
              value     = var.gpu_model
            }

            constraint {
              attribute = "${device.attr.memory}"
              operator  = ">="
              value     = "16000 MiB"
            }
          }
        }
      }

      env {
        # In "device" mode the plugin sets NVIDIA_VISIBLE_DEVICES to the
        # assigned GPUs; in "runtime" mode nothing assigns them, so it is set
        # explicitly here. Do NOT set CUDA_VISIBLE_DEVICES=all (CUDA rejects
        # "all" and masks every GPU). Leave it unset, or pin an ordinal for
        # single-GPU runs.
        NVIDIA_VISIBLE_DEVICES = var.gpu_mode == "runtime" ? var.visible_devices : ""

        COMPUTE_DEVICE = "cuda"
        HF_HOME        = "/workspace/hf"
        DATA_DIR       = "/workspace/datasets"
        CHECKPOINT_DIR = "/workspace/checkpoints"
      }
    }
  }
}
