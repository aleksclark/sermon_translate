# GPU inference job for the sermon-translate server.
#
# Runs the translation server on node-6 with Tesla V100 (16 GB) GPUs attached.
#
# node-6 facts (2x Tesla V100-SXM2-16GB, 8-core Xeon E5-1620 v4, ~32 GB RAM).
# Total GPU memory on the node is 2 x 16 GB = 32 GB VRAM; a single task here
# requests ONE GPU (16 GB VRAM) by default. Reserve capacity for existing
# node-6 workloads (traefik, syncthing, coredns, emqx, otel, idrive,
# temp-exporter) before submitting -- see deploy/README.md.
#
# PREREQUISITE: node-6 is NOT currently provisioned for GPU containers. As
# verified read-only against the cluster, it advertises neither the docker
# `nvidia` runtime nor any Nomad `nvidia/gpu` devices; only node Meta claims
# GPUs. Run deploy/scripts/preflight-gpu.sh before submitting and pick a
# `gpu_mode` that matches what the node actually advertises.
#
# SAFETY: this file only DECLARES the job. Submitting it (`nomad job run`) will
# schedule a container onto node-6 alongside the workloads above. Verify free
# CPU/RAM/GPU capacity first.

variable "image" {
  type        = string
  description = "Full image reference for the GPU server, e.g. registry/sermon-translate-server:gpu"
  default     = "sermon-translate-server:gpu"
}

variable "crosstalk_base_url" {
  type        = string
  description = "Base URL of the Crosstalk service the server integrates with."
  default     = ""
}

variable "model_cache_dir" {
  type        = string
  description = "In-container path for shared model weights (MooseFS mount)."
  default     = "/models/sermon-translate/models"
}

variable "enable_moosefs_cache" {
  type        = bool
  description = "Mount the host moosefs volume for MODEL_CACHE_DIR sharing."
  default     = true
}

variable "ice_stun_urls" {
  type    = string
  default = "stun:stun.l.google.com:19302"
}

variable "turn_urls" {
  type    = string
  default = ""
}

variable "turn_username" {
  type    = string
  default = ""
}

variable "turn_credential" {
  type    = string
  default = ""
}

variable "gpu_count" {
  type        = number
  description = "GPUs to attach (1 fits a single V100; set 2 to reserve both)."
  default     = 1
}

variable "gpu_model" {
  type        = string
  description = <<-EOT
    Exact GPU model the device plugin must assign. node-6 also advertises a
    4 GB M2000 display card; pinning the model (and VRAM floor below) keeps
    the scheduler from handing that card to inference or training.
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
                unavailable. Nomad does not track GPU usage in this mode, so
                two such jobs can silently contend for the same GPU. Pin
                visible_devices to V100 ordinals — never leave it as "all"
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

job "sermon-translate-gpu" {
  datacenters = ["home"]
  region      = "home"
  type        = "service"

  # Pin to node-6. Either constraint alone is sufficient; both are shown so an
  # operator can relax to meta-based scheduling if more GPU nodes appear.
  constraint {
    attribute = "${node.unique.name}"
    value     = "node-6"
  }

  constraint {
    attribute = "${meta.gpu}"
    value     = "true"
  }

  group "server" {
    count = 1

    shutdown_delay = "5s"

    dynamic "volume" {
      for_each = var.enable_moosefs_cache ? [1] : []
      labels   = ["models"]

      content {
        type      = "host"
        source    = "moosefs"
        read_only = false
      }
    }

    network {
      port "http" {
        to = 8000
      }
    }

    service {
      name     = "sermon-translate-gpu"
      port     = "http"
      provider = "nomad"

      check {
        type     = "http"
        path     = "/api/stats"
        interval = "15s"
        timeout  = "3s"
      }
    }

    task "server" {
      driver = "docker"

      config {
        image = var.image
        ports = ["http"]
        # The nvidia container runtime must be registered on the client. When
        # the device stanza assigns GPUs, Nomad injects NVIDIA_VISIBLE_DEVICES.
        runtime = "nvidia"
      }

      # Request GPUs from the nvidia device plugin. One V100 = 16 GB VRAM.
      # For the 2-GPU variant, set the `gpu_count` variable to 2 (both V100s,
      # 32 GB total) -- only do this if no other GPU workload needs node-6.
      # Skipped entirely when gpu_mode is "runtime", because a device stanza
      # the plugin cannot satisfy blocks placement instead of failing loudly.
      resources {
        cpu    = 4000
        memory = 12288

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

      dynamic "volume_mount" {
        for_each = var.enable_moosefs_cache ? [1] : []
        content {
          volume      = "models"
          destination = "/models"
          read_only   = false
        }
      }

      env {
        # In "device" mode the plugin sets NVIDIA_VISIBLE_DEVICES to the
        # assigned GPU IDs; in "runtime" mode nothing assigns them, so it is
        # set explicitly here. Either way the nvidia runtime exposes exactly
        # those GPUs to the container, re-indexed from 0.
        #
        # Do NOT set CUDA_VISIBLE_DEVICES=all -- CUDA rejects the literal "all"
        # and would mask every GPU. Leave it unset so CUDA sees all injected
        # devices, or pin an ordinal (e.g. CUDA_VISIBLE_DEVICES=0).
        NVIDIA_VISIBLE_DEVICES = var.gpu_mode == "runtime" ? var.visible_devices : ""

        COMPUTE_DEVICE = "cuda"

        CROSSTALK_BASE_URL = var.crosstalk_base_url
        ICE_STUN_URLS      = var.ice_stun_urls
        TURN_URLS          = var.turn_urls
        TURN_USERNAME      = var.turn_username
        TURN_CREDENTIAL    = var.turn_credential

        MODEL_CACHE_DIR = var.model_cache_dir
        HF_HOME         = "${var.model_cache_dir}/huggingface"
        TORCH_HOME      = "${var.model_cache_dir}/torch"
      }
    }
  }
}
