# PRIVATE stage.v1 listen canary — declaration only.
#
# Single count=1 whisper-listen worker for production-integration canary.
# - Service: sermon-stage-canary-listen (Nomad provider only)
# - NO Traefik public tags, NO Cloudflare hostname
# - Prefer gpu_mode=device on meta.gpu=true (node-6); use gpu_mode=cpu if VRAM contended
# - Auth: STAGE_V1_MODE=production requires -var=auth_token=... (never hardcode)
#
# SAFETY: do NOT `nomad job run` from CI. Validate:
#   nomad job validate -var=auth_token=dummy deploy/nomad/sermon-translate-stage-canary.nomad.hcl
#
# Submit (operator, after preflight + image push):
#   nomad job run \
#     -var=image=997533895598.dkr.ecr.us-east-2.amazonaws.com/sermon-translate-stage-worker \
#     -var=image_digest=sha256:... \
#     -var=auth_token="$STAGE_AUTH_TOKEN" \
#     -var=trust_proxy=true \
#     deploy/nomad/sermon-translate-stage-canary.nomad.hcl

variable "image" {
  type        = string
  description = "Container image reference (tag or registry path without digest)."
  default     = "997533895598.dkr.ecr.us-east-2.amazonaws.com/sermon-translate-stage-worker"
}

variable "image_digest" {
  type        = string
  description = "Optional immutable digest (sha256:...). When set, config.image becomes image@digest."
  default     = ""
}

variable "stage_id" {
  type        = string
  description = "Registered listen stage id for canary (warm product default: whisper-listen)."
  default     = "whisper-listen"
}

variable "model_cache_dir" {
  type        = string
  description = "Container path for model cache. Host moosefs mounts at /models, so /models/tmp/... maps to /mnt/moosefs/tmp/..."
  default     = "/models/tmp/sermon-translate/models"
}

variable "gpu_count" {
  type    = number
  default = 1
}

variable "gpu_model" {
  type    = string
  default = "Tesla V100-SXM2-16GB"
}

variable "gpu_mode" {
  type        = string
  description = "device preferred on node-6; cpu fallback when GPU contended."
  default     = "device"

  validation {
    condition     = contains(["device", "runtime", "cpu"], var.gpu_mode)
    error_message = "Gpu mode must be one of: device, runtime, or cpu."
  }
}

variable "visible_devices" {
  type    = string
  default = "0"
}

variable "auth_token" {
  type        = string
  description = "Workload bearer token injected as STAGE_AUTH_TOKEN. Pass via -var; never commit secrets."
  default     = ""
}

variable "wss_path" {
  type        = string
  description = "Live stage.v1 WebSocket path (private; not a public edge route)."
  default     = "/stage/v1/stream"
}

variable "stage_v1_mode" {
  type        = string
  description = "STAGE_V1_MODE: production|dev|test. Canary defaults to production (fail-closed auth)."
  default     = "production"

  validation {
    condition     = contains(["production", "prod", "dev", "test"], var.stage_v1_mode)
    error_message = "Stage v1 mode must be one of: production, prod, dev, or test."
  }
}

variable "trust_proxy" {
  type        = bool
  description = "STAGE_TRUST_PROXY — honor X-Forwarded-Proto from trusted internal reverse proxies."
  default     = true
}

variable "node_name" {
  type        = string
  description = "Optional hard pin to a node name (empty = any meta.gpu=true when gpu_mode!=cpu)."
  default     = ""
}

locals {
  resolved_image = var.image_digest != "" ? "${var.image}@${var.image_digest}" : var.image
}

job "sermon-translate-stage-canary" {
  datacenters = ["home"]
  region      = "home"
  type        = "service"

  meta {
    stage_kind        = "listen"
    stage_id          = var.stage_id
    canary            = "true"
    health_ready_path = "/health/ready"
    private_wss_path  = var.wss_path
    stage_v1_mode     = var.stage_v1_mode
    image_digest_set  = var.image_digest != "" ? "true" : "false"
    public_edge       = "false"
    note              = "PRIVATE canary — fleet/Nomad service only; no Cloudflare/Traefik public hostname"
  }

  # Prefer GPU nodes for whisper-listen; skip constraint when cpu fallback.
  dynamic "constraint" {
    for_each = var.gpu_mode == "cpu" ? [] : [1]
    content {
      attribute = "${meta.gpu}"
      value     = "true"
      operator  = "="
    }
  }

  dynamic "constraint" {
    for_each = var.node_name != "" ? [var.node_name] : []
    content {
      attribute = "${node.unique.name}"
      value     = constraint.value
      operator  = "="
    }
  }

  group "canary-listen" {
    count = 1

    # Mark allocation meta for operators / discovery filters.
    meta {
      canary   = "true"
      stage_id = var.stage_id
    }

    volume "models" {
      type      = "host"
      source    = "moosefs"
      read_only = false
    }

    network {
      port "ws" {
        to = 8100
      }
    }

    # PRIVATE service — Nomad provider only. No traefik tags, no public host.
    service {
      name     = "sermon-stage-canary-listen"
      port     = "ws"
      provider = "nomad"

      tags = [
        "canary",
        "stage.v1",
        "private",
        "stage-kind-listen",
      ]

      check {
        name     = "ready"
        type     = "http"
        path     = "/health/ready"
        interval = "10s"
        timeout  = "3s"
      }

      check {
        name     = "live"
        type     = "http"
        path     = "/health/live"
        interval = "15s"
        timeout  = "2s"
      }
    }

    task "worker" {
      driver = "docker"

      config {
        image   = local.resolved_image
        ports   = ["ws"]
        runtime = var.gpu_mode == "cpu" ? "runc" : "nvidia"
        # Image ENTRYPOINT is the worker module; still pass explicit args so
        # non-entrypoint images (legacy GPU server) keep working.
        command = "python"
        args = [
          "-m", "src.runtime.worker",
          "--stage-id", var.stage_id,
          "--host", "0.0.0.0",
          "--port", "8100",
        ]
      }

      resources {
        cpu    = var.gpu_mode == "cpu" ? 2000 : 2000
        memory = var.gpu_mode == "cpu" ? 4096 : 6144

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

      volume_mount {
        volume      = "models"
        destination = "/models"
        read_only   = false
      }

      env {
        NVIDIA_VISIBLE_DEVICES = var.gpu_mode == "runtime" ? var.visible_devices : ""
        COMPUTE_DEVICE         = var.gpu_mode == "cpu" ? "cpu" : "cuda"
        STAGE_ID               = var.stage_id
        MODEL_CACHE_DIR        = var.model_cache_dir
        HF_HOME                = "${var.model_cache_dir}/huggingface"
        TORCH_HOME             = "${var.model_cache_dir}/torch"
        WHISPER_MODEL_SIZE     = "base"
        STAGE_WSS_PATH         = var.wss_path
        STAGE_AUTH_TOKEN       = var.auth_token
        STAGE_V1_MODE          = var.stage_v1_mode
        STAGE_TRUST_PROXY      = var.trust_proxy ? "true" : "false"
      }
    }
  }
}
