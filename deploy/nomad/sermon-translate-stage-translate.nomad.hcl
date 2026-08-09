# Translate stage worker — stage.v1 Wave 6 packaging (declaration only).
#
# Defaults target warm opus-mt-en-es. Models load once via StageHost (D6).
# /health/ready is the admission gate.
#
# SAFETY: declaration only. Validate with: nomad job validate <file>

variable "image" {
  type        = string
  description = "Container image reference (tag or registry path)."
  default     = "sermon-translate-server:gpu"
}

variable "image_digest" {
  type        = string
  description = "Optional immutable digest (sha256:...). When set, config.image becomes image@digest."
  default     = ""
}

variable "stage_id" {
  type        = string
  description = "Registered translate stage id (warm product default: opus-mt-en-es)."
  default     = "opus-mt-en-es"
}

variable "model_cache_dir" {
  type    = string
  default = "/models/sermon-translate/models"
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
  type    = string
  default = "device"

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
  description = "Optional bearer token for private WSS (placeholder)."
  default     = ""
}

variable "wss_path" {
  type        = string
  description = "Private WebSocket path placeholder for stage protocol."
  default     = "/stage/v1/ws"
}

locals {
  resolved_image = var.image_digest != "" ? "${var.image}@${var.image_digest}" : var.image
}

job "sermon-translate-stage-translate" {
  datacenters = ["home"]
  region      = "home"
  type        = "service"

  meta {
    stage_kind           = "translate"
    stage_id             = var.stage_id
    health_ready_path    = "/health/ready"
    warm_model_notes     = "StageHost loads opus-mt once; sessions bind via adapters.open_opus_mt_session_stage"
    private_wss_path     = var.wss_path
    image_digest_set     = var.image_digest != "" ? "true" : "false"
  }

  constraint {
    attribute = "${meta.gpu}"
    value     = "true"
  }

  group "translate" {
    count = 1

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

    service {
      name     = "sermon-stage-translate"
      port     = "ws"
      provider = "nomad"

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
        command = "python"
        args = [
          "-m", "src.runtime.worker",
          "--stage-id", var.stage_id,
          "--host", "0.0.0.0",
          "--port", "8100",
        ]
      }

      resources {
        cpu    = 2000
        memory = 6144

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
        TRANSLATE_MODEL_ID     = "Helsinki-NLP/opus-mt-en-es"
        STAGE_WSS_PATH         = var.wss_path
        STAGE_AUTH_TOKEN       = var.auth_token
      }
    }
  }
}
