# Prosody stage worker (CPU baseline; optional — can stay in-orchestrator).
# stage.v1 Wave 6 packaging (declaration only).
#
# SAFETY: declaration only. Validate with: nomad job validate <file>

variable "image" {
  type        = string
  description = "Container image reference (tag or registry path)."
  default     = "sermon-translate-server:latest"
}

variable "image_digest" {
  type        = string
  description = "Optional immutable digest (sha256:...). When set, config.image becomes image@digest."
  default     = ""
}

variable "stage_id" {
  type    = string
  default = "baseline-prosody"
}

variable "model_cache_dir" {
  type    = string
  default = "/models/sermon-translate/models"
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

job "sermon-translate-stage-prosody" {
  datacenters = ["home"]
  region      = "home"
  type        = "service"

  meta {
    stage_kind           = "prosody"
    stage_id             = var.stage_id
    health_ready_path    = "/health/ready"
    warm_model_notes     = "CPU baseline; optional colocate with orchestrator"
    private_wss_path     = var.wss_path
    image_digest_set     = var.image_digest != "" ? "true" : "false"
  }

  group "prosody" {
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
      name     = "sermon-stage-prosody"
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
        command = "python"
        args = [
          "-m", "src.runtime.worker",
          "--stage-id", var.stage_id,
          "--host", "0.0.0.0",
          "--port", "8100",
        ]
      }

      resources {
        cpu    = 500
        memory = 1024
      }

      volume_mount {
        volume      = "models"
        destination = "/models"
        read_only   = false
      }

      env {
        COMPUTE_DEVICE  = "cpu"
        STAGE_ID        = var.stage_id
        MODEL_CACHE_DIR = var.model_cache_dir
        STAGE_WSS_PATH  = var.wss_path
        STAGE_AUTH_TOKEN = var.auth_token
      }
    }
  }
}
