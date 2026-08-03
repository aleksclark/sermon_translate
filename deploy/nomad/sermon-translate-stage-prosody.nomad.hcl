# Prosody stage worker (CPU baseline; optional — can stay in-orchestrator).
#
# SAFETY: declaration only.

variable "image" {
  type    = string
  default = "sermon-translate-server:latest"
}

variable "stage_id" {
  type    = string
  default = "baseline-prosody"
}

variable "model_cache_dir" {
  type    = string
  default = "/models/sermon-translate/models"
}

job "sermon-translate-stage-prosody" {
  datacenters = ["home"]
  region      = "home"
  type        = "service"

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
        type     = "http"
        path     = "/healthz"
        interval = "15s"
        timeout  = "3s"
      }
    }

    task "worker" {
      driver = "docker"

      config {
        image = var.image
        ports = ["ws"]
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
      }
    }
  }
}
