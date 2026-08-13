# API / WebRTC orchestrator (no GPU required).
# Wires remote stage workers via STAGE_RUNTIME=remote + STAGE_REMOTE_URLS.
#
# SAFETY: declaration only. Do not submit without capacity checks.

variable "image" {
  type        = string
  description = "Server image (CPU is fine; GPU image also works)."
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



variable "crosstalk_base_url" {
  type    = string
  default = ""
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

variable "model_cache_dir" {
  type        = string
  description = "In-container shared model cache path (MooseFS mount)."
  default     = "/models/sermon-translate/models"
}

variable "enable_moosefs_cache" {
  type    = bool
  default = true
}

variable "stage_remote_urls" {
  type        = string
  description = "JSON map of stage_id -> ws://host:port/ws for remote workers."
  default     = "{}"
}

job "sermon-translate-orchestrator" {
  meta {
    managed_by       = "fleet-pull-reconciler"
    deployment_owner = "aleks-clark"
    source_repo      = "aleksclark/sermon_translate"
    contract_state   = "normalized-disabled"
  }

  datacenters = ["home"]
  region      = "home"
  type        = "service"

  group "orchestrator" {
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
      name     = "sermon-translate-orchestrator"
      port     = "http"
      provider = "nomad"

      check {
        type     = "http"
        path     = "/api/stats"
        interval = "15s"
        timeout  = "3s"
      }
    }

    task "orchestrator" {
      driver = "docker"

      config {
        image = local.resolved_image
        ports = ["http"]
      }

      resources {
        cpu    = 1000
        memory = 2048
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
        COMPUTE_DEVICE     = "cpu"
        STAGE_RUNTIME      = "remote"
        STAGE_REMOTE_URLS  = var.stage_remote_urls
        MODEL_CACHE_DIR    = var.model_cache_dir
        HF_HOME            = "${var.model_cache_dir}/huggingface"
        TORCH_HOME         = "${var.model_cache_dir}/torch"
        CROSSTALK_BASE_URL = var.crosstalk_base_url
        ICE_STUN_URLS      = var.ice_stun_urls
        TURN_URLS          = var.turn_urls
        TURN_USERNAME      = var.turn_username
        TURN_CREDENTIAL    = var.turn_credential
      }
    }
  }
}
