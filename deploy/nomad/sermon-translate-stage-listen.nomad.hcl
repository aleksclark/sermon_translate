# Listen (ASR) stage worker. Prefer a V100 when real models are loaded.
#
# SAFETY: declaration only. Run deploy/scripts/preflight-gpu.sh first.

variable "image" {
  type    = string
  default = "sermon-translate-server:gpu"
}

variable "stage_id" {
  type        = string
  description = "Registered listen stage id."
  default     = "passthrough-listen"
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
    error_message = "gpu_mode must be device, runtime, or cpu."
  }
}

variable "visible_devices" {
  type    = string
  default = "0"
}

job "sermon-translate-stage-listen" {
  datacenters = ["home"]
  region      = "home"
  type        = "service"

  constraint {
    attribute = "${meta.gpu}"
    value     = "true"
    operator  = "="
  }

  group "listen" {
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
      name     = "sermon-stage-listen"
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
        image   = var.image
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
      }
    }
  }
}
