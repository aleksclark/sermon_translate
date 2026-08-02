# GPU inference job for the sermon-translate server.
#
# Runs the translation server on node-6 with one Tesla V100 (16 GB) attached via
# the Nomad nvidia device plugin. A 2-GPU variant is documented inline below.
#
# node-6 facts (2x Tesla V100-SXM2-16GB, 8-core Xeon E5-1620 v4, ~32 GB RAM).
# Total GPU memory on the node is 2 x 16 GB = 32 GB VRAM; a single task here
# requests ONE GPU (16 GB VRAM). Reserve capacity for existing node-6 workloads
# (traefik, syncthing, coredns, emqx, otel, idrive, temp-exporter) before
# submitting -- see deploy/README.md.
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
      resources {
        cpu    = 4000
        memory = 12288

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
        # The nvidia device plugin sets NVIDIA_VISIBLE_DEVICES to the assigned
        # GPU IDs; the nvidia runtime then exposes exactly those GPUs to the
        # container, re-indexed from 0. Do NOT set CUDA_VISIBLE_DEVICES=all --
        # CUDA rejects the literal "all" and would mask every GPU. Leave it
        # unset so CUDA sees all injected devices, or pin an ordinal
        # (e.g. CUDA_VISIBLE_DEVICES=0) to use one GPU out of the assigned set.
        COMPUTE_DEVICE = "cuda"

        CROSSTALK_BASE_URL = var.crosstalk_base_url
        ICE_STUN_URLS      = var.ice_stun_urls
        TURN_URLS          = var.turn_urls
        TURN_USERNAME      = var.turn_username
        TURN_CREDENTIAL    = var.turn_credential

        HF_HOME = "/models/hf"
      }
    }
  }
}
