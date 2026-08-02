#!/usr/bin/env bash
#
# Read-only preflight for the sermon-translate GPU jobs.
#
# Performs GET-only queries against the Nomad HTTP API. It never submits,
# plans, stops, drains, or otherwise mutates anything, so it is safe to run
# against a cluster with live workloads.
#
# Usage:
#   NOMAD_ADDR=http://192.168.0.99:4646 deploy/scripts/preflight-gpu.sh [node-name]
#
# Environment:
#   NOMAD_ADDR    Nomad API address (default http://127.0.0.1:4646)
#   NOMAD_TOKEN   Optional ACL token
#   GPU_COUNT     GPUs the job requests (default 1)
#   GPU_MODEL     Exact GPU model the job pins (default Tesla V100-SXM2-16GB)
#   TASK_CPU      CPU MHz the job reserves (default 4000)
#   TASK_MEMORY   Memory MB the job reserves (default 12288)
#   MIN_VRAM_MIB  Per-GPU VRAM the job constraint demands (default 16000)

set -euo pipefail

NODE_NAME="${1:-${NODE_NAME:-node-6}}"
NOMAD_ADDR="${NOMAD_ADDR:-http://127.0.0.1:4646}"
GPU_COUNT="${GPU_COUNT:-1}"
GPU_MODEL="${GPU_MODEL:-Tesla V100-SXM2-16GB}"
TASK_CPU="${TASK_CPU:-4000}"
TASK_MEMORY="${TASK_MEMORY:-12288}"
MIN_VRAM_MIB="${MIN_VRAM_MIB:-16000}"

failures=0
warnings=0

pass() { printf 'PASS  %s\n' "$1"; }
warn() { printf 'WARN  %s\n' "$1"; warnings=$((warnings + 1)); }
fail() { printf 'FAIL  %s\n' "$1"; failures=$((failures + 1)); }
info() { printf '      %s\n' "$1"; }

need() {
  command -v "$1" >/dev/null 2>&1 || {
    printf 'FAIL  required tool not found: %s\n' "$1" >&2
    exit 2
  }
}

need curl
need jq

api() {
  local path="$1"
  if [ -n "${NOMAD_TOKEN:-}" ]; then
    curl -fsS --max-time 15 -H "X-Nomad-Token: ${NOMAD_TOKEN}" "${NOMAD_ADDR}${path}"
  else
    curl -fsS --max-time 15 "${NOMAD_ADDR}${path}"
  fi
}

printf 'Preflight for GPU jobs on %s via %s\n\n' "${NODE_NAME}" "${NOMAD_ADDR}"

if ! nodes="$(api /v1/nodes 2>/dev/null)"; then
  fail "cannot reach the Nomad API at ${NOMAD_ADDR}"
  info "set NOMAD_ADDR to the cluster address (and NOMAD_TOKEN if ACLs are on)"
  exit 1
fi
pass "Nomad API reachable"

node_id="$(printf '%s' "${nodes}" | jq -r --arg n "${NODE_NAME}" \
  'map(select(.Name == $n)) | .[0].ID // empty')"

if [ -z "${node_id}" ]; then
  fail "node ${NODE_NAME} not found in the cluster"
  exit 1
fi
pass "node ${NODE_NAME} found (${node_id})"

node="$(api "/v1/node/${node_id}")"

status="$(printf '%s' "${node}" | jq -r '.Status // "unknown"')"
eligibility="$(printf '%s' "${node}" | jq -r '.SchedulingEligibility // "unknown"')"
drain="$(printf '%s' "${node}" | jq -r '.Drain // false')"

[ "${status}" = "ready" ] && pass "node status is ready" || fail "node status is ${status}, expected ready"
[ "${eligibility}" = "eligible" ] && pass "node is scheduling-eligible" \
  || fail "node scheduling eligibility is ${eligibility}, expected eligible"
[ "${drain}" = "false" ] && pass "node is not draining" || fail "node is draining"

runtimes="$(printf '%s' "${node}" \
  | jq -r '.Drivers.docker.Attributes["driver.docker.runtimes"] // ""')"
if printf '%s' "${runtimes}" | grep -q 'nvidia'; then
  pass "docker nvidia runtime is registered"
  info "runtimes: ${runtimes}"
else
  fail "docker nvidia runtime is NOT registered (runtimes: ${runtimes:-none})"
  info "the container cannot see any GPU without it"
fi

gpu_devices="$(printf '%s' "${node}" \
  | jq '[(.NodeResources.Devices // [])[] | select(.Vendor == "nvidia")]')"
gpu_groups="$(printf '%s' "${gpu_devices}" | jq 'length')"
gpu_total="$(printf '%s' "${gpu_devices}" | jq '[.[].Instances | length] | add // 0')"

if [ "${gpu_groups}" -eq 0 ]; then
  fail "the Nomad nvidia DEVICE PLUGIN is not exposing any GPUs on this node"
  info "the docker nvidia runtime alone does NOT satisfy a device \"nvidia/gpu\" stanza"
  info "a job requesting devices will sit in pending with no placement forever"
  info "remedy: enable the nomad-device-nvidia plugin on the ${NODE_NAME} client,"
  info "or submit the runtime-only variant (see deploy/README.md)"
else
  pass "nvidia device plugin exposes ${gpu_total} GPU(s) in ${gpu_groups} device group(s)"
  info "advertised models:"
  printf '%s' "${gpu_devices}" | jq -r \
    '.[] | "      \(.Name) x\(.Instances | length)  memory=\(.Attributes.memory)"'

  model_match="$(printf '%s' "${gpu_devices}" | jq --arg model "${GPU_MODEL}" '
    [ .[]
      | select(.Name == $model)
      | .Instances | length
    ] | add // 0')"

  if [ "${model_match}" -ge "${GPU_COUNT}" ]; then
    pass "${model_match} x ${GPU_MODEL} satisfy the model pin (need ${GPU_COUNT})"
  elif [ "${model_match}" -eq 0 ]; then
    fail "no GPU named \"${GPU_MODEL}\" is advertised"
    info "the job pins device.model so the M2000 (4 GB) is never assigned"
  else
    fail "only ${model_match} x ${GPU_MODEL} advertised, job requests ${GPU_COUNT}"
  fi

  eligible="$(printf '%s' "${gpu_devices}" | jq --argjson min "${MIN_VRAM_MIB}" --arg model "${GPU_MODEL}" '
    [ .[]
      | select(.Name == $model)
      | (.Attributes.memory.IntNumeratorVal // .Attributes.memory.Int // .Attributes.memory // 0) as $m
      | ($m | tonumber? // 0) as $mib
      | select($mib >= $min)
      | .Instances | length
    ] | add // 0')"

  if [ "${eligible}" -ge "${GPU_COUNT}" ]; then
    pass "${eligible} x ${GPU_MODEL} meet the >= ${MIN_VRAM_MIB} MiB VRAM floor"
  else
    fail "only ${eligible} x ${GPU_MODEL} report >= ${MIN_VRAM_MIB} MiB VRAM"
  fi

  other="$(printf '%s' "${gpu_devices}" | jq --arg model "${GPU_MODEL}" '
    [ .[] | select(.Name != $model) | "\(.Name) x\(.Instances | length)" ] | join(", ")')"
  if [ -n "${other}" ]; then
    warn "other GPUs present (excluded by model pin): ${other}"
    info "runtime mode must pin visible_devices to V100 ordinals/UUIDs, never \"all\""
  fi
fi

total_cpu="$(printf '%s' "${node}" | jq -r '.NodeResources.Cpu.CpuShares // 0')"
total_mem="$(printf '%s' "${node}" | jq -r '.NodeResources.Memory.MemoryMB // 0')"

allocs="$(api "/v1/node/${node_id}/allocations")"
used="$(printf '%s' "${allocs}" | jq '
  [ .[]
    | select(.ClientStatus == "running" or .DesiredStatus == "run")
    | .AllocatedResources.Tasks // {}
    | to_entries
    | map(.value.Cpu.CpuShares // 0) as $c
    | map(.value.Memory.MemoryMB // 0) as $m
    | {cpu: ($c | add // 0), mem: ($m | add // 0)}
  ]
  | {cpu: (map(.cpu) | add // 0), mem: (map(.mem) | add // 0)}')"

used_cpu="$(printf '%s' "${used}" | jq -r '.cpu')"
used_mem="$(printf '%s' "${used}" | jq -r '.mem')"
free_cpu=$((total_cpu - used_cpu))
free_mem=$((total_mem - used_mem))

info "node capacity: ${total_cpu} MHz CPU, ${total_mem} MB RAM"
info "allocated:     ${used_cpu} MHz CPU, ${used_mem} MB RAM"
info "free:          ${free_cpu} MHz CPU, ${free_mem} MB RAM"

if [ "${free_cpu}" -ge "${TASK_CPU}" ]; then
  pass "free CPU (${free_cpu} MHz) covers the ${TASK_CPU} MHz reservation"
else
  fail "free CPU (${free_cpu} MHz) is below the ${TASK_CPU} MHz reservation"
fi

if [ "${free_mem}" -ge "${TASK_MEMORY}" ]; then
  pass "free memory (${free_mem} MB) covers the ${TASK_MEMORY} MB reservation"
else
  fail "free memory (${free_mem} MB) is below the ${TASK_MEMORY} MB reservation"
fi

running="$(printf '%s' "${allocs}" \
  | jq -r '[.[] | select(.ClientStatus == "running") | .JobID] | unique | join(", ")')"
info "co-located jobs already on ${NODE_NAME}: ${running:-none}"

printf '\n'
if [ "${failures}" -gt 0 ]; then
  printf 'RESULT: FAIL (%d blocking, %d warning)\n' "${failures}" "${warnings}"
  exit 1
fi
if [ "${warnings}" -gt 0 ]; then
  printf 'RESULT: WARN (%d warning) - review before submitting\n' "${warnings}"
  exit 0
fi
printf 'RESULT: PASS - prerequisites satisfied\n'
