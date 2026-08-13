#!/usr/bin/env bash
# L0/L1 static contract checks for sermon-translate (no Nomad credentials, no submit).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
NOMAD_DIR="${ROOT}/deploy/nomad"
MANIFEST="${NOMAD_DIR}/deployment.yaml"
LOCK="${NOMAD_DIR}/images.lock.hcl"
ENVF="${NOMAD_DIR}/env/home.nomadvars.hcl"
CODEOWNERS="${ROOT}/.github/CODEOWNERS"
EXPECTED="${NOMAD_DIR}/tests/expected-services.json"
JOBS_DIR="${NOMAD_DIR}/jobs"

fail() { echo "FAIL: $*" >&2; exit 1; }
pass() { echo "OK: $*"; }

[[ -f "$MANIFEST" ]] || fail "missing $MANIFEST"
[[ -f "$LOCK" ]] || fail "missing $LOCK"
[[ -f "$ENVF" ]] || fail "missing $ENVF"
[[ -f "$CODEOWNERS" ]] || fail "missing CODEOWNERS"
[[ -f "$EXPECTED" ]] || fail "missing expected-services.json"
[[ -d "$JOBS_DIR" ]] || fail "missing jobs/"

REQUIRED_JOBS=(
  sermon-translate-gpu
  sermon-translate-orchestrator
  sermon-translate-stage-listen
  sermon-translate-stage-prosody
  sermon-translate-stage-speak
  sermon-translate-stage-translate
  sermon-translate-train
)

for id in "${REQUIRED_JOBS[@]}"; do
  [[ -f "${JOBS_DIR}/${id}.nomad.hcl" ]] || fail "missing job ${id}"
done
pass "seven jobspecs present"

python3 - <<'PY' "$MANIFEST" "$EXPECTED"
import json, sys
from pathlib import Path
data = json.loads(Path(sys.argv[1]).read_text())
exp = json.loads(Path(sys.argv[2]).read_text())
req = ["schema_version","project","owner","repository","ref_policy","namespace","datacenters","release_sets"]
for k in req:
    assert k in data, k
assert data["schema_version"] == 1
assert data["owner"] == "aleks-clark"
assert data["repository"] == "https://github.com/aleksclark/sermon_translate"
assert data["ref_policy"] == "signed-default-branch-commit"
assert data["namespace"] == "default"
assert data["project"] == "sermon-translate"
allowed = set(req)
extra = set(data) - allowed
assert not extra, extra
sets = {rs["name"]: rs for rs in data["release_sets"]}
assert "sermon-translate-runtime" in sets
assert "sermon-translate-train" in sets
rt = sets["sermon-translate-runtime"]
tr = sets["sermon-translate-train"]
for rs in (rt, tr):
    assert rs["env"] == "env/home.nomadvars.hcl"
    assert rs["images"] == "images.lock.hcl"
    assert rs["rollout"] == "serial"
    assert rs["prune"] == "explicit-only"
rt_ids = [j["id"] for j in rt["jobs"]]
assert rt_ids == [
    "sermon-translate-gpu",
    "sermon-translate-orchestrator",
    "sermon-translate-stage-listen",
    "sermon-translate-stage-prosody",
    "sermon-translate-stage-speak",
    "sermon-translate-stage-translate",
], rt_ids
for j in rt["jobs"]:
    assert j["spec"] == f"jobs/{j['id']}.nomad.hcl"
assert [j["id"] for j in tr["jobs"]] == ["sermon-translate-train"]
assert tr["jobs"][0]["spec"] == "jobs/sermon-translate-train.nomad.hcl"
assert exp["fleet_source_enabled"] is False
assert exp["recon_dispatch_train"] is False
assert exp["contract_state"] == "normalized-disabled"
print("manifest ok")
PY
pass "deployment.yaml plan03 schema"

# Digest-only lock; reject latest as assignment
grep -E '@sha256:[0-9a-f]{64}' "$LOCK" >/dev/null || fail "lock missing digests"
if grep -E '^\s*image_[a-z0-9_]+\s*=\s*"[^"]*:latest"' "$LOCK" >/dev/null; then
  fail "lock uses :latest"
fi
grep -F 'BLOCKED' "$LOCK" >/dev/null || fail "lock must document blocked digests"
pass "images.lock.hcl digest-only + blocked note"

# Env non-secret
if grep -Ei '^\s*(password|secret|token|private_key)\s*=' "$ENVF" >/dev/null; then
  fail "env overlay secret-like assignment"
fi
grep -F 'contract_enabled = false' "$ENVF" >/dev/null || fail "env must mark contract_enabled false"
grep -F 'recon_dispatch_train = false' "$ENVF" >/dev/null || fail "env must disable train dispatch"
pass "env overlay"

# Job invariants
for id in "${REQUIRED_JOBS[@]}"; do
  job="${JOBS_DIR}/${id}.nomad.hcl"
  grep -F "job \"${id}\"" "$job" >/dev/null || fail "$id job id"
  grep -F 'managed_by' "$job" >/dev/null || fail "$id meta"
  grep -F 'local.resolved_image' "$job" >/dev/null || fail "$id resolved_image"
  if grep -E 'image\s*=\s*"[^"]*:latest"' "$job" >/dev/null; then
    fail "$id :latest image line"
  fi
  if grep -E 'default\s*=\s*"[^"]*:latest"' "$job" >/dev/null; then
    fail "$id :latest default"
  fi
  # no parameterized (train must not be dispatchable by recon)
  if grep -E '^\s*parameterized\s*\{' "$job" >/dev/null; then
    fail "$id must not be parameterized"
  fi
done
grep -F 'type        = "batch"' "${JOBS_DIR}/sermon-translate-train.nomad.hcl" >/dev/null \
  || grep -E 'type\s*=\s*"batch"' "${JOBS_DIR}/sermon-translate-train.nomad.hcl" >/dev/null \
  || fail "train must be batch"
grep -F 'recon_never_dispatch' "${JOBS_DIR}/sermon-translate-train.nomad.hcl" >/dev/null \
  || fail "train must mark recon_never_dispatch"
pass "jobspec invariants"

grep -E '^/deploy/nomad/' "$CODEOWNERS" | grep -F '@aleksclark' >/dev/null || fail "CODEOWNERS"
pass "CODEOWNERS"

# README must say disabled / no live deploy
README="${NOMAD_DIR}/README.md"
grep -Fi 'enabled: false' "$README" >/dev/null || grep -Fi 'disabled' "$README" >/dev/null || fail "README disabled"
grep -Fi 'no live' "$README" >/dev/null || grep -Fi 'forbidden' "$README" >/dev/null || fail "README no live deploy"
pass "README disabled/no-live"

# No workflow nomad submit
if [[ -d "${ROOT}/.github/workflows" ]]; then
  if rg -n 'nomad job (run|dispatch) ' "${ROOT}/.github/workflows" -g '*.yml' >/dev/null 2>&1; then
    fail "workflows must not nomad job run/dispatch"
  fi
fi
pass "no nomad submit in workflows"

# Optional nomad L0
if command -v nomad >/dev/null 2>&1; then
  nomad fmt -check "${JOBS_DIR}" || fail "nomad fmt"
  PH="sha256:0000000000000000000000000000000000000000000000000000000000000000"
  for id in "${REQUIRED_JOBS[@]}"; do
    job="${JOBS_DIR}/${id}.nomad.hcl"
    args=(job run -output -var "image=ghcr.io/aleksclark/sermon-translate-server" -var "image_digest=${PH}")
    if grep -q 'variable "auth_token"' "$job"; then
      args+=(-var "auth_token=dummy")
    fi
    args+=("$job")
    nomad "${args[@]}" >/dev/null || fail "nomad parse $id"
  done
  pass "nomad L0 parse"
else
  echo "SKIP: nomad CLI not installed"
fi

pass "all sermon contract checks (normalized-disabled)"
