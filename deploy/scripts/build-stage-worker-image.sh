#!/usr/bin/env bash
#
# Build an immutable stage.v1 worker image tagged with the git SHA and print
# instructions / values needed to pin Nomad jobs via image@digest.
#
# Usage (from repo root or anywhere):
#   deploy/scripts/build-stage-worker-image.sh
#   REGISTRY=my.registry/sermon-translate-stage-worker ./deploy/scripts/build-stage-worker-image.sh
#   PUSH=1 ./deploy/scripts/build-stage-worker-image.sh
#
# Environment:
#   REGISTRY   Image repository (default AWS ECR path below)
#   TAG        Override tag (default: full git SHA of HEAD)
#   PUSH       If 1/true, docker push after build
#   DOCKER     docker binary (default: docker)
#   UV_SYNC_EXTRA_ARGS  forwarded as build-arg (optional extras)
#
# Does NOT run nomad. Does NOT embed secrets.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
SERVER_DIR="${REPO_ROOT}/server"
DOCKERFILE="${SERVER_DIR}/Dockerfile.stage-worker"

REGISTRY="${REGISTRY:-997533895598.dkr.ecr.us-east-2.amazonaws.com/sermon-translate-stage-worker}"
DOCKER="${DOCKER:-docker}"
PUSH="${PUSH:-0}"
UV_SYNC_EXTRA_ARGS="${UV_SYNC_EXTRA_ARGS:-}"

if [[ ! -f "${DOCKERFILE}" ]]; then
  printf 'ERROR: Dockerfile not found: %s\n' "${DOCKERFILE}" >&2
  exit 1
fi

if [[ ! -d "${SERVER_DIR}" ]]; then
  printf 'ERROR: server/ directory not found: %s\n' "${SERVER_DIR}" >&2
  exit 1
fi

if ! command -v "${DOCKER}" >/dev/null 2>&1; then
  printf 'ERROR: %s not found on PATH\n' "${DOCKER}" >&2
  exit 1
fi

if ! command -v git >/dev/null 2>&1; then
  printf 'ERROR: git not found on PATH\n' >&2
  exit 1
fi

GIT_SHA="$(git -C "${REPO_ROOT}" rev-parse HEAD)"
GIT_SHA_SHORT="$(git -C "${REPO_ROOT}" rev-parse --short=12 HEAD)"
TAG="${TAG:-${GIT_SHA}}"

IMAGE_TAGGED="${REGISTRY}:${TAG}"
IMAGE_SHORT="${REGISTRY}:${GIT_SHA_SHORT}"
IIDFILE="$(mktemp -t stage-worker-iid.XXXXXX)"
trap 'rm -f "${IIDFILE}"' EXIT

printf 'Building stage-worker image\n'
printf '  context:    %s\n' "${SERVER_DIR}"
printf '  dockerfile: %s\n' "${DOCKERFILE}"
printf '  tag:        %s\n' "${IMAGE_TAGGED}"
printf '  also tag:   %s\n' "${IMAGE_SHORT}"
printf '  git sha:    %s\n' "${GIT_SHA}"
if [[ -n "${UV_SYNC_EXTRA_ARGS}" ]]; then
  printf '  uv extras:  %s\n' "${UV_SYNC_EXTRA_ARGS}"
fi
printf '\n'

BUILD_ARGS=(
  --file "${DOCKERFILE}"
  --tag "${IMAGE_TAGGED}"
  --tag "${IMAGE_SHORT}"
  --iidfile "${IIDFILE}"
  --build-arg "UV_SYNC_EXTRA_ARGS=${UV_SYNC_EXTRA_ARGS}"
  --label "org.opencontainers.image.revision=${GIT_SHA}"
  --label "org.opencontainers.image.source=sermon-translate"
  --label "sermon-translate.component=stage-worker"
)

"${DOCKER}" build "${BUILD_ARGS[@]}" "${SERVER_DIR}"

IMAGE_ID="$(tr -d '[:space:]' < "${IIDFILE}")"
if [[ -z "${IMAGE_ID}" ]]; then
  printf 'ERROR: empty image id from --iidfile\n' >&2
  exit 1
fi

printf '\nBuilt image id: %s\n' "${IMAGE_ID}"

# Local image id is sha256:... ; RepoDigests appear after a registry push.
DIGEST_FROM_ID="${IMAGE_ID#sha256:}"
if [[ "${IMAGE_ID}" == sha256:* ]]; then
  LOCAL_DIGEST="sha256:${DIGEST_FROM_ID}"
else
  LOCAL_DIGEST="${IMAGE_ID}"
fi

printf '\n--- Immutable pin values ---\n'
printf 'image=%s\n' "${REGISTRY}"
printf 'image_tag=%s\n' "${TAG}"
printf 'image_ref=%s\n' "${IMAGE_TAGGED}"
printf 'local_image_id=%s\n' "${IMAGE_ID}"
printf 'local_digest=%s\n' "${LOCAL_DIGEST}"

should_push=0
case "${PUSH}" in
  1|true|TRUE|yes|YES|on|ON) should_push=1 ;;
esac

if [[ "${should_push}" -eq 1 ]]; then
  printf '\nPushing %s ...\n' "${IMAGE_TAGGED}"
  "${DOCKER}" push "${IMAGE_TAGGED}"
  "${DOCKER}" push "${IMAGE_SHORT}" || true

  # After push, RepoDigests holds registry@sha256:...
  REPO_DIGEST="$("${DOCKER}" inspect --format='{{index .RepoDigests 0}}' "${IMAGE_TAGGED}" 2>/dev/null || true)"
  if [[ -n "${REPO_DIGEST}" ]]; then
    # Strip registry prefix → sha256:...
    REMOTE_DIGEST="${REPO_DIGEST##*@}"
    printf '\n--- Registry digest (prefer this for Nomad image_digest) ---\n'
    printf 'image=%s\n' "${REGISTRY}"
    printf 'image_digest=%s\n' "${REMOTE_DIGEST}"
    printf 'image_at_digest=%s@%s\n' "${REGISTRY}" "${REMOTE_DIGEST}"
  else
    printf '\nWARN: RepoDigests empty after push; inspect manually:\n' >&2
    printf '  %s inspect --format='\''{{json .RepoDigests}}'\'' %s\n' "${DOCKER}" "${IMAGE_TAGGED}" >&2
  fi
else
  printf '\nNot pushed (set PUSH=1 to push).\n'
  printf 'After push, resolve the immutable registry digest with:\n'
  printf '  %s push %s\n' "${DOCKER}" "${IMAGE_TAGGED}"
  printf '  %s inspect --format='\''{{index .RepoDigests 0}}'\'' %s\n' "${DOCKER}" "${IMAGE_TAGGED}"
  printf '  # → %s@sha256:...\n' "${REGISTRY}"
  printf '\nNomad validate/run example (secrets via -var, never hardcoded):\n'
  printf '  nomad job validate \\\n'
  printf '    -var=image=%s \\\n' "${REGISTRY}"
  printf '    -var=image_digest=sha256:<from-inspect> \\\n'
  printf '    -var=auth_token=\"\$STAGE_AUTH_TOKEN\" \\\n'
  printf '    deploy/nomad/sermon-translate-stage-canary.nomad.hcl\n'
fi

printf '\nOK\n'
