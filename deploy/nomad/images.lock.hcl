# Immutable image authority for sermon-translate (digest-only).
#
# BLOCKED: no published multi-node registry digests exist yet for these
# products. Values below are placeholder zero-digests so the lock file shape
# is plan-03 compliant and contract tests can assert digest-only authority.
# Reconciler enrollment and live apply MUST remain disabled until real
# digests are published and this lock is updated in a reviewed PR.
#
# Mutable tags (:latest, :gpu, local bare names) are never deploy authority.

image_sermon_translate_server = "ghcr.io/aleksclark/sermon-translate-server@sha256:0000000000000000000000000000000000000000000000000000000000000000"
image_sermon_translate_stage_worker = "ghcr.io/aleksclark/sermon-translate-stage-worker@sha256:0000000000000000000000000000000000000000000000000000000000000000"
image_sermon_translate_trainer = "ghcr.io/aleksclark/sermon-translate-trainer@sha256:0000000000000000000000000000000000000000000000000000000000000000"

# Component mapping (var names for future renderer wiring):
#   gpu + orchestrator  -> image_sermon_translate_server
#   stage-*             -> image_sermon_translate_stage_worker
#   train (batch)       -> image_sermon_translate_trainer
