# Non-secret fleet overlay for sermon-translate (plan 03 env).
# Secrets (stage auth tokens, TURN credentials) live only in Nomad Variables.
# Source enrollment remains disabled until a separate fleet approval.

datacenters = ["home"]

model_cache_dir = "/models/sermon-translate/models"
host_volume_models = "moosefs"
host_volume_train = "moosefs"

# Placement labels (stable names/meta — not node UUIDs).
gpu_node_class = "gpu"
gpu_model = "Tesla V100-SXM2-16GB"
gpu_count_default = 1

# Non-secret public ICE defaults (credentials via Nomad Variables).
ice_stun_urls = "stun:stun.l.google.com:19302"

# Feature flags for normalized-disabled contract.
contract_enabled = false
recon_dispatch_train = false
