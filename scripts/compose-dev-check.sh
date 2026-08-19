#!/usr/bin/env bash
# Fail-closed validation for sermon-translate docker-compose.dev.yml.
# Uses `docker compose config --format json` into a mode-0700 temp file; never dumps full env/secrets.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_SLUG="sermon-translate"

die() { echo "compose-dev-check: FAIL: $*" >&2; exit 1; }
ok() { echo "compose-dev-check: ok: $*" >&2; }
info() { echo "compose-dev-check: $*" >&2; }

# Do not inherit an operator's compose file/profile/project accidentally.
unset COMPOSE_FILE COMPOSE_PROFILES COMPOSE_PROJECT_NAME || true
COMPOSE_FILE="${ROOT}/docker-compose.dev.yml"

sanitize_instance() {
  local s="${1:-}"
  s="$(printf '%s' "$s" | tr '[:upper:]' '[:lower:]')"
  s="$(printf '%s' "$s" | sed -E 's/[^a-z0-9-]+/-/g; s/-+/-/g; s/^-+//; s/-+$//')"
  if [[ ${#s} -gt 48 ]]; then
    s="${s:0:48}"
    s="$(printf '%s' "$s" | sed -E 's/-+$//')"
  fi
  [[ -z "$s" ]] && s="dev"
  printf '%s' "$s"
}

derive_instance() {
  if [[ -n "${STACKLANE_INSTANCE:-}" ]]; then
    sanitize_instance "$STACKLANE_INSTANCE"
    return
  fi
  local wt
  wt="$(basename "$ROOT")"
  if [[ -n "$wt" && "$wt" != "." && "$wt" != "/" ]]; then
    sanitize_instance "$wt"
    return
  fi
  local branch=""
  if command -v git >/dev/null 2>&1; then
    branch="$(git -C "$ROOT" rev-parse --abbrev-ref HEAD 2>/dev/null || true)"
  fi
  if [[ -n "$branch" && "$branch" != "HEAD" ]]; then
    sanitize_instance "$branch"
    return
  fi
  sanitize_instance "dev"
}

require_tools() {
  command -v docker >/dev/null 2>&1 || die "docker not found"
  docker compose version >/dev/null 2>&1 || die "docker compose not available"
  command -v python3 >/dev/null 2>&1 || die "python3 required for JSON parse"
  [[ -f "$COMPOSE_FILE" ]] || die "missing $COMPOSE_FILE"
}

render_config() {
  local project="$1"
  local instance="$2"
  local out="$3"
  local errf="$4"
  local base_dom="${STACKLANE_BASE_DOMAIN:-test}"
  umask 077
  : >"$out"
  : >"$errf"
  chmod 600 "$out" "$errf"
  if ! timeout 60s env \
    STACKLANE_INSTANCE="$instance" \
    STACKLANE_BASE_DOMAIN="$base_dom" \
    COMPOSE_PROJECT_NAME="$project" \
    docker compose -p "$project" --project-directory "$ROOT" -f "$COMPOSE_FILE" \
      config --format json >"$out" 2>"$errf"; then
    echo "FAIL: compose-config-render" >&2
    exit 1
  fi
  chmod 600 "$out"
}

run_mutation_probes() {
  local tmpdir="$1"
  local base_yml="$COMPOSE_FILE"
  python3 - "$base_yml" "$tmpdir" "$ROOT" <<'PY'
import json, os, pathlib, subprocess, sys

base_path = pathlib.Path(sys.argv[1])
tmpdir = pathlib.Path(sys.argv[2])
root = pathlib.Path(sys.argv[3])
src = base_path.read_text()

def write_mut(name, text):
    p = tmpdir / f"mut-{name}.yml"
    p.write_text(text)
    return p

mutations = [
    ("wildcard-host", src.replace('"127.0.0.1::8000"', '"0.0.0.0::8000"'), "publish-loopback"),
    ("fixed-host-port", src.replace('"127.0.0.1::8000"', '"127.0.0.1:8000:8000"'), "publish-ephemeral"),
    ("missing-enable-label", src.replace('\n      stacklane.enable: "true"\n', "\n", 1), "label-enable"),
]

failures = 0
for name, text, expect in mutations:
    mut_path = write_mut(name, text)
    out = tmpdir / f"mut-{name}.json"
    errf = tmpdir / f"mut-{name}.err"
    env = os.environ.copy()
    env.pop("COMPOSE_FILE", None)
    env.pop("COMPOSE_PROFILES", None)
    env["STACKLANE_INSTANCE"] = "mutprobe"
    env["STACKLANE_BASE_DOMAIN"] = env.get("STACKLANE_BASE_DOMAIN", "test")
    env["COMPOSE_PROJECT_NAME"] = "sermon-translate-mutprobe"
    try:
        proc = subprocess.run(
            [
                "docker", "compose", "-p", "sermon-translate-mutprobe",
                "--project-directory", str(root),
                "-f", str(mut_path),
                "config", "--format", "json",
            ],
            check=False,
            env=env,
            stdout=out.open("w"),
            stderr=errf.open("w"),
            timeout=60,
        )
    except Exception:
        print(f"ok: mutation-{name}-config-rejected", file=sys.stderr)
        continue
    if proc.returncode != 0:
        print(f"ok: mutation-{name}-config-rejected", file=sys.stderr)
        continue
    try:
        cfg = json.loads(out.read_text())
    except Exception:
        print(f"ok: mutation-{name}-invalid-json", file=sys.stderr)
        continue
    services = cfg.get("services") or {}
    bad = False
    if name == "wildcard-host":
        for sc in services.values():
            for p in sc.get("ports") or []:
                if isinstance(p, dict) and p.get("host_ip") != "127.0.0.1":
                    bad = True
    elif name == "fixed-host-port":
        for sc in services.values():
            for p in sc.get("ports") or []:
                if isinstance(p, dict) and p.get("published") not in (None, "", 0, "0"):
                    bad = True
    elif name == "missing-enable-label":
        for sc in services.values():
            labels = sc.get("labels") or {}
            if isinstance(labels, list):
                kv = {}
                for item in labels:
                    if isinstance(item, str) and "=" in item:
                        k, v = item.split("=", 1)
                        kv[k] = v
                labels = kv
            if str(labels.get("stacklane.enable", "")) not in ("true", "1"):
                bad = True
                break
    if not bad:
        print(f"FAIL: mutation-{name}", file=sys.stderr)
        failures += 1
    else:
        print(f"ok: mutation-{name}", file=sys.stderr)

if failures:
    sys.exit(2)
sys.exit(0)
PY
}

validate_rendered() {
  local json_path="$1"
  local expect_instance="$2"
  local expect_project="$3"
  local expect_base_domain="$4"
  local expect_hmr_host="$5"
  python3 - "$json_path" "$expect_instance" "$expect_project" "$PROJECT_SLUG" "$expect_base_domain" "$expect_hmr_host" <<'PY'
import json, re, sys

path, expect_instance, expect_project, project_slug, expect_base_domain, expect_hmr_host = sys.argv[1:7]
try:
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
except Exception:
    print("FAIL: compose-config-parse", file=sys.stderr)
    sys.exit(1)

errors = []

def err(rule):
    errors.append(rule)

services = cfg.get("services") or {}
if "api" not in services:
    err("missing-service-api")
if "web" not in services:
    err("missing-service-web")

name = cfg.get("name") or ""
if name and name != expect_project:
    err("compose-project-identity")

volumes_top = cfg.get("volumes") or {}
vol_keys = set(volumes_top.keys())
named_required = (
    "sermon_translate_api_venv",
    "sermon_translate_uv_cache",
    "sermon_translate_web_node_modules",
    "sermon_translate_pnpm_store",
)
for req in named_required:
    if req not in vol_keys and not any(req in k for k in vol_keys):
        err(f"named-volume-{req}")

def labels_map(sc):
    labels = sc.get("labels") or {}
    if isinstance(labels, list):
        out = {}
        for item in labels:
            if isinstance(item, str) and "=" in item:
                k, v = item.split("=", 1)
                out[k] = v
        return out
    if isinstance(labels, dict):
        return {str(k): str(v) for k, v in labels.items()}
    return {}

def volume_entries(sc):
    return sc.get("volumes") or []

def has_bind(sc, target):
    for v in volume_entries(sc):
        if isinstance(v, dict):
            tgt = v.get("target") or v.get("destination") or ""
            typ = (v.get("type") or "").lower()
            if typ == "bind" and tgt.rstrip("/") == target.rstrip("/"):
                return True
        elif isinstance(v, str) and f":{target}" in v:
            return True
    return False

def has_named(sc, name_part):
    for v in volume_entries(sc):
        if isinstance(v, dict):
            src = str(v.get("source") or "")
            if name_part in src:
                return True
        elif isinstance(v, str) and name_part in v:
            return True
    return False

def env_map(sc):
    env = sc.get("environment") or {}
    if isinstance(env, list):
        out = {}
        for e in env:
            if isinstance(e, str):
                if "=" in e:
                    k, v = e.split("=", 1)
                    out[k] = v
                else:
                    out[e] = ""
        return out
    if isinstance(env, dict):
        return {str(k): "" if v is None else str(v) for k, v in env.items()}
    return {}

def check_ports(svc_name, sc, expect_target):
    ports = sc.get("ports") or []
    if not ports:
        err(f"{svc_name}-publish-required")
        return
    for p in ports:
        if not isinstance(p, dict):
            err(f"{svc_name}-publish-object")
            continue
        if p.get("host_ip") != "127.0.0.1":
            err(f"{svc_name}-publish-loopback")
        if p.get("published") not in (None, "", 0, "0"):
            err(f"{svc_name}-publish-ephemeral")
        try:
            if int(p.get("target")) != int(expect_target):
                err(f"{svc_name}-publish-target-port")
        except (TypeError, ValueError):
            err(f"{svc_name}-publish-target-port")

def check_isolation(svc_name, sc):
    if (sc.get("network_mode") or "") == "host":
        err(f"{svc_name}-no-host-network")
    if str(sc.get("pid") or "").strip().lower() == "host":
        err(f"{svc_name}-no-host-pid")
    priv = sc.get("privileged")
    if priv is True or (isinstance(priv, str) and priv.strip().lower() in ("true", "1", "yes", "on")):
        err(f"{svc_name}-no-privileged")

api = services.get("api") or {}
check_ports("api", api, 8000)
check_isolation("api", api)
al = labels_map(api)
if str(al.get("stacklane.enable", "")) not in ("true", "1"):
    err("api-label-enable")
if str(al.get("stacklane.project", "")) != project_slug:
    err("api-label-project")
if str(al.get("stacklane.instance", "")) != expect_instance:
    err("api-label-instance")
if str(al.get("stacklane.endpoint", "")) != "api":
    err("api-label-endpoint")
if str(al.get("stacklane.port", "")) != "8080":
    err("api-label-port")
if str(al.get("stacklane.target_port", "")) != "8000":
    err("api-label-target-port")
if not has_bind(api, "/opt/src"):
    err("api-source-bind-mount")
for nv in ("sermon_translate_api_venv", "sermon_translate_uv_cache"):
    if not has_named(api, nv):
        err(f"api-named-volume-mount-{nv}")
hc = api.get("healthcheck") or {}
test = hc.get("test") or []
test_s = test if isinstance(test, str) else " ".join(str(x) for x in test)
if "/api/stats" not in test_s or "8000" not in test_s:
    err("api-healthcheck")

web = services.get("web") or {}
check_ports("web", web, 5173)
check_isolation("web", web)
wl = labels_map(web)
if str(wl.get("stacklane.enable", "")) not in ("true", "1"):
    err("web-label-enable")
if str(wl.get("stacklane.project", "")) != project_slug:
    err("web-label-project")
if str(wl.get("stacklane.instance", "")) != expect_instance:
    err("web-label-instance")
if str(wl.get("stacklane.endpoint", "")) != "web":
    err("web-label-endpoint")
if str(wl.get("stacklane.port", "")) != "3000":
    err("web-label-port")
if str(wl.get("stacklane.target_port", "")) != "5173":
    err("web-label-target-port")
if not has_bind(web, "/src"):
    err("web-source-bind-mount")
for nv in ("sermon_translate_web_node_modules", "sermon_translate_pnpm_store"):
    if not has_named(web, nv):
        err(f"web-named-volume-mount-{nv}")

wenv = env_map(web)
if wenv.get("VITE_API_PROXY_TARGET") != "http://api:8000":
    err("web-internal-proxy-dns")
if wenv.get("VITE_DEV_HOST") != "0.0.0.0":
    err("web-listen-all-interfaces")
allowed_hosts = {host.strip() for host in wenv.get("DEV_ALLOWED_HOSTS", "").split(",") if host.strip()}
if f".{expect_base_domain}" not in allowed_hosts:
    err("web-allowed-hosts-base-domain")
if not {"localhost", "127.0.0.1"}.issubset(allowed_hosts):
    err("web-allowed-hosts-loopback")
hmr_host = wenv.get("HMR_HOST", "")
if hmr_host != expect_hmr_host:
    err("web-hmr-host")
if hmr_host:
    hmr_client_port = wenv.get("HMR_CLIENT_PORT", "")
    if not hmr_client_port.isdecimal() or not 1 <= int(hmr_client_port) <= 65535:
        err("web-hmr-client-port")
    if wenv.get("HMR_PROTOCOL") not in ("ws", "wss"):
        err("web-hmr-protocol")

blob = json.dumps({"api": env_map(api), "web": wenv, "labels": [al, wl], "name": name})
if re.search(r"\.local\b", blob):
    err("no-local-domain")

whc = web.get("healthcheck") or {}
wtest = whc.get("test") or []
wtest_s = wtest if isinstance(wtest, str) else " ".join(str(x) for x in wtest)
if "5173" not in wtest_s:
    err("web-healthcheck")

deps = web.get("depends_on") or {}
if isinstance(deps, dict):
    api_dep = deps.get("api") or {}
    cond = api_dep.get("condition") if isinstance(api_dep, dict) else None
    if cond and cond != "service_healthy":
        err("web-depends-on-healthy")
elif isinstance(deps, list) and "api" not in deps:
    err("web-depends-on-api")

if errors:
    for e in errors:
        print(f"FAIL: {e}", file=sys.stderr)
    sys.exit(1)
print(f"ok: rendered-contract instance={expect_instance} project={expect_project}", file=sys.stderr)
sys.exit(0)
PY
}

main() {
  require_tools
  local instance project
  instance="$(derive_instance)"
  project="${PROJECT_SLUG}-${instance}"
  export STACKLANE_INSTANCE="$instance"
  export STACKLANE_BASE_DOMAIN="${STACKLANE_BASE_DOMAIN:-test}"

  COMPOSE_CHECK_UMASK="$(umask)"
  umask 077
  COMPOSE_CHECK_TMPDIR="$(mktemp -d "${TMPDIR:-/tmp}/sermon-translate-compose-check.XXXXXX")"
  chmod 700 "$COMPOSE_CHECK_TMPDIR"
  cleanup_check() {
    rm -rf "${COMPOSE_CHECK_TMPDIR:-}"
    umask "${COMPOSE_CHECK_UMASK:-022}"
  }
  trap cleanup_check EXIT INT TERM

  local cfg1 err1 cfg2 err2 hmr_host hmr_host_alt
  if [[ -v HMR_HOST ]]; then
    hmr_host="$HMR_HOST"
  else
    hmr_host="web.${instance}.${PROJECT_SLUG}.${STACKLANE_BASE_DOMAIN}"
  fi
  cfg1="${COMPOSE_CHECK_TMPDIR}/compose-${instance}.json"
  err1="${COMPOSE_CHECK_TMPDIR}/compose-${instance}.err"
  info "rendering compose config for instance=${instance} project=${project}"
  render_config "$project" "$instance" "$cfg1" "$err1"
  validate_rendered "$cfg1" "$instance" "$project" "$STACKLANE_BASE_DOMAIN" "$hmr_host"
  ok "default instance path (${instance})"

  local alt="slc-altcheck"
  local alt_project="${PROJECT_SLUG}-${alt}"
  cfg2="${COMPOSE_CHECK_TMPDIR}/compose-${alt}.json"
  err2="${COMPOSE_CHECK_TMPDIR}/compose-${alt}.err"
  STACKLANE_INSTANCE="$alt" render_config "$alt_project" "$alt" "$cfg2" "$err2"
  if [[ -v HMR_HOST ]]; then
    hmr_host_alt="$HMR_HOST"
  else
    hmr_host_alt="web.${alt}.${PROJECT_SLUG}.${STACKLANE_BASE_DOMAIN}"
  fi
  validate_rendered "$cfg2" "$alt" "$alt_project" "$STACKLANE_BASE_DOMAIN" "$hmr_host_alt"
  if cmp -s "$cfg1" "$cfg2"; then
    die "two instances produced identical rendered configs"
  fi
  ok "override instance path differs (${alt})"

  info "running mutation probes (fail-closed)"
  run_mutation_probes "$COMPOSE_CHECK_TMPDIR"
  ok "mutation probes"
  ok "all compose-dev checks passed"
}

main "$@"
