export SIE_BASE_URL="${SIE_BASE_URL:-http://localhost:8080}"
export SIE_API_KEY="${SIE_API_KEY:-local}"
export CODEX_HOME="${SIE_CODEX_HOME:-$HOME/.codex-sie}"
mkdir -p "$CODEX_HOME"
cat > "$CODEX_HOME/config.toml" <<TOML
model = "${SIE_MODEL:-Qwen/Qwen3.5-4B:long-context}"
model_provider = "sie"
model_context_window = ${SIE_CONTEXT_TOKENS:-32768}
model_auto_compact_token_limit = $(( ${SIE_CONTEXT_TOKENS:-32768} * 3 / 4 ))
web_search = "disabled"

[features]
multi_agent = false
goals = false
plugins = false

[analytics]
enabled = false

[shell_environment_policy]
exclude = ["SIE_API_KEY"]

[model_providers.sie]
name = "SIE"
base_url = "$SIE_BASE_URL/v1"
env_key = "SIE_API_KEY"
wire_api = "responses"
stream_idle_timeout_ms = 1800000
TOML
codex() { npx -y @openai/codex@0.167.0 "$@"; }
