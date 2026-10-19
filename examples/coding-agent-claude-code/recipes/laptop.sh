# Claude Code on this machine's sie-server, through its /v1/messages.
#
#   mise run serve -- -b sglang        # in a checkout of this repository; MLX on Apple Silicon
#   source recipes/laptop.sh && claude
#
# SIE_BASE_URL defaults to http://localhost:8080. A server with authentication off
# accepts any key; with authentication on, set SIE_API_KEY to the server's token.
# SIE_MODEL, SIE_CONTEXT_TOKENS and SIE_OUTPUT_TOKENS default to the 32K-window
# profile of Qwen/Qwen3.5-4B and its 4,096-token output cap.
export SIE_BASE_URL="${SIE_BASE_URL:-http://localhost:8080}"
export SIE_API_KEY="${SIE_API_KEY:-local}"
export ANTHROPIC_BASE_URL="$SIE_BASE_URL"
export ANTHROPIC_AUTH_TOKEN="$SIE_API_KEY"
export ANTHROPIC_MODEL="${SIE_MODEL:-Qwen/Qwen3.5-4B:long-context}"
export ANTHROPIC_DEFAULT_OPUS_MODEL="$ANTHROPIC_MODEL"
export ANTHROPIC_DEFAULT_SONNET_MODEL="$ANTHROPIC_MODEL"
export ANTHROPIC_DEFAULT_HAIKU_MODEL="$ANTHROPIC_MODEL"
export CLAUDE_CODE_SUBAGENT_MODEL="$ANTHROPIC_MODEL"
export CLAUDE_CODE_MAX_CONTEXT_TOKENS="${SIE_CONTEXT_TOKENS:-32768}"
export CLAUDE_CODE_MAX_OUTPUT_TOKENS="${SIE_OUTPUT_TOKENS:-4096}"
export CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS=1
export CLAUDE_CODE_ATTRIBUTION_HEADER=0
export CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1
export API_TIMEOUT_MS=1800000
