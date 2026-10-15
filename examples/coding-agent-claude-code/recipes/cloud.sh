# Claude Code on SIE Cloud, through /v1/messages, with GLM-5.3-Flash via SIE Relay.
#
#   export SIE_BASE_URL=https://...   # your SIE Cloud endpoint, without /v1
#   export SIE_API_KEY=sk-sie-...
#   source recipes/cloud.sh && claude
#
# The window matches what the :relay profile declares in /v1/models.
export ANTHROPIC_BASE_URL="$SIE_BASE_URL"
export ANTHROPIC_AUTH_TOKEN="$SIE_API_KEY"
export ANTHROPIC_MODEL="zai-org/GLM-5.3-Flash:relay"
export ANTHROPIC_DEFAULT_OPUS_MODEL="$ANTHROPIC_MODEL"
export ANTHROPIC_DEFAULT_SONNET_MODEL="$ANTHROPIC_MODEL"
export ANTHROPIC_DEFAULT_HAIKU_MODEL="$ANTHROPIC_MODEL"
export CLAUDE_CODE_SUBAGENT_MODEL="$ANTHROPIC_MODEL"
export CLAUDE_CODE_MAX_CONTEXT_TOKENS=131072
export CLAUDE_CODE_MAX_OUTPUT_TOKENS=32768
export CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS=1
export CLAUDE_CODE_ATTRIBUTION_HEADER=0
export CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1
export API_TIMEOUT_MS=1800000
