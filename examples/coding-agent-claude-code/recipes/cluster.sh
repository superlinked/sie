# Claude Code on a self-hosted SIE cluster, through its gateway's /v1/messages.
#
#   export SIE_BASE_URL=https://your-gateway   # without /v1
#   export SIE_API_KEY=...
#   export SIE_MODEL=zai-org/GLM-5.3-Flash      # a tools-capable generation model, or model:profile
#   export SIE_CONTEXT_TOKENS=...               # the window /v1/models declares for it
#   export SIE_OUTPUT_TOKENS=...                # at most the profile's max_output_tokens
#   source recipes/cluster.sh && claude
export ANTHROPIC_BASE_URL="$SIE_BASE_URL"
export ANTHROPIC_AUTH_TOKEN="$SIE_API_KEY"
export ANTHROPIC_MODEL="$SIE_MODEL"
export ANTHROPIC_DEFAULT_OPUS_MODEL="$ANTHROPIC_MODEL"
export ANTHROPIC_DEFAULT_SONNET_MODEL="$ANTHROPIC_MODEL"
export ANTHROPIC_DEFAULT_HAIKU_MODEL="$ANTHROPIC_MODEL"
export CLAUDE_CODE_SUBAGENT_MODEL="$ANTHROPIC_MODEL"
export CLAUDE_CODE_MAX_CONTEXT_TOKENS="$SIE_CONTEXT_TOKENS"
export CLAUDE_CODE_MAX_OUTPUT_TOKENS="$SIE_OUTPUT_TOKENS"
export CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS=1
export CLAUDE_CODE_ATTRIBUTION_HEADER=0
export CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1
export API_TIMEOUT_MS=1800000
