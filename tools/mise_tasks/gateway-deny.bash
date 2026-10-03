#!/usr/bin/env bash
#MISE description="Audit workspace Rust dependencies with cargo-deny"
#USAGE arg "[checks]" help="Comma-separated subset of checks (advisories,licenses,bans,sources)"

set -euo pipefail

CHECKS="${usage_checks-all}"
if [[ -z "$CHECKS" || "$CHECKS" == ,* || "$CHECKS" == *, || "$CHECKS" == *,,* ]]; then
    echo "ERROR: checks must be a non-empty comma-separated list." >&2
    exit 2
fi
IFS=',' read -r -a CHECK_ARGS <<< "$CHECKS"
SEEN_CHECKS=" "
for check in "${CHECK_ARGS[@]}"; do
    case "$check" in
        advisories|licenses|bans|sources|all) ;;
        *)
            echo "ERROR: unknown cargo-deny check: $check" >&2
            exit 2
            ;;
    esac
    if [[ "$SEEN_CHECKS" == *" $check "* ]]; then
        echo "ERROR: duplicate cargo-deny check: $check" >&2
        exit 2
    fi
    SEEN_CHECKS+="$check "
done
if (( ${#CHECK_ARGS[@]} > 1 )) && [[ "$SEEN_CHECKS" == *" all "* ]]; then
    echo "ERROR: cargo-deny check 'all' cannot be combined with individual checks." >&2
    exit 2
fi

echo "## Running cargo-deny check (${CHECKS}) for the Rust workspace"
# One sweep over the public Rust workspace using the root policy and lockfile.
# --config is set explicitly: cargo-deny would auto-discover deny.toml from
# the manifest dir, but pinning it makes the contract obvious + works
# regardless of cwd if the script is invoked from elsewhere.
mise exec cargo:cargo-deny -- cargo-deny \
    --workspace \
    --all-features \
    --config deny.toml \
    check \
    "${CHECK_ARGS[@]}"
