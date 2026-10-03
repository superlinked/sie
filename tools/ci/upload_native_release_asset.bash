#!/usr/bin/env bash
set -euo pipefail

: "${RELEASE_ASSET_FILENAME:?RELEASE_ASSET_FILENAME is required}"
: "${GITHUB_REPOSITORY:?GITHUB_REPOSITORY is required}"
: "${RELEASE_TAG:?RELEASE_TAG is required}"
: "${GH_TOKEN:?GH_TOKEN is required}"

wheel="${1:?wheel path is required}"
if [[ ! -f "$wheel" || -L "$wheel" || "$(basename "$wheel")" != "$RELEASE_ASSET_FILENAME" ]]; then
  echo "invalid native release asset path: $wheel" >&2
  exit 2
fi

local_size="$(wc -c < "$wheel" | tr -d ' ')"
local_sha="$(sha256sum "$wheel" | cut -d' ' -f1)"
expected_url="https://github.com/$GITHUB_REPOSITORY/releases/download/$RELEASE_TAG/$RELEASE_ASSET_FILENAME"

release_json="$(gh api "repos/$GITHUB_REPOSITORY/releases/tags/$RELEASE_TAG")"
asset_count="$(jq --arg name "$RELEASE_ASSET_FILENAME" '[.assets[] | select(.name == $name)] | length' <<<"$release_json")"
if [[ "$asset_count" -gt 1 ]]; then
  echo "release contains duplicate native asset names: $RELEASE_ASSET_FILENAME" >&2
  exit 1
fi

verify_asset_bytes() {
  local json="$1"
  local asset_id remote_size remote_digest remote_sha downloaded
  asset_id="$(jq -r --arg name "$RELEASE_ASSET_FILENAME" '.assets[] | select(.name == $name) | .id' <<<"$json")"
  remote_size="$(jq -r --arg name "$RELEASE_ASSET_FILENAME" '.assets[] | select(.name == $name) | .size' <<<"$json")"
  remote_digest="$(jq -r --arg name "$RELEASE_ASSET_FILENAME" '.assets[] | select(.name == $name) | .digest // empty' <<<"$json")"
  test "$remote_size" = "$local_size"
  if [[ -n "$remote_digest" ]]; then
    test "$remote_digest" = "sha256:$local_sha"
    return
  fi
  downloaded="$(mktemp)"
  gh api -H 'Accept: application/octet-stream' \
    "repos/$GITHUB_REPOSITORY/releases/assets/$asset_id" > "$downloaded"
  remote_sha="$(sha256sum "$downloaded" | cut -d' ' -f1)"
  rm -f "$downloaded"
  test "$remote_sha" = "$local_sha"
}

if [[ "$asset_count" -eq 1 ]]; then
  verify_asset_bytes "$release_json"
  echo "identical native release asset already exists: $RELEASE_ASSET_FILENAME"
else
  gh release upload --repo "$GITHUB_REPOSITORY" "$RELEASE_TAG" "$wheel"
fi

verified_json=""
for _ in 1 2 3 4 5 6; do
  verified_json="$(gh api "repos/$GITHUB_REPOSITORY/releases/tags/$RELEASE_TAG")"
  verified_count="$(jq --arg name "$RELEASE_ASSET_FILENAME" '[.assets[] | select(.name == $name)] | length' <<<"$verified_json")"
  if [[ "$verified_count" -eq 1 ]]; then
    break
  fi
  sleep 2
done
test "$verified_count" -eq 1
actual_url="$(jq -r --arg name "$RELEASE_ASSET_FILENAME" '.assets[] | select(.name == $name) | .browser_download_url' <<<"$verified_json")"
test "$actual_url" = "$expected_url"
verify_asset_bytes "$verified_json"
