# Public release contract

SIE uses one release-please train and one `vX.Y.Z` identity. Public commits on
`main` produce a release pull request and update the public `CHANGELOG.md`.
Merging that pull request creates the tag and GitHub Release, then the same
workflow calls the Python, npm, Docker, and Helm reusable workflows directly
with the exact tag, version, commit SHA, and publication intent. It also builds
the native Linux audio wheel and attaches it to that exact GitHub Release. A
tag event is not used as an indirect trigger.

The release pull request updates all supported package manifests, public Rust
service versions, native-audio release markers, the TypeScript runtime version,
and Helm `version`/`appVersion`. The release job refreshes the coupled root
locks on that pull-request branch. Publication jobs never rewrite source
manifests or locks.

## Supported artifacts

The PyPI train contains exactly: `sie-sdk`, `sie-server`, `sie-langchain`,
`sie-llamaindex`, `sie-haystack`, `sie-dspy`, `sie-crewai`, `sie-chroma`,
`sie-lancedb`, `sie-qdrant`, and `sie-weaviate`.

The npm train contains exactly: `@superlinked/sie-sdk`,
`@superlinked/sie-chroma`, `@superlinked/sie-langchain`,
`@superlinked/sie-llamaindex`, and `@superlinked/sie-lancedb`.

`sie-audio-prep` is not published to PyPI. Each release instead carries the
exact Linux asset
`sie_audio_prep-<version>-cp312-abi3-manylinux_2_28_x86_64.whl` at
`https://github.com/superlinked/sie/releases/download/v<version>/<filename>`.
The build checks out the release SHA, uses the pinned manylinux container,
Maturin, and Zig toolchain, and validates the wheel tag, metadata, and native
extension before attachment.

The image train contains the declared server platform/bundle matrix plus
`sie-gateway`, `sie-config`, `sie-mcp`, `sie-server-sidecar`, and the chart's
L4 Candle worker. The chart is packaged only after every versioned image has
been built and verified.

`mise run docker -- matrix` prints the authoritative server matrix. The
`build-server` and `build-service` subcommands require an exact 40-character
source revision; `verify` checks the complete versioned set, and `alias`
repeats that verification before moving any `latest` reference.

## Fail-closed publication

Build, test, pack, artifact upload, and source-closure verification are safe to
run before activation. Every registry write additionally requires both the
reusable workflow's boolean `publish` input and repository variable
`PUBLIC_RELEASE_PUBLISHING_ENABLED` to equal `true`. Only the isolated publish
jobs receive OIDC or package-write permission. Python and npm use trusted
publishing; no long-lived registry token is part of the contract.

Every reusable write job independently requires a `push` on
`refs/heads/main` in `superlinked/sie`, requires `github.sha` to equal the
release input SHA, and fetches the release tag to prove that it resolves to the
same commit. A same-repository pull request or manually invoked caller cannot
turn on publication even after the repository latch is enabled. The writers
are additionally isolated behind protected `pypi`, `npm`, `ghcr`, `helm`, and
`github-release` environments.

Release-please and release-PR lock refreshes use a dedicated GitHub App token,
not `GITHUB_TOKEN`. The protected `release-automation` environment supplies
`PUBLIC_RELEASE_APP_ID` and `PUBLIC_RELEASE_APP_PRIVATE_KEY`; the App must be
installed only on `superlinked/sie` with repository Contents and Pull requests
read/write permissions. Missing credentials fail the release workflow before
any mutation. App-authored PR creation and lock pushes emit normal
`pull_request` / `synchronize` events, and the workflow verifies the remote PR
head after the final push. Branch protection must require `CI / Required` on
that exact head before the release PR can merge.

To activate publication after this setup is merged:

1. Disable every previous writer and independently prove that it can no longer
   write tags, releases, packages, images, or charts.
2. Create and protect the `release-automation`, `pypi`, `npm`, `ghcr`, `helm`,
   and `github-release` GitHub Environments. Restrict them to `main`; put the
   App ID/private key only in `release-automation`.
3. Install the narrowly permissioned public-release GitHub App and configure
   its environment variable/secret. Confirm a release PR creation and a lock
   refresh each trigger CI on the exact resulting head.
4. Configure each registry trusted publisher for this exact repository,
   reusable workflow filename, and matching environment.
5. With the publication latch still absent, run every Docker matrix cell on
   the declared public `ubuntu-24.04` runners, including CUDA 12, CUDA 13, and
   SM89 Candle. Record disk, timeout, and build success evidence. Local testing
   proves only the CPU Candle image; public CUDA runner capacity is not yet
   proven. If capacity is insufficient, select a publicly governed runner in a
   separate reviewed change.
6. Run and inspect all other no-write build/pack paths, including the exact
   native audio asset.
7. Set `PUBLIC_RELEASE_PUBLISHING_ENABLED=true` only after those checks pass.

Repository settings, environments, trusted publishers, and the latch are not
configured by this source change.

## Partial failures and reruns

A GitHub Release can exist while one publisher is red. That state is an
incomplete release: do not create another tag and do not edit or move the
existing tag. Rerun only the failed jobs from the original top-level workflow
run so they retain the same immutable tag and SHA. Docker `latest` aliases do
not move until the complete versioned image set verifies; Helm publication
waits for that Docker verification. The native audio wheel is part of release
completeness and must exist under the exact filename above. If source, version,
tag, PR-head, or asset validation no longer matches, stop instead of repairing
the release from a different commit.
