# Releases

SIE uses one release-please train with `vX.Y.Z` tags. The last released version
at setup is **0.7.3**. Changes after that release determine the next version;
the setup does not create or republish 0.7.3.

## Versioning

`.release-please-manifest.json` records the current released version, and
`release-please-config.json` fixes the public history boundary at the actual
`v0.7.3` commit `60996d9c30168e0f8e85b680295f147fdee87f61`. The workflow
also verifies that the public `v0.7.3` stable release and tag exist and that the
tag belongs to the release branch. Release-please discovers the matching
release natively; the checked-in bootstrap SHA remains the fail-closed fallback
for that same historical boundary.

The current pre-1.0 policy is retained: ordinary features and fixes advance the
patch version; breaking changes advance the minor version. Public conventional
commits generate `CHANGELOG.md`. No old changelog sections are rewritten.

The release PR updates the coordinated Python/npm package versions, gateway,
sidecar and audio release fields, TypeScript runtime version, and Helm metadata.
It also refreshes the coupled public locks and stamps the new version into the
committed server and gateway OpenAPI documents. CI regenerates those documents
and requires identical bytes, so they are not release-please extra files: its
JSON rewrite changes number formatting and precision, string escapes, and key
order. Config and MCP join this train for their first PyPI publication.
The refresh also formats the SDK package metadata rewritten by release-please
with the lockfile-pinned Biome version and the SDK's formatter configuration.
Only that manifest is formatted; its release version and metadata are preserved.
Independently versioned implementation crates are not silently renumbered: a
Rust worker image follows the release image tag even where its crate has an
independent version.

Release PRs receive the same mandatory CI checks as other PRs. Release-please
and its lock refresh use a repository-scoped GitHub App so their PR updates
trigger normal CI. No PR build is a publication job.

## Published outputs

The Python train contains 13 distributions:

- `sie-sdk`, `sie-server`, `sie-config`, and `sie-mcp`;
- `sie-langchain`, `sie-llamaindex`, `sie-haystack`, `sie-dspy`,
  `sie-crewai`, `sie-chroma`, `sie-lancedb`, `sie-qdrant`, and
  `sie-weaviate`.

The npm train contains five packages under `@superlinked`: `sie-sdk`,
`sie-chroma`, `sie-langchain`, `sie-llamaindex`, and `sie-lancedb`.

The GHCR image names are `sie-server`, `sie-gateway`, `sie-config`, `sie-mcp`,
`sie-server-sidecar`, and `sie-server-rust`, under `ghcr.io/superlinked`.
`.github/release-matrix.json` defines the supported server platform/bundle
combinations. Missing bundle source or build recipes are errors, not an
instruction to skip a release target. The Rust worker retains its explicit
CUDA/architecture image suffix.

The Helm chart is published at
`oci://ghcr.io/superlinked/charts/sie-cluster` after its versioned images verify.

Native audio is distributed as a GitHub Release asset, not a PyPI project:

```text
sie_audio_prep-<version>-cp312-abi3-manylinux_2_28_x86_64.whl
```

A Linux amd64 sidecar executable and checksum are also provided for consumers
that embed the binary rather than run its container:

```text
sie-server-sidecar-v<version>-linux-amd64
sie-server-sidecar-v<version>-linux-amd64.sha256
```

The executable is extracted from the already-verified sidecar image, not rebuilt.
Its attached provenance records the source revision, architecture, and ABI;
the compatibility check uses Debian 12. Rust library dependencies
can still be consumed using Cargo's Git support; no crates.io publication is
implied by an image or binary release.

## Build, verify, publish

The top-level `release.yml` has two automatic entrypoints:

- A push to `main` runs the App-authored release-please and release-PR lock
  refresh steps only when `PUBLIC_RELEASE_AUTOMATION_ENABLED` is exactly
  `true`. An absent variable leaves authoring cleanly skipped.
- The App-created stable `release: published` event runs preparation, builds,
  verification, and direct publisher fanout at the tagged commit.

This separates the commit that triggers release-please from the commit it
releases. A later `main` push cannot cause packages from an earlier release to
be published with the later run's provenance. The App credential is required
because its events trigger workflows; a release created with `GITHUB_TOKEN`
does not provide that handoff. Publication does not depend on a tag-push event.

Release work is serialized with GitHub's `queue: max` concurrency mode and no
in-progress cancellation, so newer events do not replace pending releases. The
queue has GitHub's documented bound of 100 pending runs. The current actionlint
schema does not recognize `queue`; CI exempts only that diagnostic and separately
checks the required queue setting. See [workflow
concurrency](https://docs.github.com/en/actions/how-tos/write-workflows/choose-when-workflows-run/control-workflow-concurrency).

PR and candidate builds produce archives without publishing. They use the
actual package versions in that source tree. Release builds additionally
require the complete package set to match the release version.

The `Release candidate` workflow rehearses all six artifact families without
registry credentials or release-writing permissions. It runs on release
pipeline changes and release version updates, and can be dispatched manually
against a reviewed branch. It builds the full image matrix and extracts the
native sidecar from its tested image, as well as building Python, npm, audio,
and the packaged Helm chart. Require its final `Release candidate / Complete`
check on the final release PR head before merging that PR.

On a stable release, every builder must pass the `artifacts-ready` gate before
any publisher can start. Publishers consume the retained archives. This keeps
a failed image or chart build from leaving packages partially published.
Registry failures can still interrupt publication; recovery resumes the
original failed publisher with its same version, source and retained bytes.

Build outputs are tested before upload. Publisher jobs consume those same
archives or images; they do not independently rebuild them. Before a release
upload, the run commit, release output commit, checked-out source, and stable
tag must identify the same revision. Versioned outputs are immutable: an
existing matching upload may be accepted, but different bytes at the same
version are a failure.

The Candle CUDA image is checked on driverless runners using its source-bound
image configuration, extracted ELF executable, and image-local shared-library
resolution. Only the host-provided `libcuda.so.1` may be unresolved; all other
missing libraries and loader diagnostics fail validation. The same check runs
in Rust CUDA image PR CI. These checks do not exercise GPU inference. CPU Rust
and the other service images retain their executable `--help` smoke checks.

Package Helm charts with `mise run helm -- package --destination DIR`. This
stages the checked-in model and bundle catalogs, checks their exact bytes in
the archive, and lints and renders that archive after staging is removed,
including the embedded-config mode. CI and publication use this same path.

Floating image aliases move only after the full versioned image set verifies.
An older release recovery keeps those aliases unchanged when a newer stable
release exists; it repairs only the requested versioned outputs.
The dependent chart waits for that verification. A GitHub Release can exist
while a publisher fails; the release-completion check, not the release page
alone, indicates that all expected outputs are available.

Ordinary PR checks include lint, typechecking, unit tests, public integration,
packed-distribution checks and CPU/container verification. Full release image
builds include the declared CUDA variants. Building a CUDA image is not a claim
that GPU inference was exercised.

## Terraform module follow-up

Every stable SIE release needs a follow-up in all four public Terraform module
repositories. The modules provision infrastructure; their installation examples
select the SIE Helm chart, cloud values files, and runtime images. SIE's release
workflow does not update or release these repositories automatically.

| Cloud | Repository |
| --- | --- |
| AWS | [terraform-aws-sie](https://github.com/superlinked/terraform-aws-sie) |
| Google Cloud | [terraform-google-sie](https://github.com/superlinked/terraform-google-sie) |
| Azure | [terraform-azure-sie](https://github.com/superlinked/terraform-azure-sie) |
| Alibaba Cloud | [terraform-alicloud-sie](https://github.com/superlinked/terraform-alicloud-sie) |

The agent or maintainer carrying the SIE release owns this follow-up:

1. Verify that the stable tag, release source, final publication job, versioned
   images, and public Helm chart agree. Use the successfully published version,
   not an open release PR or a tag whose publication is incomplete.
2. Inspect current `main`, open update/release PRs, and the module's contributor
   and release instructions in each repository. Work in isolated worktrees and
   update an existing matching PR when possible.
3. Update every SIE chart pin, matching tagged cloud-overlay URL, chart reference,
   and image-mirroring example. Preserve the image's platform, bundle, and Rust
   architecture suffixes. Keep the Terraform module's own version, provider
   constraints, and infrastructure sizing independent of the SIE version.
   Retain applicable upgrade guidance and check for new breaking changes.
4. Run the module's existing relevant checks and render the actual published
   chart with its documented cloud values and Terraform output formats. Verify
   that the rendered image tags exist, mirrored tags cover enabled workers,
   and workload identity, payload-store wiring, and GPU selectors still match
   the example. This validation does not require applying infrastructure.
5. Open focused PRs, follow CI and review findings on their final heads, and
   merge within the authorized scope through normal repository protections.
   Do not bypass a missing review or attempt to approve a PR as its author.
6. After the updates merge, inspect the module's release-please PR and confirm it
   includes the new SIE references. For generated manifest/changelog-only release
   PRs, review that metadata directly; do not request or wait for a separate
   CodeRabbit review when bot auto-review skips. Reinspect every changed head,
   and take unexpected non-generated changes through the normal code-review flow.
   Inspect exact-head workflow runs: `action_required` CI needs workflow-execution
   approval. Approve execution where authorized, or report that gate. Require
   actual Terraform CI completion and normal repository approvals, unless the
   user explicitly authorizes bypassing those approvals. A skipped bot review
   does not supply a required approval. Module release versions are independent
   of SIE's version. Verify the resulting immutable tag and Terraform Registry
   version/download source; a merged update PR alone does not publish a module
   version.

Report SIE artifact publication and Terraform follow-up separately. For each
module, include the update PR/merge SHA and published module version, or the
exact outstanding release PR, review, CI, or publication gate. If module
publication is outside the current authorization, report it as pending. Do not
leave this follow-up implicit or describe pending module releases as published.

## Publisher setup

Finalize these identities before enabling uploads:

| Publisher | Repository | Workflow identity | Environment | Authority |
| --- | --- | --- | --- | --- |
| PyPI distributions | `superlinked/sie` | `release.yml` | `pypi` | Trusted Publishing, upload-job `id-token: write` |
| npm packages | `superlinked/sie` | `release.yml` | `npm` | Trusted Publishing, upload-job `id-token: write` |
| Images | `superlinked/sie` | Release image workflow | `ghcr` | `GITHUB_TOKEN`, `packages: write` |
| Helm chart | `superlinked/sie` | Release chart workflow | `helm` | `GITHUB_TOKEN`, `packages: write` |
| Release assets | `superlinked/sie` | Native asset workflows | `github-release` | `GITHUB_TOKEN`, `contents: write` |

PyPI/npm upload jobs live in the top-level workflow so the configured OIDC
identity is unambiguous. Existing package names use their existing registry
settings; only new PyPI projects need pending publishers. PyPI allows only one
pending project per publisher identity, so bootstrap config and MCP sequentially
with the same repository, `release.yml` workflow, and `pypi` environment. Register
`sie-config` first. Its next OIDC token exchange creates the project and converts
the pending publisher into an ordinary publisher. Then register `sie-mcp` and
request another original-run publisher retry so a new token exchange creates it.
One upload attempt mints its token only once; adding the second pending publisher
does not extend a token already issued to the first attempt. Recovery verifies
and skips byte-identical files already uploaded and consumes the same retained
archives. Audio assets do not need another PyPI registration.

npm supports one trusted publisher per package. Its actual upload job uses a
GitHub-hosted runner and a pinned supported npm version. Ordinary builds and
tests use Blacksmith. Do not carry a long-lived npm token into the OIDC upload
job. See [npm trusted publishing](https://docs.npmjs.com/trusted-publishers/).

For GHCR, explicitly grant this repository's Actions jobs access to each
existing package and confirm public visibility. A source label or successful
anonymous pull does not prove write permission. GHCR uses the repository token,
not an external OIDC registration.

Protect `main` and the `v*` tag namespace before enabling publication. Restrict
stable tag creation to the release App, and forbid tag updates/deletions. A tag
pattern allowed by an environment is not a substitute for those repository
rules. Keep creation bypass and tag immutability in separate rulesets so the
App's permission to create does not also permit replacing or deleting tags.
The App does not need a bypass of main-branch review or CI requirements.
Preparation and every publisher verify the protected tag, exact source,
and ancestry in protected `main`.

The App and manual recovery use the main-only `release-automation` environment.
The `pypi`, `npm`, `ghcr`, `helm`, and `github-release` environments allow only
the protected release tags, not pull-request branches. `release-automation` supplies
`PUBLIC_RELEASE_APP_ID` and `PUBLIC_RELEASE_APP_PRIVATE_KEY`. Install that App
only on this repository with Contents and Pull requests read/write permission.
These App credentials are not distribution-registry credentials.

Keep the repository variable `PUBLIC_RELEASE_AUTOMATION_ENABLED` absent while
setting up the App and `release-automation` environment. After the App is
installed, its ID/private key are configured, protected `main` and stable-tag
rules are in force, and the release workflow has passed review, set the variable
to exactly `true` to enable release-PR authoring. This authoring gate is separate
from artifact publication and must not be enabled as a substitute for publisher
configuration.

Actual publication additionally requires
`PUBLIC_RELEASE_PUBLISHING_ENABLED=true`. Keep it absent until the release
baseline, required checks, publisher identities, package access and build
capacity have been checked. Build-only paths need neither this setting nor
registry credentials. Do not enable two competing publishers for the same
artifact destination.

## Recovery

### Release-PR authoring

To refresh an open release PR, use the push-to-`main` authoring workflow. Merge
any automation repair through normal review and CI, then approve that commit's
`release-automation` job. The manual entrypoint below only recovers failed
publication; it does not refresh release PRs.

Before resuming a waiting authoring run, inspect its source revision and workflow:
release-please can read current `main`, but the run still executes the workflow
from its original push. An old run can therefore generate current release files
using outdated refresh steps. Cancel obsolete queued **push** runs before
cancelling the old waiting run, then retain the authoring run from the intended
current source. Do not cancel publication or recovery runs as part of this
cleanup. Keep the environment protections and approve the retained run normally.
If the intended push run was cancelled, rerun that exact push run after checking
its source and workflow, while GitHub still permits the rerun; do not replay the
obsolete backlog. Verify the generated PR's source ancestry after refresh, too:
the approved run's revision does not freeze release-please's reads of `main`.

### Artifact publication

Retain release build artifacts for at least 30 days. Normal recovery reruns
failed publication jobs from the original release workflow run. This preserves
the original commit/ref and reuses the original outputs.

The manual recovery entrypoint validates an existing version and its original
run before requesting those reruns. It is not an alternate uploader from newer
source. npm's automatic provenance uses the run's commit; checking out an older
tag inside a newer run would not change that identity.

In **Actions → Release → Run workflow**, select `main` and provide `version`
(without `v`), the numeric `original_run` ID, and an optional `family` selection.
The tag is derived from the version. Select the original release-event
publication run, not the main-push authoring run. Its `prepare` job must have
succeeded, a selected publisher family must have failed, and its original
archives must still exist. A skipped-only family or a failure only in the
completion check is not treated as a successful retry request. Diagnose those
conditions explicitly instead of rerunning release-please and losing its
original release outputs. Failed builds can be rerun from the original run
before retrying publication.

If the original run or artifacts are unavailable, or an existing version has
conflicting bytes, stop and resolve that release explicitly. Do not move a tag,
overwrite a package, or attach a newly rebuilt conflicting native asset.

When filling a missing older npm version after a newer version is already
`latest`, recovery uses the version-scoped `release-vX.Y.Z` dist-tag. It does not
move `latest` backwards. Normal new releases use `latest`; identical existing
versions are verified and left untouched.

## Candidate compatibility checks

An unmerged public commit can be built without creating a stable release.
Consumers may select its exact archives/image digests for an isolated test run
while their stable dependency pins stay unchanged. Candidate artifacts are
code: approve the exact source before executing them in an environment with
private access. Candidate builds do not receive production or publishing
credentials.

Local build, pack, and smoke checks cannot prove account-side OIDC trust,
registry permissions, or hosted-runner capacity. Those remain checks for the
first authorized publication from the final release source.
