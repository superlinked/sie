# Public release contract

SIE uses one release-please train and one `vX.Y.Z` identity. Public commits on
`main` produce a release pull request and update the public `CHANGELOG.md`.
Merging that pull request creates the tag and GitHub Release, then the same
workflow calls the Python, npm, Docker, and Helm reusable workflows directly
with the exact tag, version, commit SHA, and publication intent. A tag event is
not used as an indirect trigger.

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

The image train contains the declared server platform/bundle matrix plus
`sie-gateway`, `sie-config`, `sie-mcp`, `sie-server-sidecar`, and the chart's
L4 Candle worker. The chart is packaged only after every versioned image has
been built and verified.

## Fail-closed publication

Build, test, pack, artifact upload, and source-closure verification are safe to
run before activation. Every registry write additionally requires both the
reusable workflow's boolean `publish` input and repository variable
`PUBLIC_RELEASE_PUBLISHING_ENABLED` to equal `true`. Only the isolated publish
jobs receive OIDC or package-write permission. Python and npm use trusted
publishing; no long-lived registry token is part of the contract.

To activate publication after this setup is merged:

1. Disable every previous writer and independently prove that it can no longer
   write tags, releases, packages, images, or charts.
2. Create and protect the `pypi` and `npm` GitHub Environments.
3. Configure each registry trusted publisher for this exact repository,
   reusable workflow filename, and matching environment.
4. Run and inspect a no-write build/pack release validation.
5. Set `PUBLIC_RELEASE_PUBLISHING_ENABLED=true` only after those checks pass.

Repository settings, environments, trusted publishers, and the latch are not
configured by this source change.

## Partial failures and reruns

A GitHub Release can exist while one publisher is red. That state is an
incomplete release: do not create another tag and do not edit or move the
existing tag. Rerun only the failed jobs from the original top-level workflow
run so they retain the same immutable tag and SHA. Docker `latest` aliases do
not move until the complete versioned image set verifies; Helm publication
waits for that Docker verification. If source, version, or tag validation no
longer matches, stop instead of repairing the release from a different commit.
