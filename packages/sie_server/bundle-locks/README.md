# Saved bundle dependency closures

Each target directory fixes the complete runtime distribution set installed in
`/app/bundle-libs`: exact versions, public artifact URLs and SHA256 hashes, or an
exact Git commit. It includes dependencies absent from the root `uv.lock`.
Image builds consume `pylock.toml` with the pinned uv 0.9.28 installer; they do
not resolve the overlay again from ranged bundle declarations.

The target set is the 12 Linux amd64 release-matrix combinations plus separate
CPU arm64 locks for `default`, `ctranslate2`, `sglang`, and `transformers5`.
Python 3.12 and the target architecture are part of each manifest. An image
combination without a saved lock fails before installation.

## Boundary and receipts

This contract covers the runtime distribution inventory of `bundle-libs`.
It does **not** make the whole image reproducible. The base `.venv`, operating
system packages, native audio-prep build, Python image, and build dependencies
used for an sdist/VCS package remain separate inputs. Repeated source builds
can differ even when the source commit and runtime inventory are fixed.

CUDA 13 still overlays cu130 Torch on a base environment containing cu12 Torch.
A saved inventory or CPU import check does not establish GPU namespace
isolation; a missing cu130 `nvidia` submodule may fall through to the base.
Live GPU qualification remains separate from lock validation.

Before installing, the verifier binds the lock to the root lock, resolved
bundle declarations, target, uv version, exact exception manifest, and the
compiler/verifier/declaration-resolver source hashes. After installation it
checks the actual distribution set, versions, dependency metadata, requested
extras, and recorded direct-source identities without importing package code.
Missing, additional, duplicate, incompatible, or substituted distributions fail
the build. The runtime image retains both `/app/bundle-libs-inventory.json` and
the existing `/app/bundle-libs-freeze.txt`.

The installer verifies artifact hashes. Installed PEP610 metadata is a separate
check: uv 0.9.28 records a direct archive URL with an empty `archive_info` hash
table. The verifier checks the URL and any recorded hash; it does not claim
that installed metadata independently proves the downloaded bytes. VCS URLs
and resolved commits must match exactly.

## Root pins and deliberate differences

All compatible root versions constrain compilation. `exceptions.json` records
every target-specific version or source difference with an exact version,
artifact-set fingerprint, and reason. An unexplained new difference or stale
exception fails validation. Packages missing from the root lock still receive
full version/source locks here.

The existing index routes are preserved. In particular, the CUDA 12 PyPI
Torch 2.9.1 wheel requires CUDA 12.8 NVIDIA packages while the root Linux Torch
build uses cu129. Their exact differences are explicit. CPU/cu130 wheel variants
and engine-required Transformers, Torch, and related versions are also explicit.
Same-byte public mirrors and a root entry without artifact records are labeled
source differences; they are not reported as dependency incompatibilities.

The CUDA 13 SGLang index omits upload timestamps for its native cu130 wheels.
The global cutoff therefore excludes those index candidates in uv 0.9.28.
The `source_pins` records preserve the established cu130 variants with exact
public wheel URLs, versions, SHA256 hashes, and reasons. These are explicit
exceptions for the named artifacts, not a disabled cutoff: all index-resolved
candidates still obey the compilation cutoff. The verifier rejects a missing,
changed, or substituted source pin, and binds its digest into the build receipt.

## Validation and updates

From a bootstrapped clean checkout:

```console
# Validate every saved lock and dry-run the Docker-pinned installer.
mise exec -- uv run --frozen --no-sync python tools/ci/bundle_lock.py check
mise install uv@0.9.28
SIE_BUNDLE_LOCK_UV="$(mise exec --no-deps uv@0.9.28 -- which uv)"
export SIE_BUNDLE_LOCK_UV
SIE_RUN_IMAGE_RESOLUTION=1 mise exec -- uv run --frozen --no-sync pytest -q packages/sie_server/tests/core/test_bundle_image_resolution.py
```

The required Bundle resolution CI job runs these checks for all 16 targets.
Its dry runs check the actual saved-lock reader without installing model
packages. Image builds perform the additional installed-inventory check.
The explicit installer path prevents `uv run` from selecting the separate uv
version installed in the project environment; its version is still checked.

For an intentional update, first review any new root/bundle conflicts and edit
the exact exception records. Compile with an explicit cutoff using only public
indexes, then review the complete lock and manifest diff:

```console
# Recompile one target with an explicit dependency cutoff.
mise exec -- uv run --frozen --no-sync python tools/ci/bundle_lock.py compile --uv "$SIE_BUNDLE_LOCK_UV" --target cpu-default-amd64 --exclude-newer 2026-10-18T00:00:00Z
```

The compiler rejects a different uv version and strips ambient pip/uv index,
override, and hash-bypass settings. Some public mirror entries omit hashes even
with `--generate-hashes`; during explicit compilation the helper downloads at
most 64 MiB per missing artifact and records its SHA256. A missing/oversized
artifact fails instead of saving an unhashed lock. Changing compiler policy
requires reviewing regenerated manifests, even if package versions stay equal.
