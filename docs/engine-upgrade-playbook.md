# Engine upgrade playbook

Use this playbook for any change that moves an inference engine or its base
image: an SGLang, TensorRT-LLM, CTranslate2 or MLX version, a pinned torch,
transformers or xgrammar version, or a CUDA base image. Copy the checklist at
the end into the pull request description and fill it in.

An engine bump can break profiles without breaking any build. SGLang 0.5.20
renamed three launch flags that profiles and adapters used (public #320), and
14 of the 45 profiles on that lane failed to start while every check passed.
The steps below exist so that a renamed flag, a changed default or a raised
driver floor is found before merge, and is visible to users when it ships.

## 1. Read the upstream notes

Before changing a pin, read the engine's release notes for every version you
skip. Look for:

- removed, renamed or split launch flags, and flags that became deprecated;
- changed defaults, such as a graph capture that now covers more models and
  needs more memory;
- new exact pins the engine brings with it (torch, transformers, kernels);
- a new CUDA requirement, which raises the lane's minimum driver.

## 2. Bump the pin on a branch

Change the bundle's exact pin in `packages/sie_server/bundles/<bundle>.yaml`.
The bundle comment records why the version was chosen and what moved with it.
Do not edit model profiles yet.

## 3. Tier 0: launch contract, on CPU

Tier 0 runs on every pull request that touches an engine, a profile, an
adapter's launch code or a Dockerfile. Inside each built image it renders every
routed profile's launch arguments with the real adapter and parses them with
the engine's own argument parser. It also compares the engine's full flag set
with the snapshot in `conformance/engine-flags/<engine>-<version>.txt`.

On an engine bump, Tier 0 fails until the new snapshot is committed, and its
failure lists the flags the new engine removed and added since the previous
snapshot. Every image build prints the new snapshot in a collapsible log
group named after the file, and the CUDA 13 image workflow also uploads it as
the `engine-launch-<bundle>` artifact. Commit it, then read the diff:

- A removed flag that an adapter or profile still passes fails every route
  that uses it, by name.
- A flag that became an ambiguous prefix fails the same way. SIE passes flags
  only by their exact declared name.

The unit test `packages/sie_server/tests/adapters/test_sglang_launch_flags.py`
holds every routed profile to the committed snapshot of the engine its bundle
pins, so a stale spelling also fails the ordinary Python check.

## 4. Fix spellings in the adapter, not profile by profile

When the engine renames a flag the adapter emits, override the spelling on the
adapter class that the bumped bundle loads, as the CUDA 13 adapters do with
`_PREFILL_GRAPH_OFF_FLAG` and `_MAMBA_STRATEGY_FLAG`. Rename a flag in a profile's
`extra_launch_args` only when the profile itself passes it. Re-run Tier 0 until
every route parses.

## 5. Tier 1: GPU smoke of the affected lane

Every profile of the affected lane must start and answer one request per
declared task on its GPU class: encode, score, extract or generate. Run the
tightest-memory profiles first. A changed default, such as a new graph capture,
shows up here as an out-of-memory error at startup, not in Tier 0.

Record, for each profile, the GPU, the observed driver and CUDA version, the
image digest, the result and the duration. Profiles that need scarce
multi-GPU shapes are advisory in the release gate and blocking on the engine
bump itself.

## 6. Matrix diff

Generate the compatibility matrix for the branch and compare it with the
current release:

```console
$ mise exec -- uv run --frozen --project . python -m tools.ci.compatibility_matrix generate \
    --version <next-version> --evidence <tier-1-evidence>.json --output /tmp/next.json
$ mise exec -- uv run --frozen --project . python -m tools.ci.compatibility_matrix diff \
    docs/compatibility/sie-compatibility-v<current>.json /tmp/next.json
```

The evidence file has the shape of `docs/compatibility/gpu-smoke-v0.10.1.json`:
the image digest of each lane under `images`, and one entry per smoke run under
`runs`. Without it every cell of the new release is at most `declared`, and the
diff shows lost evidence instead of the upgrade's outcome.

The diff lists every cell whose status changed and every lane whose engine
version, CUDA runtime or minimum driver changed. A raised minimum driver, or a
cell that becomes `unsupported`, is an environment break for users: list it
under BREAKING in the release notes.

## 7. Release notes

Add an "Engine upgrades" entry with the old and new versions, the renamed
flags, any profile whose behaviour changed, and the lanes whose minimum driver
moved.

## 8. Rollback

The previous versioned image stays pullable by its tag and digest. To roll
back, revert the pin and the snapshot change, and point deployments at the
previous versioned image.

## Pull request checklist

```markdown
### Engine upgrade

- Engine: <name> <old version> -> <new version>
- Bundles and lanes: <bundle> (<lane>), ...
- Upstream notes read: <links>

### Tier 0: launch contract
- [ ] `conformance/engine-flags/<engine>-<new version>.txt` committed
- Flags removed: <list or none>
- Flags added: <list or none>
- [ ] Adapter spellings updated: <classes>
- [ ] Every routed profile parses on every affected lane

### Tier 1: GPU smoke
| Lane | Profile | GPU | Driver | Result | Duration |
| --- | --- | --- | --- | --- | --- |
| | | | | | |

- [ ] Tightest-memory profiles started
- [ ] Multi-GPU profiles run, or the capacity override is recorded

### Matrix diff
- Changed cells: <count, with the newly unsupported ones listed>
- Minimum driver changes: <lane: old -> new, or none>

### Release notes
- [ ] "Engine upgrades" entry drafted
- [ ] BREAKING entry for any raised driver floor or newly unsupported cell
```
