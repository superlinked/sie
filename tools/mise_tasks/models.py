#!/usr/bin/env -S uv run --frozen --project . --package sie-sdk python
# fmt: off
#MISE description="Inspect model catalog and bundle assignments"
#USAGE flag "-a --adapter <adapter>" help="Filter by adapter type (e.g., sglang, sentence_transformer)"
#USAGE flag "-b --bundle <bundle>" help="Filter by bundle name"
#USAGE flag "-m --model <model>" help="Show details for specific model"
#USAGE flag "--missing" help="Show models missing from all bundles"
#USAGE flag "--json" help="Output as JSON"
#USAGE flag "-v --verbose" help="Show full adapter paths"
# fmt: on

"""Model catalog inspection tool.

Shows models, their adapters, bundles, and compatibility information.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from common.env import apply_mise_env, resolve_project_root

apply_mise_env()
try:
    import yaml
except ImportError:
    print("Error: PyYAML required. Run: mise exec -- uv sync")
    sys.exit(1)


def get_project_root() -> Path:
    """Get the project root directory."""
    return resolve_project_root()


def load_bundles(bundles_dir: Path, models_dir: Path) -> dict[str, list[str]]:
    """Load all bundle definitions."""
    from sie_sdk.bundle_utils import match_bundle_models

    bundles: dict[str, list[str]] = {}
    for bundle_file in bundles_dir.glob("*.yaml"):
        with open(bundle_file) as f:
            data = yaml.safe_load(f) or {}
        bundle_name = data.get("name", bundle_file.stem)
        bundles[bundle_name] = match_bundle_models(bundle_file, models_dir)
    return bundles


def load_model_configs(models_dir: Path) -> dict[str, dict]:
    """Load all model configurations."""
    configs: dict[str, dict] = {}
    for config_file in models_dir.glob("*.yaml"):
        with open(config_file) as f:
            cfg = yaml.safe_load(f)
            if not cfg:
                continue
            # Model configs use sie_id (e.g., "BAAI/bge-m3"), fall back to name or filename
            model_name = cfg.get("sie_id") or cfg.get("name") or config_file.stem.replace("__", "/")
            configs[model_name] = cfg
    return configs


def get_adapter_short_name(adapter: str) -> str:
    """Extract short adapter name from full path."""
    if ":" in adapter:
        return adapter.rsplit(":", maxsplit=1)[-1]
    return adapter.rsplit(".", maxsplit=1)[-1]


def get_bundles_for_model(model_name: str, bundles: dict[str, list[str]]) -> list[str]:
    """Find which bundles contain a model."""
    return [bundle_name for bundle_name, models in bundles.items() if model_name in models]


def main() -> int:
    """Main entry point."""
    # Parse flags from environment
    filter_adapter = os.environ.get("usage_adapter")
    filter_bundle = os.environ.get("usage_bundle")
    filter_model = os.environ.get("usage_model")
    show_missing = os.environ.get("usage_missing", "false").lower() == "true"
    output_json = os.environ.get("usage_json", "false").lower() == "true"
    verbose = os.environ.get("usage_verbose", "false").lower() == "true"

    project_root = get_project_root()
    models_dir = project_root / "packages" / "sie_server" / "models"
    bundles_dir = project_root / "packages" / "sie_server" / "bundles"

    # Load data
    bundles = load_bundles(bundles_dir, models_dir)
    configs = load_model_configs(models_dir)

    # Build model info
    model_info = []
    for name, cfg in sorted(configs.items()):
        # Get adapter from first profile's adapter_path
        profiles = cfg.get("profiles", {})
        first_profile = next(iter(profiles.values()), {}) if profiles else {}
        adapter_full = first_profile.get("adapter_path", "unknown")
        adapter_short = get_adapter_short_name(adapter_full)
        model_bundles = get_bundles_for_model(name, bundles)
        inputs = cfg.get("inputs", [])
        outputs = cfg.get("tasks", [])

        info = {
            "name": name,
            "adapter": adapter_short if not verbose else adapter_full,
            "bundles": model_bundles,
            "inputs": inputs,
            "outputs": outputs,
        }

        # Apply filters
        if filter_adapter and filter_adapter.lower() not in adapter_short.lower():
            continue
        if filter_bundle and filter_bundle not in model_bundles:
            continue
        if filter_model and filter_model.lower() not in name.lower():
            continue
        if show_missing and model_bundles:
            continue

        model_info.append(info)

    # Output
    if output_json:
        print(json.dumps(model_info, indent=2))
        return 0

    # Table output
    if not model_info:
        print("No models found matching filters.")
        return 0

    # Calculate column widths
    name_width = max(len(m["name"]) for m in model_info)
    adapter_width = max(len(m["adapter"]) for m in model_info)

    # Header
    print(f"{'Model':<{name_width}}  {'Adapter':<{adapter_width}}  Bundles")
    print("-" * (name_width + adapter_width + 30))

    # Rows
    for m in model_info:
        bundles_str = ", ".join(m["bundles"]) if m["bundles"] else "(none)"
        print(f"{m['name']:<{name_width}}  {m['adapter']:<{adapter_width}}  {bundles_str}")

    print()
    print(f"Total: {len(model_info)} models")

    # Summary by adapter
    if not filter_adapter:
        print("\nBy adapter:")
        adapter_counts: dict[str, int] = {}
        for m in model_info:
            adapter_counts[m["adapter"]] = adapter_counts.get(m["adapter"], 0) + 1
        for adapter, count in sorted(adapter_counts.items(), key=lambda x: -x[1]):
            print(f"  {adapter}: {count}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
