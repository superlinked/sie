from pathlib import Path
from functools import lru_cache

_PROMPTS_DIR = Path(__file__).parent


@lru_cache(maxsize=None)
def _read_template(name: str) -> str:
    path = _PROMPTS_DIR / name
    if not path.suffix:
        path = path.with_suffix(".md")
    if not path.is_file():
        available = ", ".join(p.stem for p in _PROMPTS_DIR.glob("*.md"))
        raise FileNotFoundError(
            f"Prompt '{name}' not found. Available: {available}"
        )
    return path.read_text(encoding="utf-8")


def load_prompt(name: str, **kwargs: str) -> str:
    """Load a prompt template by name and fill in placeholders.

    Usage:
        prompt = load_prompt("summarize_model", model_id="BAAI/bge-large-en", readme="...")
    """
    template = _read_template(name)
    if kwargs:
        return template.format(**kwargs)
    return template


def list_prompts() -> list[str]:
    """Return the names of all available prompt templates."""
    return sorted(p.stem for p in _PROMPTS_DIR.glob("*.md"))
