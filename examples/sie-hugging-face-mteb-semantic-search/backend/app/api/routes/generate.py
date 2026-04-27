import json
import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.db import models as db_models
from app.db.session import get_db
from app.prompts import load_prompt
from app.services.openrouter import generate_text
from app.services.chroma import upsert_embedding

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/generate", tags=["generate"])


def _get_model_row(
    db: Session, storage_id: str, hf_id: str
) -> db_models.Model:
    storage = (
        db.query(db_models.Storage)
        .filter(db_models.Storage.storage_id == storage_id)
        .one_or_none()
    )
    if storage is None:
        raise HTTPException(404, f"Storage id '{storage_id}' not found")
    model = (
        db.query(db_models.Model)
        .filter(
            db_models.Model.storage_id == storage.id,
            db_models.Model.hf_id == hf_id,
        )
        .first()
    )
    if model is None:
        raise HTTPException(404, f"Model '{hf_id}' not found")
    return model


def _live_readme_for_prompt(hf_id: str, max_chars: int = 4000) -> str:
    """Live-fetch the HF README for prompt injection. Truncated to max_chars.

    Returns an empty string if the README cannot be loaded — the rest of
    the prompt still has HF metadata + MTEB, so generation can proceed.
    """
    from huggingface_hub import ModelCard

    try:
        text_value = ModelCard.load(hf_id).text or ""
    except Exception as exc:
        logger.info("Live README fetch failed for %s: %s", hf_id, exc)
        return ""
    return text_value[:max_chars]


def _build_model_json(model: db_models.Model) -> str:
    """Build a JSON string of key HF metadata for prompt injection."""
    data = {
        "hf_id": model.hf_id,
        "author": model.author,
        "pipeline_tag": model.pipeline_tag,
        "library_name": model.library_name,
        "downloads_30d": model.downloads_30d,
        "likes": model.likes,
        "tags": model.tags,
        "config": model.config,
        "card_data": model.card_data,
    }
    return json.dumps(data, indent=2, default=str)


def _build_mteb_summary(model: db_models.Model) -> str:
    """Format compact MTEB scores as simple one-per-line summaries."""
    if not model.mteb_scores:
        return "(no MTEB results available)"
    lines: list[str] = []
    for r in model.mteb_scores:
        task_name = r.get("task_name", "unknown")
        main = r.get("main_score")
        if main is None:
            lines.append(f"- {task_name}: (no score)")
            continue
        try:
            score = float(main)
        except (TypeError, ValueError):
            lines.append(f"- {task_name}: (no score)")
            continue
        lines.append(f"- {task_name}: {score:.4f} ({score * 100:.2f}%)")
    return "\n".join(lines)


# --- Render prompt endpoint ---

class RenderRequest(BaseModel):
    storage_id: str
    hf_id: str


class RenderResponse(BaseModel):
    model_config = {"protected_namespaces": ()}

    prompt_text: str
    model_json: str
    mteb_summary: str


@router.post("/render-prompt", response_model=RenderResponse)
def render_prompt(payload: RenderRequest, db: Session = Depends(get_db)):
    """Render the detailed_description prompt with all variables filled in.

    README is no longer stored locally; fetch it live from HuggingFace and
    truncate to 4 000 chars for the prompt.
    """
    model = _get_model_row(db, payload.storage_id, payload.hf_id)
    model_json = _build_model_json(model)
    mteb_summary = _build_mteb_summary(model)

    model_readme = _live_readme_for_prompt(model.hf_id)

    prompt_text = load_prompt(
        "detailed_description",
        model_id=model.hf_id,
        model_json=model_json,
        model_readme=model_readme,
        mteb_summary=mteb_summary,
    )
    return RenderResponse(
        prompt_text=prompt_text,
        model_json=model_json,
        mteb_summary=mteb_summary,
    )


# --- Generate endpoints ---

class GenerateDetailedRequest(BaseModel):
    prompt_text: str = Field(..., description="The (possibly edited) detailed prompt")
    model: Optional[str] = Field(
        None, description="OpenRouter model name (uses default from config if omitted)"
    )


class GenerateFromDetailedRequest(BaseModel):
    storage_id: str
    hf_id: str
    detailed_description: str = Field(
        ..., description="The 6K detailed description output"
    )
    model: Optional[str] = Field(
        None, description="OpenRouter model name (uses default from config if omitted)"
    )


class GenerateResponse(BaseModel):
    text: str


@router.post("/detailed", response_model=GenerateResponse)
def generate_detailed(payload: GenerateDetailedRequest):
    """Generate the 6K detailed description from the prompt text."""
    text = generate_text(payload.prompt_text, max_tokens=16384, model=payload.model)
    return GenerateResponse(text=text)


@router.post("/long", response_model=GenerateResponse)
def generate_long(
    payload: GenerateFromDetailedRequest,
    db: Session = Depends(get_db),
):
    """Generate the 2K long description from the 6K detailed description."""
    db_model = _get_model_row(db, payload.storage_id, payload.hf_id)
    model_json = _build_model_json(db_model)
    mteb_summary = _build_mteb_summary(db_model)

    prompt = load_prompt(
        "long_description",
        detailed_description=payload.detailed_description,
        model_json=model_json,
        mteb_summary=mteb_summary,
    )
    text = generate_text(prompt, max_tokens=8192, model=payload.model)
    return GenerateResponse(text=text)


@router.post("/short", response_model=GenerateResponse)
def generate_short(
    payload: GenerateFromDetailedRequest,
    db: Session = Depends(get_db),
):
    """Generate the 200-char short description from the 6K detailed description."""
    db_model = _get_model_row(db, payload.storage_id, payload.hf_id)
    model_json = _build_model_json(db_model)
    mteb_summary = _build_mteb_summary(db_model)

    prompt = load_prompt(
        "short_description",
        detailed_description=payload.detailed_description,
        model_json=model_json,
        mteb_summary=mteb_summary,
    )
    text = generate_text(prompt, max_tokens=4096, model=payload.model)
    return GenerateResponse(text=text)


# --- Save endpoint ---

class SaveRequest(BaseModel):
    storage_id: str
    hf_id: str
    short_description: Optional[str] = None
    long_description: Optional[str] = None


@router.post("/save")
def save_descriptions(payload: SaveRequest, db: Session = Depends(get_db)):
    """Persist generated descriptions back to the database."""
    model = _get_model_row(db, payload.storage_id, payload.hf_id)
    if payload.short_description is not None:
        model.short_description = payload.short_description
    if payload.long_description is not None:
        model.long_description = payload.long_description
    db.commit()

    try:
        upsert_embedding(
            payload.storage_id,
            payload.hf_id,
            model.short_description,
            model.long_description,
        )
    except Exception as exc:
        logger.warning("Chroma upsert failed for %s: %s", payload.hf_id, exc)

    return {"status": "saved"}
