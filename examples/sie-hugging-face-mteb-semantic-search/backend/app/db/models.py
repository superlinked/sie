from datetime import datetime
from typing import Optional

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    Integer,
    String,
    BigInteger,
    ForeignKey,
    func,
    JSON,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .session import Base


class Storage(Base):
    __tablename__ = "storage_ids"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    storage_id: Mapped[str] = mapped_column(String, unique=True, index=True)
    description: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    models: Mapped[list["Model"]] = relationship(
        back_populates="storage", cascade="all, delete-orphan"
    )


class Model(Base):
    __tablename__ = "models"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    storage_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("storage_ids.id", ondelete="CASCADE")
    )
    hf_id: Mapped[str] = mapped_column(String, index=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    # Core ids / times
    author: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    sha: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    created_at_hf: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    last_modified: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    # Flags
    private: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)
    disabled: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)

    # Metrics
    downloads: Mapped[Optional[int]] = mapped_column(BigInteger, nullable=True)
    downloads_all_time: Mapped[Optional[int]] = mapped_column(BigInteger, nullable=True)
    downloads_30d: Mapped[Optional[int]] = mapped_column(BigInteger, nullable=True)
    likes: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    trending_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Taxonomy / tags
    tags: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)
    pipeline_tag: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    library_name: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    mask_token: Mapped[Optional[str]] = mapped_column(String, nullable=True)

    # Structured metadata kept in DB (small).
    # README is NOT stored locally — fetch live from HF on demand via
    # GET /api/models/readme/{hf_id}. See overview.md.
    config: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    card_data: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)

    # MTEB benchmark results: compact [{task_name, main_score}] only.
    # The raw nested per-subset/per-split JSON is NOT stored (too large).
    mteb_scores: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)

    # Descriptions
    short_description: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)
    long_description: Mapped[Optional[str]] = mapped_column(String(2048), nullable=True)

    storage: Mapped["Storage"] = relationship(back_populates="models")

