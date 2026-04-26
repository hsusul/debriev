"""Matter ORM model."""

from sqlalchemy import String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, IdMixin, TimestampMixin


class Matter(Base, IdMixin, TimestampMixin):
    """Top-level litigation workspace."""

    __tablename__ = "matters"

    name: Mapped[str] = mapped_column(String(255), nullable=False)
    court: Mapped[str | None] = mapped_column(String(255))
    jurisdiction: Mapped[str | None] = mapped_column(String(255))
    status: Mapped[str] = mapped_column(String(50), default="ACTIVE", nullable=False)

    source_documents = relationship(
        "SourceDocument",
        back_populates="matter",
        cascade="all, delete-orphan",
    )
    evidence_bundles = relationship(
        "EvidenceBundle",
        back_populates="matter",
        cascade="all, delete-orphan",
    )
    drafts = relationship("Draft", back_populates="matter", cascade="all, delete-orphan")
