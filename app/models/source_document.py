"""Source document ORM model."""

from uuid import UUID

from sqlalchemy import Enum, Float, ForeignKey, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.enums import ParserStatus, SourceType
from app.models.base import Base, IdMixin, TimestampMixin


class SourceDocument(Base, IdMixin, TimestampMixin):
    """Evidence source metadata for a matter."""

    __tablename__ = "source_documents"

    matter_id: Mapped[UUID] = mapped_column(
        ForeignKey("matters.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    file_name: Mapped[str] = mapped_column(String(255), nullable=False)
    source_type: Mapped[SourceType] = mapped_column(
        Enum(SourceType, name="source_type_enum"),
        nullable=False,
    )
    raw_file_path: Mapped[str] = mapped_column(Text, nullable=False)
    parser_status: Mapped[ParserStatus] = mapped_column(
        Enum(ParserStatus, name="parser_status_enum"),
        default=ParserStatus.PENDING,
        nullable=False,
    )
    parser_confidence: Mapped[float | None] = mapped_column(Float)

    matter = relationship("Matter", back_populates="source_documents")
    evidence_bundles = relationship(
        "EvidenceBundle",
        secondary="evidence_bundle_source_documents",
        back_populates="source_documents",
    )
    segments = relationship("Segment", back_populates="source_document", cascade="all, delete-orphan")
