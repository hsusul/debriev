"""User decision ORM model."""

from datetime import datetime
from uuid import UUID

from sqlalchemy import DateTime, Enum, ForeignKey, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.enums import DecisionAction
from app.models.base import Base, IdMixin


class UserDecision(Base, IdMixin):
    """Human override or annotation on a verification run."""

    __tablename__ = "user_decisions"

    verification_run_id: Mapped[UUID] = mapped_column(
        ForeignKey("verification_runs.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    action: Mapped[DecisionAction] = mapped_column(Enum(DecisionAction, name="decision_action_enum"), nullable=False)
    note: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    verification_run = relationship("VerificationRun", back_populates="user_decisions")
