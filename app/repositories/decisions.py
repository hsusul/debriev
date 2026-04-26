"""User decision repository."""

from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import UserDecision
from app.schemas.verification import UserDecisionCreate


class DecisionsRepository:
    """Persistence helpers for user decisions."""

    def __init__(self, session: Session) -> None:
        self.session = session

    def create(self, run_id: UUID, payload: UserDecisionCreate) -> UserDecision:
        decision = UserDecision(verification_run_id=run_id, **payload.model_dump())
        self.session.add(decision)
        self.session.flush()
        return decision

    def list_by_run(self, run_id: UUID) -> list[UserDecision]:
        stmt = select(UserDecision).where(UserDecision.verification_run_id == run_id).order_by(UserDecision.created_at)
        return list(self.session.scalars(stmt))

