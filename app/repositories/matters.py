"""Matter repository."""

from uuid import UUID

from sqlalchemy.orm import Session

from app.models import Matter
from app.schemas.matter import MatterCreate


class MatterRepository:
    """Persistence helpers for matters."""

    def __init__(self, session: Session) -> None:
        self.session = session

    def create(self, payload: MatterCreate) -> Matter:
        matter = Matter(**payload.model_dump())
        self.session.add(matter)
        self.session.flush()
        return matter

    def get(self, matter_id: UUID) -> Matter | None:
        return self.session.get(Matter, matter_id)

