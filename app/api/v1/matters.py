"""Matter routes."""

from uuid import UUID

from fastapi import APIRouter, Depends, status
from sqlalchemy.orm import Session

from app.api.deps import get_db_session
from app.core.exceptions import NotFoundError
from app.repositories.matters import MatterRepository
from app.schemas.matter import MatterCreate, MatterRead

router = APIRouter(prefix="/api/v1", tags=["matters"])


@router.post("/matters", response_model=MatterRead, status_code=status.HTTP_201_CREATED)
def create_matter(payload: MatterCreate, db: Session = Depends(get_db_session)):
    repository = MatterRepository(db)
    matter = repository.create(payload)
    db.commit()
    db.refresh(matter)
    return matter


@router.get("/matters/{matter_id}", response_model=MatterRead)
def get_matter(matter_id: UUID, db: Session = Depends(get_db_session)):
    matter = MatterRepository(db).get(matter_id)
    if matter is None:
        raise NotFoundError("Matter not found.")
    return matter

