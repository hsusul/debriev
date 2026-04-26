"""Top-level API router."""

from fastapi import APIRouter, Depends

from app.api.deps import get_app_settings
from app.api.v1.assertions import router as assertions_router
from app.api.v1.audit import router as audit_router
from app.api.v1.claims import router as claims_router
from app.api.v1.drafts import router as drafts_router
from app.api.v1.matters import router as matters_router
from app.api.v1.sources import router as sources_router
from app.api.v1.verification import router as verification_router
from app.config import Settings

router = APIRouter()


@router.get("/health", tags=["health"])
def health(settings: Settings = Depends(get_app_settings)) -> dict[str, str]:
    """Simple health endpoint."""

    return {"status": "ok", "app": settings.app_name}


router.include_router(matters_router)
router.include_router(sources_router)
router.include_router(drafts_router)
router.include_router(assertions_router)
router.include_router(claims_router)
router.include_router(verification_router)
router.include_router(audit_router)

