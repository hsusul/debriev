"""Shared schema helpers."""

from pydantic import BaseModel, ConfigDict


class ORMModel(BaseModel):
    """Base response model configured for ORM serialization."""

    model_config = ConfigDict(from_attributes=True)

