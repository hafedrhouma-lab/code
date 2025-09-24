from typing import List
from pydantic import BaseModel, Field


class GeographicalUnitResponse(BaseModel):
    chain_id: str
    tag_1: str = Field(..., description="Geographical cuisine tag")
    explanation_tag_1: str = Field(..., description="Reason for cuisine choice (max 100 chars)")


class DishTypeUnitResponse(BaseModel):
    chain_id: str
    tag_1: str = Field(..., description="First dish type tag")
    tag_2: str = Field(..., description="Second dish type tag")
    explanation_tag_1: str = Field(..., description="Reason for tag_1 choice (max 100 chars)")
    explanation_tag_2: str = Field(..., description="Reason for tag_2 choice (max 100 chars)")


class GeographicalResponse(BaseModel):
    chains: List[GeographicalUnitResponse] = Field(..., description="List of chain cuisine responses")


class DishTypeResponse(BaseModel):
    chains: List[DishTypeUnitResponse] = Field(..., description="List of chain cuisine responses")