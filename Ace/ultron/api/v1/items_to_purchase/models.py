import math
from typing import Optional

from pydantic import conlist, BaseModel, validator, conint


class ItemsToPurchaseRequest(BaseModel):
    vendor_id: int
    items_to_purchase: conlist(str, min_items=1)
    limit: conint(ge=1, le=10) = 5


class PurchaseItemRecommendation(BaseModel):
    item_name: str
    global_product_id: str
    global_vendor_id: str


class VendorItemMetadata(BaseModel):
    global_vendor_id: str
    global_product_id: str
    product_sku: str

    @validator("*", pre=True)
    def validate_all_fields(cls, val):
        if val is None or (isinstance(val, float) and math.isnan(val)):
            return ""
        return val

    class Config:
        orm_mode = True


class PurchaseRecommendation(BaseModel):
    name: Optional[str]
    metadata: VendorItemMetadata


class ItemToPurchaseRecommendation(BaseModel):
    original_search: str
    items: list[PurchaseRecommendation]


class VendorIDMapping(BaseModel):
    vendor_id: int
    global_vendor_id: str
