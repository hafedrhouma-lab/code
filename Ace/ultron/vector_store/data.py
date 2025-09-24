from typing import Optional

from pydantic import BaseModel


class ShoppingItemData(BaseModel):
    item_id: int
    content: str
    chain_id: int
    chain_name: str
    order_count: int
    unique_order_dates: int
    vertical: str
    distance: float
    avg_original_item_price_lc: Optional[float]
    is_promotional_category: Optional[bool]
