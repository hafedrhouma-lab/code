import pydantic as pd
from pydantic import Field


class UserOfflineFeatures(pd.BaseModel):
    freq_items: str = Field(default="no_frequent_items")
    freq_items_names: str = Field(default="no_frequent_items")
    prev_items: str = Field(default="first_item")
    prev_items_names: str = Field(default="first_item")
    chain_prev_items: str = Field(default="discovery_order")
    chain_prev_items_names: str = Field(default="discovery_order")
