from enum import Enum
from typing import Optional

from pydantic import BaseModel


class CountryCode(str, Enum):
    kw = ("kw",)
    sa = ("sa",)
    bh = ("bh",)
    ae = ("ae",)
    om = ("om",)
    qa = ("qa",)
    jo = ("jo",)
    eg = ("eg",)
    iq = "iq"


class VariantName(str, Enum):
    REFINED_REWARDS_RL = "refined_rewards_rl"
    UPLIFT_MODEL_RL = "uplift_model_rl"
    NEURAL_NETWORKS = "neural_networks"


class HeroBannerVariant(BaseModel):
    name: VariantName
    banners: list[Optional[str]]


class HeroBannersResponse(BaseModel):
    timestamp: str
    variants: list[HeroBannerVariant]


class HeroBannersRequest(BaseModel):
    country_code: CountryCode
    customer_id: int
