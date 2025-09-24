import asyncio
import logging
from datetime import datetime

import newrelic.agent
import pandas as pd
import polars as pl
import structlog
from ipm.utils import cmab_utils as cmab

from ace.perf import perf_manager
from ace.storage import db
from nba import data, SERVICE_NAME
from nba.base_logic import BaseLogic
from nba.hero_banner import model
from nba.input import HeroBannerVariant, HeroBannersResponse, VariantName

CUSTOMER_CONTEXT_QUERY = "SELECT * FROM ipm_account_context WHERE account_id = $1"

logger = structlog.get_logger()


class Logic(BaseLogic):
    NAME = SERVICE_NAME + ":hero_banners_v2"
    MODEL_TAG = model.MODEL_TAG

    _user_context: pd.DataFrame
    _features: pd.DataFrame

    @newrelic.agent.function_trace()
    async def _get_customer_context(self, customer_id) -> pd.DataFrame:
        async with db.connections().acquire() as conn:
            df: pl.DataFrame = await db.fetch_as_df(conn, CUSTOMER_CONTEXT_QUERY, customer_id)
            customer_context = df.to_pandas()
            return customer_context.fillna(0)

    @newrelic.agent.function_trace()
    async def prepare_features(self):
        # 1. Get User Context
        customer_context = await self._get_customer_context(self.request.customer_id)

        # 2. Preprocessing (data coming from the DB should have some scaling and processing)
        # (potentially heavy call, do it in a thread)
        self._user_context = await asyncio.to_thread(cmab.etl_test, customer_context)
        assert self._user_context is not None, "user context is not initialized"

        # 3. Fetch and filter banner context table
        self._features = data.get_banners_context()
        assert self._features is not None, "banner context is not initialized"

    @newrelic.agent.function_trace()
    async def _select_banner(self, variant: VariantName, agent):
        newrelic.agent.add_custom_span_attribute("variant", variant.value)
        # Ideally this models (agents) should be packed in BentoML format and uploaded to the Model Registry,
        # so we can use BentoML runners to run them (in a completely separate process)
        with perf_manager(f"Inference for variant={variant.value}", level=logging.INFO):
            banners = await asyncio.to_thread(
                agent.select_banner,
                user_context=self._user_context,
                banners_context=self._features,
                account_id=self.request.customer_id,
            )
            return HeroBannerVariant(name=variant, banners=banners)

    @newrelic.agent.function_trace()
    async def predict(self) -> HeroBannersResponse:
        variants_output = []
        if self._user_context.empty or self._features.empty:
            variants_output = [
                HeroBannerVariant(name=variant, banners=[]) for variant in self.agents_manager.agents.keys()
            ]
        else:
            # Compute the models concurrently (in parallel)
            banners = [self._select_banner(variant, agent) for variant, agent in self.agents_manager.agents.items()]
            variants_output = await asyncio.gather(*banners)

        return HeroBannersResponse(
            timestamp=datetime.now().isoformat(),
            variants=variants_output,
        )
