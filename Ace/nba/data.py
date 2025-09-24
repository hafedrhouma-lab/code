import datetime as dt
import typing as t
from functools import cache

import newrelic.agent
import pandas as pd
import polars as pl
import pytz
import structlog
from ipm.utils import cmab_utils as cmab

from ace.storage import db

logger = structlog.get_logger()

BANNERS_CONTEXT: t.Optional[pd.DataFrame] = None
BANNERS_CONTEXT_LATEST_DATE: t.Optional[dt.datetime] = None
SNAPSHOT_DATE_COL: str = "snapshot_date"


def is_loaded() -> bool:
    dfs = [
        (BANNERS_CONTEXT, "banners_context"),
    ]
    for df, df_name in dfs:
        if df is None:
            logger.warning(f"Data frame {df_name} is not loaded")
            return False
        if df.empty:
            logger.warning(f"Data frame {df_name} is empty")
            return False
    return True


INFERENCE_BANNER_QUERY = f"""
    SELECT {SNAPSHOT_DATE_COL}, {{banner_columns}}
    FROM ipm_banner_context
    WHERE {SNAPSHOT_DATE_COL} = (SELECT max({SNAPSHOT_DATE_COL}) FROM ipm_banner_context) AND banner_group IN ('nba')
    AND $1 BETWEEN start_date AND end_date;
"""


COUNTRY_TIMEZONE = {
    2: pytz.timezone("Asia/Riyadh"),
    10: pytz.timezone("Asia/Baghdad"),
    5: pytz.timezone("Asia/Muscat"),
    3: pytz.timezone("Asia/Bahrain"),
    6: pytz.timezone("Asia/Qatar"),
    4: pytz.timezone("Asia/Dubai"),
    9: pytz.timezone("Africa/Cairo"),
    8: pytz.timezone("Asia/Amman"),
    1: pytz.timezone("Asia/Kuwait"),
}

COUNTRY_CODE_TO_COUNTRY_ID = {
    "sa": 2,
    "iq": 10,
    "om": 5,
    "bh": 3,
    "qa": 6,
    "ae": 4,
    "eg": 9,
    "jo": 8,
    "kw": 1,
}

BANNER_ID_TO_NAME = {
    0: "tpro_sub",
    1: "tpro_non_sub",
    2: "food",
    3: "tmart",
    4: "grocery",
}


def time_of_day(hour: int) -> str:
    if 0 <= hour <= 5:
        return "midnight"
    elif 6 <= hour <= 11:
        return "morning"
    elif 12 <= hour <= 17:
        return "afternoon"
    elif 18 <= hour <= 23:
        return "evening"
    else:
        raise ValueError(f"Hour must be between 0 and 23. Specified hour={hour}.")


def get_day_interval(country_code) -> (str, dt.datetime):
    country_id = COUNTRY_CODE_TO_COUNTRY_ID.get(country_code.lower())
    now = dt.datetime.now().astimezone(COUNTRY_TIMEZONE.get(country_id, pytz.utc))
    day_interval = time_of_day(now.hour)
    return day_interval


def map_banners(banners):
    if not banners:
        return []
    return [BANNER_ID_TO_NAME.get(b) for b in banners]


@cache
def get_continuous_banner_features() -> list[str]:
    if isinstance(cmab.CONTINUOUS_FEATURES_BANNER, dict):
        return list(cmab.CONTINUOUS_FEATURES_BANNER.keys())
    return cmab.CONTINUOUS_FEATURES_BANNER


def prepare_banner_inference_data(banners: pl.DataFrame) -> pd.DataFrame:
    banners_df: pd.DataFrame = banners.to_pandas()
    # drop optional snapshot date column
    banners_df.drop(SNAPSHOT_DATE_COL, axis="columns", inplace=True, errors="ignore")
    # fill empty data with zeros, otherwise it can break inference
    banners_df.fillna(0, inplace=True)
    banners_df.set_index("banner_id", inplace=True)

    banners_df["banner_discount_value"] = banners_df["banner_discount_value"].astype("float")

    continuous_banner_features = get_continuous_banner_features()
    banner_inference_dict = pd.concat(
        [
            banners_df[cmab.CATEGORICAL_FEATURES_BANNER].reset_index(drop=True),
            pd.DataFrame(
                cmab.CT_BANNER_CONTEXT.fit_transform(banners_df[continuous_banner_features]),
                columns=cmab.CT_BANNER_CONTEXT.get_feature_names_out(),
            ),
        ],
        axis=1,
    )
    banner_inference_dict.set_index(banners_df.index, inplace=True)
    return banner_inference_dict


@cache
def build_banner_inference_query() -> str:
    banner_columns = ",\n".join(
        ["banner_id"]
        + [col for col in cmab.CATEGORICAL_FEATURES_BANNER]
        + [col for col in list(cmab.CONTINUOUS_FEATURES_BANNER.keys())]
    )
    return INFERENCE_BANNER_QUERY.format_map({"banner_columns": banner_columns})


@newrelic.agent.function_trace()
async def refresh_banners_inference_data() -> (pd.DataFrame, dt.datetime):
    """Load banners data from the latest available snapshot.
    If no new data is found, then use the previously loaded one.
    If no data is found at all, then raise exception.
    """
    global BANNERS_CONTEXT
    global BANNERS_CONTEXT_LATEST_DATE

    today_date = dt.datetime.utcnow().date()
    query = build_banner_inference_query()

    logger.info("Getting banner inference data ...")
    async with db.connections().acquire() as conn:
        df: "pl.DataFrame" = await db.fetch_as_df(conn, query, today_date)

    if df.is_empty():
        if BANNERS_CONTEXT is None:
            raise RuntimeError("No banner data found at all!")

        logger.info(f"No NEW banner data found to fetch. Latest {SNAPSHOT_DATE_COL}={BANNERS_CONTEXT_LATEST_DATE}")
        return BANNERS_CONTEXT_LATEST_DATE

    # refresh the latest snapshot date
    max_snapshot_date_df = df.select(pl.col(SNAPSHOT_DATE_COL).max())
    BANNERS_CONTEXT_LATEST_DATE = max_snapshot_date_df[0, 0]

    # refresh banners' inmemory data
    BANNERS_CONTEXT = prepare_banner_inference_data(df)

    logger.info(
        f"Banner data have been loaded in memory {df.estimated_size('mb'):.3f} mb. "
        f"Today date: {today_date}. "
        f"New {SNAPSHOT_DATE_COL}: {BANNERS_CONTEXT_LATEST_DATE}"
    )
    return BANNERS_CONTEXT, BANNERS_CONTEXT_LATEST_DATE


def get_banners_context() -> pd.DataFrame:
    if BANNERS_CONTEXT is None:
        raise RuntimeError("Banner inference data is not loaded")

    return BANNERS_CONTEXT
