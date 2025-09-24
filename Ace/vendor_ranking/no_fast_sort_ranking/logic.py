import newrelic.agent

from vendor_ranking import SERVICE_NAME
from vendor_ranking.personalized_ranking import model
from vendor_ranking.personalized_ranking.logic import Logic as PersonalizedLogic
from vendor_ranking.price_parity.logic import apply_ranking_penalty


class LogicPenaltyOnly(PersonalizedLogic):
    """CatBoost
    FastSort - NO
    Penalization - YES
    """

    NAME = SERVICE_NAME + ":personalized_no_fast_sort_with_penalty"
    MODEL_TAG = model.MODEL_TAG
    TOP_N_FAST_SORT_LIMIT: int = None  # disable fast sort

    @newrelic.agent.function_trace()
    async def sort(self) -> list[int]:
        final_sorting = await super().sort()
        final_sorting = await apply_ranking_penalty(final_sorting)

        return final_sorting
