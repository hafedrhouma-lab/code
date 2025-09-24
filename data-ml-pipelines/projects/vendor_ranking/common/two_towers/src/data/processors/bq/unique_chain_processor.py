import numpy as np
from ..base_processor import BaseProcessor


class UniqueChainProcessor(BaseProcessor):
    def process(self) -> np.array:
        return np.concatenate(
            [
                self.df.chain_id.unique(),
                np.array(
                    [
                        "no_frequent_orders",
                        "no_frequent_clicks",
                        "first_order",
                        "no_recent_clicks"
                    ]
                )
            ],
            axis=0
        )
