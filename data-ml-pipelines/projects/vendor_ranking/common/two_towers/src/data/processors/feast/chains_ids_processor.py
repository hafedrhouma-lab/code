import numpy as np
from ..base_processor import BaseProcessor


class ChainsIdsProcessor(BaseProcessor):
    def process(self):
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
