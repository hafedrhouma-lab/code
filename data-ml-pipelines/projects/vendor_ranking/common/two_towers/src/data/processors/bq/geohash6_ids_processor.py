import numpy as np
from ..base_processor import BaseProcessor


class GeohashIdsProcessor(BaseProcessor):
    def process(self):
        self.df['geohash6'] = self.df['geohash'].astype(str).str[:6]
        return np.array(self.df['geohash6'].unique())
