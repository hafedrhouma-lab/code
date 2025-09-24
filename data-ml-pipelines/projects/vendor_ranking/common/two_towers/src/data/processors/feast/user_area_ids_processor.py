from ..base_processor import BaseProcessor


class UserAreaIdsProcessor(BaseProcessor): #TODO: check bq same as feast?
    def process(self):
        return self.df['delivery_area_id'].astype(str).unique()
