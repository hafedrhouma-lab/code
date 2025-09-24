from ..base_processor import BaseProcessor


class UserAreaIdsProcessor(BaseProcessor): #TODO: check bq same as feast?
    def process(self):
        area_ids_array = self.df["unique_area_ids"].iloc[0]
        return area_ids_array.tolist()