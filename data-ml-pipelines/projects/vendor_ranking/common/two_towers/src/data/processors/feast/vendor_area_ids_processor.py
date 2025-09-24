from ..base_processor import BaseProcessor


class VendorAreaIdsProcessor(BaseProcessor): #TODO: check bq same as feast?
    def process(self):
        self.df['area_id'] = self.df['chain_freq_area_ids'].str.split(' ')
        df_vendor_area_ids = self.df.explode('area_id')

        vendor_area_ids = df_vendor_area_ids['area_id'].astype(str).unique()
        cleaned_list = list(filter(lambda x: x and x != 'None', vendor_area_ids))

        return cleaned_list

