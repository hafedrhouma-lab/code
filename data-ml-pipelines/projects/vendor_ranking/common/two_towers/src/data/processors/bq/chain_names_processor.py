from ..base_processor import BaseProcessor


class ChainNamesProcessor(BaseProcessor):
    def process(self):
        return self.df[['chain_name', 'chain_id']]

