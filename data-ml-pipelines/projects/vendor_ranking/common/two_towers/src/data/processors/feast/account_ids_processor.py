from ..base_processor import BaseProcessor


class AccountIdsProcessor(BaseProcessor):
    def process(self) -> list:
        return list(self.df.account_id.unique())
