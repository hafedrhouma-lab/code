from ..base_processor import BaseProcessor


class CuisinesNamesProcessor(BaseProcessor): #TODO: check bq same as feast?
    def process(self) -> list:
        self.df['cuisine'] = self.df['tlabel'].str.lower().str.split()
        df_unique_cuisine_names = self.df.explode('cuisine').drop_duplicates(
            subset=['cuisine']).reset_index(drop=True)

        return df_unique_cuisine_names['cuisine'].dropna()
