import pandas as pd
from ..base_processor import BaseProcessor


class CuisinesNamesProcessor(BaseProcessor): #TODO: check bq same as feast?
    def process(self):
        df_unique_cuisine_names = self.df
        unique_cuisines = [
            x.strip().lower()
            for cuisines in df_unique_cuisine_names['cuisine'].values
            for x in cuisines.split()
        ]
        unique_cuisines_names = pd.Series(unique_cuisines).unique()

        return unique_cuisines_names