import pandas as pd
from ..base_processor import BaseProcessor


def remove_punctuation(search_query):
    import string
    return (
        search_query.translate(str.maketrans("", "", string.punctuation))
            .lower()
            .strip()
    )


class SearchVocabProcessor(BaseProcessor): #TODO: check bq same as feast?
    def process(self) -> list:
        df_search_vocab_list = self.df

        searches_text = list(df_search_vocab_list["search_words"].values)

        search_words = [
            word
            for search_text in searches_text
            for word in remove_punctuation(search_text).split()
        ]
        searches_df = pd.DataFrame(data=search_words, columns=["search"])
        searches_count = searches_df.search.value_counts()

        searches_vocab = list(
            searches_count[
                (searches_count / searches_count.sum()).cumsum() < 0.95
            ].index.values
        )
        return searches_vocab
