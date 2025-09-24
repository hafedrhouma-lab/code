import re
from ..base_processor import BaseProcessor


class SearchVocabProcessor(BaseProcessor): #TODO: check bq same as feast?
    def process(self) -> list:
        search_words = self.df['most_recent_15_search_keywords'].apply(
            lambda x: re.sub(r'[^\w\s]', '', str(x))).str.split()

        df_exploded = search_words.explode(ignore_index=True)
        df_exploded = df_exploded[df_exploded.str.strip() != '']

        word_counts_df = df_exploded.value_counts().reset_index()
        word_counts_df.columns = ['word', 'count']
        word_counts_df = word_counts_df.sort_values(by='count', ascending=False)

        total_count = word_counts_df['count'].sum()

        word_counts_df['cumsum'] = word_counts_df['count'].cumsum()
        word_counts_df['cum_percentage'] = word_counts_df['cumsum'] / total_count

        filtered_words_df = word_counts_df[word_counts_df['cum_percentage'] <= 0.95]
        filtered_words_df = filtered_words_df.drop(['cumsum', 'cum_percentage'], axis=1)

        return list(filtered_words_df['word'])
