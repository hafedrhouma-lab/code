import os
import fire
import pandas as pd

from src.utils.log import Logger
from src.prompt.food.regex_tagging import get_label_regex


__here__ = os.path.dirname(os.path.abspath(__file__))
requests_local_dir = os.path.join(__here__, 'tmp_requests')
requests_local_filename = os.path.join(
    requests_local_dir,
    "requests_to_parallel_process.jsonl"
)
save_local_filepath = os.path.join(
    requests_local_dir,
    "requests_to_parallel_process_results.jsonl"
)

data_file_path = 'src/data/datasets/food/data_most_selling_food_vendor_items.csv'


class TagItems:
    def __init__(self):
        self._logger = Logger(self.__class__.__name__).get_logger()

    def tag_items(self, execution_date):
        data = pd.read_csv(data_file_path)
        data['item_name_en'] = data['item_name_en'].astype(str)
        data['item_description_en'] = data['item_description_en'].astype(str)

        data['label'] = data.apply(get_label_regex, axis=1)

        data.to_csv('src/data/datasets/food/tagged_items_regex.csv')


if __name__ == "__main__":
    fire.Fire(TagItems)
