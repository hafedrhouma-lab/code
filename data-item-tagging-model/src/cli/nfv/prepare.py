import fire
import pandas as pd

from src.utils.log import Logger

from src.data.processor.utils import remove_noise
from src.data.processor.nfv.clean_data import clean_tags, clean_tags_list
from src.data.processor.utils import create_dummy_vector
from src.data.processor.nfv.tag_mapping import expected_tag_list


data_file_path = 'src/data/datasets/nfv/parsed_gpt_data.csv'


class PrepareData:
    def __init__(self):
        self._logger = Logger(self.__class__.__name__).get_logger()

    def prepare(self, execution_date):
        data = pd.read_csv(data_file_path)

        data['combined'] = data['item_name'] + ' ' + data['item_description']
        data['tags'] = data['tags'].apply(clean_tags)
        data['cleaned_text'] = data['combined'].apply(remove_noise)

        data = data.dropna(subset=['tags']).reset_index(drop=True)
        data['tags'] = data['tags'].apply(lambda x: str(x).lower())

        data['tag_list'] = data['tags'].str.replace(' ', '').str.split(',')

        data['tag_list'] = data['tag_list'].apply(clean_tags_list)

        data['dummy_vector'] = data['tag_list'].apply(
            lambda tags: create_dummy_vector(expected_tag_list, tags)
        )

        indexes = data[
            data['dummy_vector'].apply(lambda x: all(elem == 0 for elem in x))
        ].index.tolist()

        data = data[~data.index.isin(indexes)]
        data = data.reset_index(drop=True)

        data.to_csv('src/data/datasets/nfv/cleaned_data.csv', index=False)


if __name__ == "__main__":
    fire.Fire(PrepareData)
