import fire
import os
import pandas as pd
import json

from tutils.db_utils import BigQuery
from tutils import fs_utils as fs

from src.utils.log import Logger
from src.utils.helpers import (
    read_jsonl_from_gcp,
    save_jsonl_locally,
    count_lines_in_jsonl,
    delete_directory
)

from src import (
    PROJECT_NAME,
    BUCKET_NAME,
    TAGGED_ITEM_FOLDER,
    OPENAI_RESPONSES_FOLDER
)

__here__ = os.path.dirname(os.path.abspath(__file__))
responses_local_dir = os.path.join(__here__, 'tmp_requests')
responses_local_filename = os.path.join(
    responses_local_dir,
    "requests_to_parallel_process_results.jsonl"
)


class ParseResponseFile:
    def __init__(self):
        self._logger = Logger(self.__class__.__name__).get_logger()
        self.bq = BigQuery()

    def parse_response(self, execution_date):
        # jsonl_data = read_jsonl_from_gcp(
        #     bucket_name=BUCKET_NAME,
        #     file_path=f'{PROJECT_NAME}/nfv/{OPENAI_RESPONSES_FOLDER}/'
        #               f'{execution_date}/responses_tagging_{category}_{execution_date}.jsonl'
        # )

        #os.makedirs(responses_local_dir, exist_ok=True)
        #save_jsonl_locally(jsonl_data, responses_local_filename)
        num_requests = count_lines_in_jsonl(responses_local_filename)

        self._logger.info(
            f'Starting jsonl data processing: '
            f'{num_requests} lines to process.'
        )

        output_df = self.process_output(responses_local_filename)

        # fs.save_to_gcs(
        #     obj=output_df,
        #     gcs_path=f"{BUCKET_NAME}/{PROJECT_NAME}/nfv/{TAGGED_ITEM_FOLDER}/"
        #              f"{execution_date}/tagged_items_{category}_{execution_date}.csv"
        # )
        self._logger.info(
            f'{output_df.shape[0]} Chatgpt tagged Items saved on GCP'
        )

        self._logger.info(
            f'{num_requests - output_df.shape[0]} Items skipped or having error.'
        )

        output_df.to_csv('parsed_gpt_data.csv', index=False)

        #delete_directory(self, responses_local_dir)

    def process_output(self, jsonl_file_path):
        data_rows = []

        with open(jsonl_file_path, 'r') as jsonl_file:
            for line in jsonl_file:
                try:
                    line_json = json.loads(line)

                    user_input_str = line_json[0]["messages"][1]['content']
                    user_input_json = json.loads(user_input_str)

                    assistant_response_str = line_json[1]["choices"][0]["message"]['content']
                    assistant_response_json = json.loads(assistant_response_str)

                    data_row = {
                        'item_id': assistant_response_json.get('item_id'),
                        'item_name': user_input_json.get('item_name'),
                        'item_description': user_input_json.get('item_description'),
                        'tags': assistant_response_json.get('tags')
                    }

                    data_rows.append(data_row)

                except (KeyError, IndexError, AttributeError, json.JSONDecodeError) as e:
                    self._logger.info("Skipped a line due to an error:", e)
                    continue

        df = pd.DataFrame(data_rows)
        return df


if __name__ == "__main__":
    fire.Fire(ParseResponseFile)
