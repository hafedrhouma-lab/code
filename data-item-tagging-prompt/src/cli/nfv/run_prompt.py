import os
import fire
import asyncio

from tutils.db_utils import BigQuery

from src.utils.log import Logger
from src.utils.helpers import (
    save_jsonl_to_gcp,
    read_jsonl_from_gcp,
    save_jsonl_locally,
    count_lines_in_jsonl,
    read_jsonl_and_get_data,
    delete_directory
)

from src.utils.api_request_parallel_processor import process_api_requests_from_file

from src import (
    PROJECT_NAME,
    BUCKET_NAME,
    OPENAI_REQUESTS_FOLDER,
    OPENAI_RESPONSES_FOLDER
)

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


class TagNFVItems:
    def __init__(self):
        self._logger = Logger(self.__class__.__name__).get_logger()
        self.bq = BigQuery()

    def run_prompt(self, execution_date):

        # jsonl_data = read_jsonl_from_gcp(
        #     bucket_name=BUCKET_NAME,
        #     file_path=f'{PROJECT_NAME}/nfv/{OPENAI_REQUESTS_FOLDER}/'
        #               f'{execution_date}/requests_tagging_{execution_date}.jsonl'
        # )
        #
        # os.makedirs(requests_local_dir, exist_ok=True)
        # save_jsonl_locally(jsonl_data, requests_local_filename)
        num_requests = count_lines_in_jsonl(requests_local_filename)

        self._logger.info(
            f'Starting NFV data tagging using ChatGPT4: '
            f'{num_requests} items to process.'
        )

        asyncio.run(
            process_api_requests_from_file(
                requests_filepath=requests_local_filename,
                save_filepath=save_local_filepath,
                request_url='https://api.openai.com/v1/chat/completions',
                api_key=os.getenv("OPENAI_API_KEY"),
                max_requests_per_minute=200,
                max_tokens_per_minute=250000,
                token_encoding_name="cl100k_base",
                max_attempts=100,
                logging_level=20,
            )
        )

        data_to_write = read_jsonl_and_get_data(save_local_filepath)

        # save_jsonl_to_gcp(
        #     data_to_write,
        #     bucket_name=BUCKET_NAME,
        #     file_path=f'{PROJECT_NAME}/nfv/{OPENAI_RESPONSES_FOLDER}/'
        #               f'{execution_date}/responses_tagging_{category}_{execution_date}.jsonl'
        # )
        #
        # delete_directory(self, requests_local_dir)


if __name__ == "__main__":
    fire.Fire(TagNFVItems)