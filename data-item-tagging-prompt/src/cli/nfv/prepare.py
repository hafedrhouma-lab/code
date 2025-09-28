import fire
from tutils.db_utils import BigQuery
from tutils import fs_utils as fs

from src.utils.log import Logger
from src.data.datasets.fetch import FetchItem
from src.utils.helpers import read_file, save_jsonl_locally
from src.utils.openai_utils import create_requests_chat_completion


from src import (
    PROJECT_NAME,
    BUCKET_NAME,
    OPENAI_REQUESTS_FOLDER
)

prompt_file = 'src/prompt/nfv/tagging_prompt.txt'


class PrepareTagging:
    def __init__(self):
        self._logger = Logger(self.__class__.__name__).get_logger()
        self.bq = BigQuery()

    def prepare(self, execution_date):
        data = FetchItem('nfv_items').get()

        system_prompt = read_file(prompt_file)

        jobs = create_requests_chat_completion(system_prompt, data)
        # save_jsonl_to_gcp(
        #     jobs,
        #     bucket_name=BUCKET_NAME,
        #     file_path=f'{PROJECT_NAME}/nfv/{OPENAI_REQUESTS_FOLDER}/'
        #               f'{execution_date}/requests_tagging_{execution_date}.jsonl'
        # )
        save_jsonl_locally(jobs, 'src/cli/nfv/tmp_requests/requests_to_parallel_process.jsonl')

        self._logger.info(
            f'{data.shape[0]} Items saved on GCP'
        )


if __name__ == "__main__":
    fire.Fire(PrepareTagging)
