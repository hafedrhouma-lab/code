import asyncio
import os
import pathlib
import pickle
from functools import cache
from typing import List, Optional

import aioboto3
import decouple
import structlog
from langchain.vectorstores import VectorStore
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

logger = structlog.get_logger()

STAGE = decouple.config("STAGE", default="qa")
BUCKET_NAME = f"talabat-{STAGE}-ace-airflow-eu-west-2"
BUCKET_DIRECTORY = "chatbot" if STAGE == "prod" else "chatbot_sample"


LOCAL_PATH = pathlib.Path(__file__).parent.resolve()


@cache
def _get_prompt(file: pathlib.Path) -> str:
    with open(file) as f:
        return f.read()


def get_system_message(prompt: pathlib.Path, vars_to_change: dict[str, str] = None) -> str:
    system_template = _get_prompt(prompt)
    if vars_to_change:
        for var in vars_to_change:
            system_template = system_template.replace(var, vars_to_change[var])

    return system_template


class ChatConfig(BaseModel):
    engine: str
    max_tokens: int
    temperature: float
    functions: Optional[List[dict]] = None
    function_call: Optional[str] = None


@cache
def get_config(file: pathlib.Path) -> ChatConfig:
    return ChatConfig.parse_file(file)


def get_retriever_local(k=3) -> dict:
    # Load the retriever object from the pickle file
    file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "fixtures/retrievers/retriever.pickle"))
    with open(file_path, "rb") as f:
        vectorstore = pickle.load(f)

    retriever = vectorstore.as_retriever(search_kwargs={"k": k})

    # Initialize the dictionary with a single entry for the default retriever
    retrievers = {"default": retriever}

    return retrievers


async def get_retriever_s3(k=3) -> dict:
    retrievers = {}

    def load_vs(pickle_file_path: str) -> VectorStore:
        """
        Load the retriever object from the pickle file
        """
        with open(pickle_file_path, "rb") as f:
            return pickle.load(f)

    async def download_and_process(key):
        if os.path.splitext(key)[1] == ".pickle":
            # Download the pickle file from S3 to local disk
            pickle_file_path = os.path.basename(key)
            logger.info(f"Downloading {pickle_file_path} from S3")
            await bucket._download_file(key, pickle_file_path)
            logger.info(f"Successfully downloaded {pickle_file_path} from S3")

            vectorstore = await run_in_threadpool(load_vs, pickle_file_path)

            # Get the vendor ID from the filename
            vendor_id = os.path.splitext(pickle_file_path)[0].replace("tmart_item_names_openai_", "")

            # Convert the vectorstore object to a retriever object and store it in the dictionary
            retrievers[str(vendor_id)] = vectorstore.as_retriever(search_kwargs={"k": k})

    session = aioboto3.Session()
    async with session.resource("s3") as s3:
        bucket = await s3.Bucket(BUCKET_NAME)
        objects = bucket.objects.filter(Prefix=f"{BUCKET_DIRECTORY}/tmart_item_names_openai_")
        keys = [obj.key async for obj in objects]

        await asyncio.gather(*[download_and_process(key) for key in keys])

    return retrievers
