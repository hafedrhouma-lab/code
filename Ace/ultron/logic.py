import json
import os
import pickle
import re
from functools import cache
from json import JSONDecodeError
from typing import AsyncIterable, Iterable, TYPE_CHECKING
from typing import Dict, Any
from typing import Tuple
from unittest import mock

import langchain.chat_models.openai as langchain_openai
import newrelic.agent
import numpy as np
import openai
import pydantic
import structlog
from fastapi import HTTPException
from gptcache import Config
from gptcache import cache as semantic_cache
from gptcache.adapter import openai as cache_openai
from gptcache.embedding import SBERT
from gptcache.manager import CacheBase, VectorBase, get_data_manager
from gptcache.similarity_evaluation.distance import SearchDistanceEvaluation
from langchain.chains import OpenAIModerationChain
from langchain.chat_models import ChatOpenAI
from langchain.output_parsers import PydanticOutputParser, OutputFixingParser
from langchain.schema import (
    HumanMessage,
    SystemMessage,
    ChatGeneration,
    BaseMessage,
    AIMessage,
)
from pydantic import parse_obj_as, PostgresDsn
from starlette import status

from ace.configs.manager import ConfigManager
from ace.newrelic import add_transaction_attrs, add_transaction_attr
from ultron import data
from ultron.api.v1.items_to_purchase.models import VendorIDMapping, PurchaseItemRecommendation
from ultron.api.v1.semantic_search.models import VerticalsSearch
from ultron.data import ChatConfig
from ultron.input import UserIntents
from ultron.runners import cross_encoder
from ultron.runners import text_embeddings

if TYPE_CHECKING:
    from structlog.stdlib import BoundLogger

LOG: "BoundLogger" = structlog.get_logger()


# init blacklisted words and competitors lists
def read_blacklist(path) -> set[str]:
    with open(path, "r") as f:
        blacklist = {x.lower() for x in f.read().splitlines()}
    return blacklist


def read_cached_data(path) -> dict[str, list[str]]:
    with open(path, "rb") as f:
        cached_data = pickle.load(f)
    return cached_data


def get_entire_blacklist() -> tuple[set[str], set[str]]:
    blacklisted = read_blacklist(data.LOCAL_PATH / "data" / "blacklisted.txt")
    competitors = read_blacklist(data.LOCAL_PATH / "data" / "competitors.txt")
    blacklist = blacklisted | competitors
    special_characters = read_blacklist(data.LOCAL_PATH / "data" / "special_characters.txt")
    return blacklist, special_characters


def get_cached_data() -> dict[str, list[str]]:
    cached_data = read_cached_data(data.LOCAL_PATH / "data" / "cached_data_165k_squ_home.pkl")
    return cached_data


BLACKLIST, SPECIAL_CHARACTERS = get_entire_blacklist()
CACHED_DATA_SQU = get_cached_data()


def read_vendor_ids() -> dict[int, str]:
    file_path = data.LOCAL_PATH / "data" / "vendor_ids.json"
    with open(file_path, "r") as config_file:
        vendor_ids_data = json.load(config_file)
    vendor_ids = parse_obj_as(list[VendorIDMapping], vendor_ids_data)
    return {item.vendor_id: item.global_vendor_id for item in vendor_ids}


def get_global_vendor_id(vendor_ids: dict[int, str], vendor_id: int) -> str:
    if vendor_id not in vendor_ids:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail={"CODE": "GLOBAL_VENDOR_ID_MAPPING_NOT_FOUND"}
        )
    return vendor_ids[vendor_id]


def pre_embedding_squ_home_get_query(data: Dict[str, Any], **_: Any) -> str:
    return data.get("messages")[-1]["content"]


# Worker class for Semantic Cache
class SemanticCacheWorker:
    def __init__(self, semantic_cache_model_name: str, version="v01"):
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        LOG.debug(f"Use model '{semantic_cache_model_name}' for semantic cache")
        self.encoder = SBERT(semantic_cache_model_name)
        self.number_of_dim = len(self.encoder.to_embeddings("Hello, world."))
        self.configs = Config(similarity_threshold=0.95)
        self.data_manager = self.init_data_manager()
        semantic_cache.init(
            pre_embedding_func=pre_embedding_squ_home_get_query,
            embedding_func=self.encoder.to_embeddings,
            data_manager=self.data_manager,
            similarity_evaluation=SearchDistanceEvaluation(),
            config=self.configs,
        )
        semantic_cache.set_openai_key()

        self.prompt = data.get_system_message(
            data.LOCAL_PATH / "prompts" / f"smart_semantic_search_home_prompt_{version}.txt"
        )
        self.config = data.get_config(data.LOCAL_PATH / "configs" / f"smart_semantic_search_home_config_{version}.json")

    def init_data_manager(self):
        """
        Initializes the data manager with two bases: CacheBase and VectorBase.

        CacheBase: This is where the cache queries are stored. All tables associated
                   with this base have a prefix "gptcache". It uses a PostgreSQL database
                   for storage.

        VectorBase: This is where the actual embeddings are stored. In this case, the
                    embeddings are stored in the same PostgreSQL database. The embeddings
                    are vectors with a specified number of dimensions and are retrieved
                    based on the inner product similarity.

        Returns:
            DataManager: An instance of the data manager with the initialized bases.
        """
        db_config = ConfigManager.load_configuration().storage.postgres

        pg_url = pydantic.parse_obj_as(
            PostgresDsn,
            f"postgresql://{db_config.user}:{db_config.password}"
            f"@{db_config.host}:{db_config.port}/{db_config.database}",
        )

        from sqlalchemy import create_engine as __create_engine

        def new_create_engine(*args, **kwargs):
            # https://www.postgresql.org/docs/16/libpq-connect.html#LIBPQ-KEEPALIVES
            (connect_args := kwargs.pop("connect_args", {})).update(
                {"keepalives": 1, "keepalives_idle": 10, "keepalives_interval": 5, "keepalives_count": 5}
            )
            return __create_engine(*args, pool_pre_ping=True, connect_args=connect_args, **kwargs)

        # Adjust Postgres connection arguments
        with mock.patch("sqlalchemy.create_engine", new=new_create_engine):
            # Don't write cache access reports to the database,
            # otherwise every cache access will be stored.
            with mock.patch(
                "gptcache.manager.data_manager.SSDataManager.report_cache", new=lambda *args, **kwargs: None
            ):
                cache_base = CacheBase("postgresql", sql_url=pg_url)
                vector_base = VectorBase(
                    "pgvector",
                    url=pg_url,
                    dimension=self.number_of_dim,
                    top_k=1,
                    index_params={"index_type": "inner_product", "params": {"lists": 100, "probes": 100}},
                )
                return get_data_manager(cache_base, vector_base)

    @classmethod
    def preprocess(cls, content: str):
        return [
            {"role": "user", "content": content},
        ]

    async def inference(self, messages):
        messages = [{"role": "system", "content": self.prompt}] + messages

        return await cache_openai.ChatCompletion.acreate(
            model=self.config.engine,
            messages=messages,
            max_tokens=self.config.max_tokens,
        )

    @classmethod
    def postprocess(cls, response: dict):
        content = response["choices"][0]["message"]["content"]
        return content

    async def work(self, content: str):
        data = self.preprocess(content=content)
        response = await self.inference(data)
        answer = self.postprocess(response)
        return answer


class Conversation:
    prompt: list[BaseMessage]
    number_of_words: int = 24  # Maximum number of words in a chunk
    _chat: ChatOpenAI

    def __init__(self, config: ChatConfig, prompt: str | SystemMessage | list[SystemMessage], openai_api_key: str):
        # By default, it is 6, that leads to API responses longer than 30s, because of retries
        # in case of errors on the OpenAI side. For example, token's RateLimitError.
        max_retries = 3
        self._chat = ChatOpenAI(
            temperature=config.temperature,
            openai_api_key=openai_api_key,
            model_name=config.engine,
            max_tokens=config.max_tokens,
            max_retries=max_retries,
        )
        if isinstance(prompt, str):
            self.prompt = [SystemMessage(content=prompt)]
        elif isinstance(prompt, SystemMessage):
            self.prompt = [prompt]
        else:
            self.prompt = prompt

    @newrelic.agent.function_trace()
    async def _generate(self, messages: list[BaseMessage]) -> dict:
        messages = self.prompt + messages
        message_dicts, params = self._chat._create_message_dicts(messages, None)

        return await langchain_openai.acompletion_with_retry(self._chat, messages=message_dicts, **params)

    @newrelic.agent.function_trace()
    async def _generate_stream(self, messages: list[BaseMessage]) -> AsyncIterable[dict]:
        messages = self.prompt + messages
        message_dicts, params = self._chat._create_message_dicts(messages, None)
        params["stream"] = True

        return await langchain_openai.acompletion_with_retry(self._chat, messages=message_dicts, **params)

    @newrelic.agent.function_trace()
    async def generate(self, messages: Iterable[BaseMessage]) -> BaseMessage:
        response = await self._generate(list(messages))

        chat_response = self._chat._create_chat_result(response)
        generation: ChatGeneration = chat_response.generations[0]

        return generation.message

    async def generate_stream(self, messages: Iterable[BaseMessage]) -> AsyncIterable[tuple[str, str]]:
        # Send messages to generate a response stream
        response = await self._generate_stream(list(messages))

        buffer = ""  # Holds characters until a word or sentence is completed
        word_buffer = []  # Holds words until they are processed
        reason = None

        # Process each item in the response asynchronously
        async for i in response:
            # Get the reason why generation finished (e.g., stop token reached)
            reason = i["choices"][0]["finish_reason"]

            content = i["choices"][0]["delta"].get("content", " ")

            buffer += content

            # If a word or sentence is completed, process the buffer
            if buffer and buffer[-1] in [" ", "?"]:
                # Split the buffer into words and add to word_buffer
                word_buffer.extend(buffer.split())
                buffer = ""

                # Begin integration:
                # Process the words in word_buffer, grouping them into chunks
                temp_chunk = []  # Holds words for the current chunk
                special_chunk_flag = False  # Indicates if a special chunk($$$ items_to_purhase_for_recipe...$$$) is
                # being processed
                chunks = []  # Holds all the chunks

                # Process each word in word_buffer
                for word in word_buffer:
                    # If word starts a special chunk, and we're not yet in a special chunk
                    if "$$$" in word and not special_chunk_flag:
                        if temp_chunk:
                            # If temp_chunk has words, add them as a chunk
                            chunks.append(" ".join(temp_chunk))
                            temp_chunk = []
                        # Start a new special chunk
                        temp_chunk.append(word)
                        special_chunk_flag = True
                    # If word ends a special chunk, and we're in a special chunk
                    elif "$$$" in word and special_chunk_flag:
                        # End the special chunk
                        temp_chunk.append(word)
                        # Exclude the last word (which is "$$$") from this chunk
                        chunks.append(" ".join(temp_chunk[:-1]))
                        chunks.append(temp_chunk[-1])
                        temp_chunk = []
                        special_chunk_flag = False
                    # If temp_chunk has less than 24 words, add the word to it
                    elif len(temp_chunk) < self.number_of_words:
                        temp_chunk.append(word)
                    # If temp_chunk has 24 words, add it as a chunk and start a new one
                    else:
                        chunks.append(" ".join(temp_chunk))
                        temp_chunk = [word]

                # If temp_chunk has words, add them as a chunk
                if temp_chunk:
                    chunks.append(" ".join(temp_chunk))

                # Process each chunk
                for chunk in chunks:
                    chunk += " "
                    # Check the chunk for moderation issues or keywords
                    if await check_moderation_output(chunk) or check_for_keywords_output(chunk):
                        # If issues are found, raise an exception
                        raise HTTPException(status_code=400, detail="Malicious stream detected")
                    else:
                        # If no issues are found, yield the chunk and reason
                        yield chunk, reason

                # Reset word_buffer for the next iteration
                word_buffer = []

            # If words are left in word_buffer, yield them as a final chunk
        if word_buffer:
            yield " ".join(word_buffer), reason


@cache
def for_intent_check(openai_api_key: str):
    prompt = data.get_system_message(data.LOCAL_PATH / "prompts" / "mal_intent_v01.txt")
    config = data.get_config(data.LOCAL_PATH / "configs" / "mal_intent_v01.json")

    return Conversation(config, prompt, openai_api_key=openai_api_key)


@cache
def for_recipe(version: str, openai_api_key: str):
    prompt = data.get_system_message(data.LOCAL_PATH / "prompts" / f"recipes_prompt_{version}.txt")
    config = data.get_config(data.LOCAL_PATH / "configs" / f"recipes_config_{version}.json")

    return Conversation(config, prompt, openai_api_key=openai_api_key)


@newrelic.agent.function_trace()
def for_recipe_missing_items(basket_items: list[str]):
    var_basket_items = "\n".join(["- " + i.strip() for i in basket_items])
    prompt = data.get_system_message(
        data.LOCAL_PATH / "prompts" / "missing_basket_items_recipe_prompt_v01.txt",
        {"var_basket_items": var_basket_items},
    )
    config = data.get_config(data.LOCAL_PATH / "configs" / "missing_basket_items_recipe_config_v01.json")

    return Conversation(config, prompt)


def for_support_chatbot(version: str, openai_api_key: str):
    support_output_parser = PydanticOutputParser(pydantic_object=UserIntents)

    format_instructions = support_output_parser.get_format_instructions()

    support_output_parser = OutputFixingParser.from_llm(
        parser=support_output_parser, llm=ChatOpenAI(openai_api_key=openai_api_key)
    )

    prompt = data.get_system_message(
        data.LOCAL_PATH / "prompts" / f"support_prompt_{version}.txt",
        vars_to_change={"var_format_instructions": format_instructions},
    )

    config = data.get_config(data.LOCAL_PATH / "configs" / f"support_config_{version}.json")
    return Conversation(config, prompt, openai_api_key=openai_api_key), support_output_parser


@newrelic.agent.function_trace()
def for_smart_semantic_search(version: str, openai_api_key: str):
    prompt = data.get_system_message(data.LOCAL_PATH / "prompts" / f"smart_semantic_search_prompt_{version}.txt")
    chat_config = data.get_config(data.LOCAL_PATH / "configs" / f"smart_semantic_search_config_{version}.json")
    return Conversation(chat_config, prompt, openai_api_key=openai_api_key)


@cache
def conversation_for_smart_semantic_search_home(version: str, openai_api_key: str):
    prompt = data.get_system_message(data.LOCAL_PATH / "prompts" / f"smart_semantic_search_home_prompt_{version}.txt")
    chat_config = data.get_config(data.LOCAL_PATH / "configs" / f"smart_semantic_search_home_config_{version}.json")
    return Conversation(chat_config, prompt, openai_api_key=openai_api_key)


def for_search_recommendations(version: str, openai_api_key: str):
    prompt = data.get_system_message(data.LOCAL_PATH / "prompts" / f"search_recommendation_prompt_{version}.txt")
    chat_config = data.get_config(data.LOCAL_PATH / "configs" / f"search_recommendation_config_{version}.json")
    return Conversation(chat_config, prompt, openai_api_key=openai_api_key)


@newrelic.agent.function_trace()
async def for_conversational_semantic_search(version: str, dialog: list):
    prompt = data.get_system_message(
        data.LOCAL_PATH / "prompts" / f"conversational_semantic_search_prompt_{version}.txt"
    )
    config = data.get_config(data.LOCAL_PATH / "configs" / f"conversational_semantic_search_config_{version}.json")

    converted_dialog = [{"role": "system", "content": prompt}]
    for m in dialog:
        if m.role == "user":
            converted_dialog += [{"role": "user", "content": m.content}]
        else:
            converted_dialog += [{"role": "system", "content": m.content}]

    response = await openai.ChatCompletion.acreate(
        model=config.engine,
        messages=converted_dialog,
        functions=config.functions,
        function_call=config.function_call,
    )

    message = response["choices"][0]["message"]

    return message


async def check_malicious_intent(messages: Iterable[BaseMessage], openai_api_key: str) -> bool:
    conv = for_intent_check(openai_api_key)

    messages = list(messages) + [
        HumanMessage(
            content="Respond TRUE if the user query above is related ONLY TO FOOD AND GROCERY Themes, "
            "and RESPOND FALSE if it is not and/or has malicious intent, "
            "or anything related to politics, religion, terrorism, blasphemy, crimes, or similar."
        ),
        AIMessage(content="Response:"),
    ]

    ai_response = await conv.generate(messages)

    return "true" not in ai_response.content.lower()


def extract_text_in_curly_brackets(text):
    pattern = r"\{(.*?)\}"
    matches = re.findall(pattern, text, re.DOTALL)
    return matches


def remove_newlines(match):
    return match.group().replace("\n", " ")


def extract_and_parse_json(data: str) -> dict:
    prepared_data = data.replace("\n", "").replace("\r", "").replace("\t", "").replace("```json", "").replace("```", "")

    try:
        return json.loads(prepared_data)
    except JSONDecodeError:
        add_transaction_attrs((("prepared_data", prepared_data), ("initial_data", data)))
        LOG.error(f"Failed to parse data as JSON: {prepared_data}")
        raise


def extract_and_parse_for_support_chatbot(data: str, parser) -> dict:
    try:
        return dict(parser.parse(data))
    except:
        add_transaction_attr("data", data)
        raise


def extract_comparison_sign(condition: str):
    match = re.search(r"([<>]=?|==)(\s*\d+)", condition)
    if match:
        comparison_sign = match.group(1)
        value = float(match.group(2))
        return comparison_sign, value
    else:
        return None, None


@newrelic.agent.function_trace()
async def get_items_embeddings(items: list[str]) -> list[np.ndarray]:
    runner = text_embeddings.get_runner()
    embeddings = await runner.async_run(items)
    return embeddings


@newrelic.agent.function_trace()
def check_moderation(dialog, openai_api_key: str):
    entire_dialog = ""
    for message in dialog:
        entire_dialog += "- " + message.content + "\n"

    moderation_chain = OpenAIModerationChain(error=True, openai_api_key=openai_api_key)

    error = moderation_chain.run(entire_dialog)

    return error


@newrelic.agent.function_trace()
async def check_moderation_output(text):
    response = await openai.Moderation.acreate(input=text)
    output = response["results"][0]["flagged"]

    return output


@newrelic.agent.function_trace()
def check_for_keywords_output(text):
    input_words = re.split("\\W", text)

    # Check if any word in input_string is in the blacklist
    for word in input_words:
        if word.lower() in BLACKLIST:
            return True

    # If no blacklisted word is found, return False
    return False


@newrelic.agent.function_trace()
def check_for_keywords(dialog):
    entire_dialog = ""
    for message in dialog:
        entire_dialog += message.content + "\n"

    # Splitting the input_string into individual words using regular expression
    input_words = re.split("\\W", entire_dialog)

    # Check if any word in input_string is in the blacklist
    for word in input_words:
        if word.lower() in BLACKLIST:
            return True

    # If no blacklisted word is found, return False
    return False


@newrelic.agent.function_trace()
def check_for_adversarial_attack(dialog):
    entire_dialog = ""
    for message in dialog:
        entire_dialog += message.content + "\n"

    for c in SPECIAL_CHARACTERS:
        if c in entire_dialog:
            return True

    return False


def get_unique_messages(messages):
    unique_messages = []
    for msg in messages:
        if msg.content not in unique_messages:
            unique_messages.append(msg)

    return unique_messages


async def rerank_recommendations(
    requested_item: str, recommended_items: list[PurchaseItemRecommendation], threshold: float = -2.0
) -> list[PurchaseItemRecommendation]:
    if not any(item for item in recommended_items):
        return recommended_items

    input_pairs: list[Tuple[str, str]] = [(requested_item, item.item_name) for item in recommended_items]
    # Get scores for each pair
    all_scores: list[float] = await cross_encoder.get_runner().predict.async_run(input_pairs)

    recommendations_with_scores: list[Tuple[PurchaseItemRecommendation, float]] = list(
        zip(recommended_items, all_scores)
    )

    # Filter recommendations based on the threshold, score can be 7, 11, 5 if similar, and -11, -8 if not similar
    reranked_recommendations_with_score: list[Tuple[PurchaseItemRecommendation, float]] = sorted(
        [item for item in recommendations_with_scores if item[1] >= threshold], key=lambda x: x[1], reverse=True
    )
    return [item[0] for item in reranked_recommendations_with_score]


def convert_to_json(query: str, input_string: str) -> dict:
    valid_categories = set(item.value for item in VerticalsSearch)
    input_set = set(item.strip() for item in input_string.split(","))
    categories = list(input_set & valid_categories)

    if not categories:
        categories = ["food"]

    return {"query": query, "category": categories}


def get_value_from_json_squ(query):
    query = query.lower().strip()
    result = CACHED_DATA_SQU.get(query, ["food"])
    return ", ".join(result)
