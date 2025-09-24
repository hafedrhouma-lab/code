import base64
import dataclasses
import random
import typing as t
import zlib
from dataclasses import dataclass
from datetime import datetime

import aioboto3
import newrelic.agent
import structlog
from decouple import config
from newrelic.api.transaction import Transaction
from orjson import orjson

import ace

SQS_QUEUE_URL = config("ACE_LOGS_SQS_QUEUE_URL", default="")
LOGS_SAMPLE_PERCENTAGE = config("ACE_LOGS_SAMPLE_PERCENTAGE", cast=int, default=10)

_session = aioboto3.Session()

logger = structlog.get_logger()


@dataclass
class LogEntry:
    reference_id: str
    service_name: str
    model_tag: t.Optional[str]
    request_timestamp: str
    request: dict  # await request.json()
    metadata: dict
    response: t.Optional[t.Any] = None
    model_input: t.Optional[t.Any] = None
    model_output: t.Optional[t.Any] = None

    def __init__(self, service_name, request: dict) -> None:
        super().__init__()
        self.service_name = service_name
        self.request_timestamp = datetime.now().isoformat()
        self.request = request
        self.metadata = {}

        apm_transaction: t.Optional[Transaction] = newrelic.agent.current_transaction()
        self.reference_id = apm_transaction.guid if apm_transaction else None

    def add_metadata(self, key: str, value: t.Any):
        self.metadata[key] = value


async def log_execution(exec_log: LogEntry) -> None:
    # Sampling, take only LOGS_SAMPLE_PERCENTAGE % of the requests
    if random.randrange(1, 101) > LOGS_SAMPLE_PERCENTAGE:
        logger.debug("Skipping the request from logging")
        return

    await _log_execution(exec_log)


@ace.asyncio.to_thread
@newrelic.agent.function_trace()
def _prepare_for_sqs(exec_log: LogEntry) -> str:
    message = dataclasses.asdict(exec_log)
    json_message = orjson.dumps(message, option=orjson.OPT_NON_STR_KEYS)
    # SNS/SQS can handle only UTF-8, but not arbitrary binary data.
    # So we have to base64 encode it.
    # It makes the message bigger by around 20%, but still dramatically better than sending uncompressed JSON.
    compressed_message_bytes = base64.b64encode(zlib.compress(json_message))
    return compressed_message_bytes.decode("ascii")


@newrelic.agent.background_task(name="dump_model_execution", group="background")
async def _log_execution(exec_log: LogEntry) -> None:
    ace.newrelic.add_transaction_attr("model_tag", exec_log.model_tag)

    compressed_message = await _prepare_for_sqs(exec_log)

    if not SQS_QUEUE_URL:
        logger.warn("SQS for model logs is not configured, skipping")
        return

    async with _session.client("sqs") as sqs:
        await sqs.send_message(
            QueueUrl=SQS_QUEUE_URL,
            MessageBody=compressed_message,
            MessageAttributes={"model_tag": {"DataType": "String", "StringValue": str(exec_log.model_tag)}},
        )
