from abc import ABC, abstractmethod

from ace.model_log import LogEntry
from nba import SERVICE_NAME
from nba.hero_banner.agents import AgentsManager
from nba.input import HeroBannersRequest, HeroBannersResponse


class BaseLogic(ABC):
    NAME: str = SERVICE_NAME
    MODEL_TAG: str = None

    def __init__(self, request: HeroBannersRequest, agents_manager: AgentsManager, exec_log: LogEntry, **kwargs):
        self.request = request
        self.agents_manager = agents_manager
        self.exec_log = exec_log
        exec_log.service_name = self.NAME
        exec_log.model_tag = self.MODEL_TAG

    @abstractmethod
    async def prepare_features(self) -> None:
        raise NotImplementedError()

    @abstractmethod
    async def predict(self) -> HeroBannersResponse:
        raise NotImplementedError()
