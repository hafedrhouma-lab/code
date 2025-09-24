from abc import ABCMeta
from typing import ClassVar

from abstract_ranking.base_logic import BaseLogic
from vendor_ranking import SERVICE_NAME


class VendorBaseLogic(BaseLogic["VendorList"], metaclass=ABCMeta):
    NAME: ClassVar[str] = SERVICE_NAME
    SERVICE_NAME: ClassVar[str] = SERVICE_NAME
