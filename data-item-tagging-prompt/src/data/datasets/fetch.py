"""Definition of FetchItemData Class"""
import os

from dataclasses import dataclass
import pandas as pd
import jinja2
from tutils.db_utils import BigQuery

from src.utils.log import Logger

__here__ = os.path.dirname(os.path.abspath(__file__))

template_loader = jinja2.FileSystemLoader(searchpath=os.path.join(__here__, "../queries"))
template_env = jinja2.Environment(loader=template_loader)

db = BigQuery()


@dataclass
class FetchItem:
    """Class to query item data (decription, title, ingredients)
    """
    query_name: str

    def __post_init__(self):
        self._logger = Logger(self.__class__.__name__).get_logger()
        self.query_template = {
            "nfv_items": "nfv/item_information.sql.j2",
        }

    def get(self: str) -> pd.DataFrame:
        """query food items data
        :returns:
            Dataframe holding food data descriptions
        """
        template = self.query_template[self.query_name]
        query = template_env.get_template(template).render()

        result_df = db.read(query)
        self._logger.info(
            f'Retrieved {result_df.shape[0]} Items using query: {self.get_template_path(template)}'
        )

        return result_df

    @staticmethod
    def get_template_path(sub_path: str) -> str:
        """Generate the full path for a given template name"""
        return os.path.join(__here__, "../queries", sub_path)
