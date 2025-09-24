"""Definition of FetchSessionsSearch Class"""
import os
import pandas as pd
from google.cloud import bigquery
import jinja2

from src.utils.log import Logger

__here__ = os.path.dirname(os.path.abspath(__file__))

template_loader = jinja2.FileSystemLoader(
    searchpath=os.path.join(__here__, "queries/aggregates")
)
template_env = jinja2.Environment(loader=template_loader)

client = bigquery.Client()


class FetchQueriesPerformances:
    """Class to query performances for searches
    aggregated by user selected localisation
    """

    query_template = {
        "search_query": "agg_session_level_performances.sql.j2",
        "country_metric": "agg_country_search_performances.sql.j2",
        "in_vendor_search": "agg_in_vendor_search_performances.sql.j2",
    }

    def __init__(self, type_data: str):
        self.type_data = type_data
        self._logger = Logger(self.__class__.__name__).get_logger()

    def get(self, user_selected_values: dict) -> pd.DataFrame:
        """get query result given user's param
        :returns:
            Dataframe holding query result
        """
        query = template_env.get_template(
            self.query_template[self.type_data]
        ).render(
            user_selected_values=user_selected_values
        )

        query_job = client.query(query)

        self._logger.info(
            f"Fetched Query Aggregate: "
            f"From {user_selected_values.get('date_start')} "
            f"For {user_selected_values.get('country')} "
            f"with nb of rows: {len(query_job.result().to_dataframe())}"
        )

        return query_job.result().to_dataframe(
            create_bqstorage_client=True
        )
