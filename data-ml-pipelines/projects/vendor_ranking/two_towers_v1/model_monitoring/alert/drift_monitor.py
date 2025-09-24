import os
import json
import structlog
import requests
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

from projects.vendor_ranking.common.two_towers.src.data.datasets.query_loader import QueryLoader
from projects.vendor_ranking.common.two_towers.src.data import get_data_fetcher
from projects.vendor_ranking.common.two_towers.src.utils.misc_utils import get_date_minus_days

load_dotenv(override=True)
LOG = structlog.getLogger(__name__)

CURRENT_DATE = datetime.now().strftime("%Y-%m-%d")


class DriftMonitor:
    TEMPLATES = {
        'metrics_query': 'collect_drifts_metrics.sql.j2',
        'slack_message_template': 'slack_message_template.j2'
    }

    def __init__(self, param_date: str, slack_webhook_url: str, group_name_param: str,
                 template_loader: QueryLoader = None):
        self.param_date = param_date
        self.slack_webhook_url = slack_webhook_url
        self.group_name_param = group_name_param
        self.template_loader = template_loader or QueryLoader(
            template_dir=str(Path(__file__).parent.resolve() / "")
        )

    def fetch_drift_data(self) -> list[dict]:
        drifts_metrics_query = self.template_loader.load_query(
            self.TEMPLATES['metrics_query'],
            param_date=self.param_date
        )
        df_drifts_metrics_query = get_data_fetcher().fetch_data(
            description='Drift Information and Metrics',
            source="sql",
            query=drifts_metrics_query
        )
        return df_drifts_metrics_query.to_dict(orient='records')

    def format_message(self, data: list[dict]) -> str:
        slack_message_template = self.template_loader.load_query(
            self.TEMPLATES['slack_message_template'],
            group_name_param=self.group_name_param,
            data=data
        )
        return slack_message_template

    def send_slack_message(self, message: str) -> None:
        slack_payload = {
            "channel": self.group_name_param,
            "text": message
        }
        slack_payload_json = json.dumps(slack_payload)

        try:
            response = requests.post(
                self.slack_webhook_url,
                data=slack_payload_json,
                headers={'Content-Type': 'application/json'}
            )
            response.raise_for_status()
            LOG.info("Message sent to Slack successfully!")
        except requests.exceptions.RequestException as e:
            LOG.error(f"Failed to send message to Slack. Error: {e}")
            raise RuntimeError(f"Failed to send message to Slack: {e}") from e

    def run(self) -> None:
        try:
            data = self.fetch_drift_data()
            if data:
                slack_message = self.format_message(data)
                self.send_slack_message(slack_message)
            else:
                LOG.info(f"No drift data found for test_date = {self.param_date}.")
        except Exception as e:
            LOG.error(f"An error occurred in DriftMonitor.run: {e}")


def main():
    param_date = get_date_minus_days(CURRENT_DATE, 1)
    slack_webhook_url = os.getenv("SLACK_WEBHOOK_URL")
    group_name_param = os.getenv("SLACK_GROUP_NAME", "ML Ops Team Internal")

    drift_monitor = DriftMonitor(
        param_date,
        slack_webhook_url,
        group_name_param
    )
    drift_monitor.run()


if __name__ == "__main__":
    main()
