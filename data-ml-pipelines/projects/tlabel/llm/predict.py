import sys
from pathlib import Path
root_dir = Path(__file__).parent
[sys.path.insert(0, str(root_dir := root_dir.parent)) for _ in range(3)]
import pandas as pd
import pandas_gbq
import os
import mlflow
import structlog
from openai import OpenAI
from dotenv import load_dotenv
from jinja2 import Template
from base.v1.db_utils import BigQuery
import json
import yaml
from projects.tlabel.llm.output_formats import GeographicalResponse
from google.api_core.exceptions import NotFound
from google.cloud import secretmanager
import argparse


load_dotenv(override=True)
LOG: structlog.stdlib.BoundLogger = structlog.get_logger()
CURRENT_DIR = Path(__file__).parent
PROMPTS_DIR = CURRENT_DIR / "prompts"
DATA_DIR = CURRENT_DIR / "data"
SQL_DIR = CURRENT_DIR / "sql"
TAGS_DIR = CURRENT_DIR / "tags"
LOG.info(f"Current directory: {CURRENT_DIR}")
LOG.info(f"Prompts directory: {PROMPTS_DIR}")
LOG.info(f"Data directory: {DATA_DIR}")
#PROJECT_ID = 'tlb-data-prod'
PROJECT_ID = os.getenv("PROJECT_ID")


class ModelPredictor:
    def __init__(self, category, country_code, overwrite=False, batch_size=50):
        """
        Initialize the model predictor

        Args:
            category (str): Category of the tags. Allowed values: ['geographical', 'cuisine', 'dish']
            country_code (str): Country code. Allowed values: ['AE', 'KW', 'OM', 'BH', 'QA', 'JO', 'EG', IQ]
            overwrite (bool): Default is False. If True, overwrite all chains in `dim_chain_tags` table
            batch_size (int): Default is 50. Batch size for processing chains
        """
        exp_name = os.environ['MLFLOW_EXPERIMENT_NAME']
        self.MODEL_NAME = exp_name
        LOG.info(f"Tracking uri: {mlflow.get_tracking_uri()}")
        LOG.info(f"Experiment id: {exp_name}")
        self.llm_model = "gpt-4o-mini"
        self.dim_chain_tags_table = f"{PROJECT_ID}.data_tlabel.dim_chain_tags"
        self.dim_tags_table = f"{PROJECT_ID}.data_tlabel.dim_tags"
        self.overwrite = overwrite
        self.indexed_chains = {}
        self.category = category
        self.country_code = country_code
        self.tags_map = {}
        self.batch_size = batch_size
        self.system_prompt = ''
        self.proxy_client = None
        self.initialize()

        LOG.info(f"LLM Model: {self.llm_model}")

        self.bq_agent = BigQuery()

    def initialize(self, proxy = 1):
        self.tags_map = self.get_relevant_tags(country_code=self.country_code, category=self.category)
        if len(self.tags_map) == 0:
            raise ValueError(f"No tags found for country {self.country_code} - category: {self.category}")
        self.system_prompt = self.get_system_message()
        token = self.get_secret_from_gcp()
        LOG.info(f'Token ending with "{token[-4:]}" in use')
        if proxy:
            self.proxy_client = OpenAI(
                api_key=token,
                base_url="https://data.talabat.com/api/public/genai")
        else:
            self.proxy_client = OpenAI(
                api_key=token)


    @staticmethod
    def get_secret_from_gcp():
        client = secretmanager.SecretManagerServiceClient()
        name = f"projects/{PROJECT_ID}/secrets/tlabel_openai/versions/latest"
        print("PROJECT_ID is ", PROJECT_ID)
        response = client.access_secret_version(request={"name": name})
        return response.payload.data.decode("UTF-8")

    def get_system_message(self) -> str:
        with open(os.path.join(PROMPTS_DIR, f'{self.category}_prompt.txt')) as f:
            system_message = f.read()

        tags = [f"- {tag}" for tag in self.tags_map.keys()]
        system_message += '\n'.join(tags)

        return system_message

    @staticmethod
    def get_relevant_tags(country_code, category) -> dict:
        # Read country yaml and get the corresponding category
        tags_map = {}

        with open(os.path.join(TAGS_DIR, f"tags_mapping.yaml")) as file:
            all_tags = yaml.load(file, Loader=yaml.FullLoader)
        category_tags = all_tags[category]

        for tag, details in category_tags.items():
            if country_code in details['country_code']:
                tags_map[tag] = details['tag_id']
        return tags_map

    def get_country_chains_menu(self, batch_list: str) -> pd.DataFrame:

        sql_file = "chains_menu.sql"

        with open(os.path.join(SQL_DIR, sql_file)) as file:
            template = Template(file.read())
        query = template.render(
            country_code=self.country_code,
            batch_list=batch_list
        )
        df = self.bq_agent.read(query)

        return df

    @staticmethod
    def preprocess_inference_data(df):
        df['parent_orders_percentage'] = df['parent_orders_percentage'].replace(0, "N/A")
        df['menu_items'] = df[
            ['parent_item_category_name', 'parent_item_name_en', 'parent_item_description_en', 'parent_orders_percentage']].apply(
            lambda x: ' -- '.join(x.dropna().astype(str)) + '%', axis=1)
        df = df[['chain_id', 'chain_name', 'menu_items']]
        df = df.groupby(['chain_id', 'chain_name'], as_index=False).head(50).reset_index(drop=True)
        return df

    def get_llm_response(self, menu):
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": str(menu)}
        ]
        try:
            print(f'\n\n using Daniiar proxy ({self.proxy_client}, token = {self.get_secret_from_gcp()}) \n\n')
            response = self.proxy_client.beta.chat.completions.parse(
                model=self.llm_model,
                messages=messages,
                response_format=GeographicalResponse
            )
        except:
            self.initialize(proxy = 0)
            print(f'\n\n using openai creds ({self.proxy_client}, token = {self.get_secret_from_gcp()}) directly \n\n')
            response = self.proxy_client.beta.chat.completions.parse(
                model=self.llm_model,
                messages=messages,
                response_format=GeographicalResponse
            )
    
        
        # Parse the content if it’s a JSON string
        content = response.choices[0].message.content
        if isinstance(content, str):
            try:
                content = json.loads(content)  # Parse the JSON content
            except json.JSONDecodeError as e:
                LOG.error(f"Failed to parse response as JSON: {e}")
                return []  # Return an empty list or handle the error as needed

        return content['chains']

    @staticmethod
    def response_to_df(response):
        df = pd.DataFrame(response)
        df_long = pd.melt(
            df, id_vars=["chain_id"],
            value_vars=["tag_1"],
            var_name="tag_type",
            value_name="tag_id"
        ).drop(columns=["tag_type"])
        #
        df_long["explanation"] = pd.melt(
            df, id_vars=["chain_id"],
            value_vars=["explanation_tag_1"],
            var_name="explanation_type",
            value_name="explanation"
        )["explanation"]
        return df_long

    def update_dim_tags(self):
        current_bq_tags_set = self.get_bq_country_category_tags()
        new_tags = {tag_name: tag_if for tag_name, tag_if in self.tags_map.items() if tag_if not in current_bq_tags_set}
        if new_tags:
            LOG.info(f"New tags found: {new_tags}")
            df = pd.DataFrame(new_tags.items(), columns=['tag_name', 'tag_id'])
            df['country_code'] = self.country_code
            df['category'] = self.category
            df['dwh_entry_timestamp'] = pd.Timestamp('today')
            self.bq_agent.write(df=df, table_name=self.dim_tags_table, if_exist_action='append')

    def get_bq_country_category_tags(self) -> set:
        LOG.info(f"Retrieving tags from BQ for country: {self.country_code}")
        sql_file = "country_category_tags.sql"
        with open(os.path.join(SQL_DIR, sql_file)) as file:
            template = Template(file.read())
        query = template.render(
            country=self.country_code,
            category=self.category,
            dim_tags_table=self.dim_tags_table
        )
        try:
            df = self.bq_agent.read(query)
            tags_set = set(df['tag_id'].unique())
        except NotFound:
            tags_set = set()
        except pandas_gbq.exceptions.GenericGBQException as e:
            tags_set = set()

        return tags_set

    def get_chain_list(self) -> int:
        LOG.info(f"Retrieving chain list from BQ for country: {self.country_code}")
        if self.overwrite:
            LOG.info("Overwrite flag is set. Fetching all chains")
            sql_file = "chain_list_all.sql"
        else:
            LOG.info("Fetching new chains only")
            sql_file = "chain_list_new.sql"
        with open(os.path.join(SQL_DIR, sql_file)) as file:
            template = Template(file.read())
        query = template.render(
            country_code=self.country_code,
            category=self.category,
            dim_chain_tags=self.dim_chain_tags_table,
            dim_tags=self.dim_tags_table

        )
        df = self.bq_agent.read(query)
        return df['chain_id'].tolist()

    def delete_existing_records(self, df):
        chain_ids = df['chain_id'].unique().tolist()
        chain_ids_str = ', '.join([f"'{i}'" for i in chain_ids])
        sql_file = "delete_existing_records.sql"
        with open(os.path.join(SQL_DIR, sql_file)) as file:
            template = Template(file.read())
        query = template.render(
            dim_chain_tags=self.dim_chain_tags_table,
            chain_ids=chain_ids_str,
            tag_prefix=self.category.lower()[0:3]
        )
        try:
            self.bq_agent.send_dml(query)
        except NotFound:
            pass

    def construct_batches(self, chain_list):
        n_batches = len(chain_list) // self.batch_size
        list_of_batches = [tuple(chain_list[i:i + self.batch_size]) for i in range(0, len(chain_list), self.batch_size)]
        return list_of_batches, n_batches

    def __call__(self):
        LOG.info(f"PROJECT_ID: {PROJECT_ID}")
        chain_list = self.get_chain_list()
        LOG.info(f"Total chains: {len(chain_list)}")
        list_of_batches, n_batches = self.construct_batches(chain_list)
        self.update_dim_tags()

        df_mini_batch = pd.DataFrame()
        for i, batch_list in enumerate(list_of_batches):
            batch_list_str = ", ".join(map(str, batch_list))  # convert tuple to a safe SQL string
            df_items = self.get_country_chains_menu(batch_list_str)
            df_menu = self.preprocess_inference_data(df_items)
            chain_stack = df_menu['chain_id'].unique().tolist()
            LOG.info(f"[Batch {i}/{n_batches}] - chains [total]: {len(batch_list)}")
            LOG.info(f"[Batch {i}/{n_batches}] - chains [active items]: {len(chain_stack)}")

            while chain_stack:
                LOG.info(f"[Batch {i}/{n_batches}] - remaining to process: {len(chain_stack)}")
                processing_chains = []
                for _ in range(5):
                    if chain_stack:
                        processing_chains.append(chain_stack.pop())

                df_menu_batch = df_menu[df_menu['chain_id'].isin(processing_chains)]
                user_query = df_menu_batch.groupby(['chain_id', 'chain_name'])['menu_items'].apply(list).to_dict()
                response = self.get_llm_response(user_query)
                df = self.response_to_df(response)
                df['tag_id'] = df['tag_id'].apply(lambda x: x if x in self.tags_map else '')
                df['tag_id'] = df['tag_id'].map(self.tags_map)
                df['country_code'] = self.country_code
                df['dwh_entry_timestamp'] = pd.Timestamp('today')
                df_mini_batch = pd.concat([df_mini_batch, df], ignore_index=True)
                if len(chain_stack) == 0:
                    if self.overwrite:
                        self.delete_existing_records(df_mini_batch)
                    self.bq_agent.write(
                        df=df_mini_batch, table_name=self.dim_chain_tags_table, if_exist_action='append'
                    )
                    LOG.info(f"[Batch {i}/{n_batches}]: Checkpoint: BQ records updated")
                    df_mini_batch = pd.DataFrame()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Run ModelPredictor with specified parameters.")

    parser.add_argument("--category", type=str, default="geographical", help="Category for prediction")
    parser.add_argument("--country_code", type=str, default="EG", help="Country code")
    parser.add_argument("--overwrite", action='store_true', help="Whether to overwrite existing data")
    parser.add_argument("--batch_size", type=int, default=50, help="Batch size for processing chains")

    args = parser.parse_args()

    LOG.info(f"Arguments passed: {args}")

    modelpredictor = ModelPredictor(
        category=args.category,
        country_code=args.country_code,
        overwrite=args.overwrite,
        # overwrite=True,
        batch_size=args.batch_size
    )
    modelpredictor()
