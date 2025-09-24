import sys
from pathlib import Path
root_dir = Path(__file__).parent
[sys.path.insert(0, str(root_dir := root_dir.parent)) for _ in range(3)]
import os
import mlflow
import structlog
import yaml
from dotenv import load_dotenv
import pandas as pd
from ese.sim_score import K2K
from tutils.db_utils import BigQuery


load_dotenv(override=True)
LOG: structlog.stdlib.BoundLogger = structlog.get_logger()


class ModelPredictor:

    def __init__(self):
        exp_name = os.environ['MLFLOW_EXPERIMENT_NAME']
        self.MODEL_NAME = exp_name
        LOG.info(f"Tracking uri: {mlflow.get_tracking_uri()}")
        LOG.info(f"Experiment id: {exp_name}")

        with open('config.yaml') as file:
            self.config = yaml.load(file, Loader=yaml.FullLoader)
        LOG.info(f"Config: {self.config}")

        self.k2k = K2K()

        self.tags = list(set(self.config['tags']))
        self.sim_threshold = self.config['SimilarityThreshold']

        tags_per_items = self.config['tags_per_item']

        if tags_per_items > 0:
            self.tags_per_items = tags_per_items
        else:
            self.tags_per_items = len(self.tags)

    def _get_unique_chain_list(self):
        bq = BigQuery()
        query = '''
            SELECT distinct ei.chain_id
            FROM `tlb-data-prod.data_platform.ese_items` ei
            INNER JOIN `tlb-data-prod.data_platform.dim_chain` dc
            ON ei.chain_id = dc.chain_id
            WHERE dc.chain_status < 4
        '''
        df = bq.read(query)
        chain_list = df['chain_id'].tolist()
        return chain_list

    def _batch_chain_list(self, chain_list, batch_size=20):
        chain_batches = [chain_list[i:i + batch_size] for i in range(0, len(chain_list), batch_size)]
        return chain_batches

    def _retrieve_chain_batch_items(self, chain_batch):
        query = f'''SELECT 
                parent_item_id,
                parent_item_name_en,
                parent_item_description_en,
                parent_item_category_name,
            FROM `tlb-data-prod.data_platform.ese_items`
            WHERE chain_id IN ({','.join(map(str, chain_batch))})
        '''
        bq = BigQuery()
        df_items_batch = bq.read(query)
        return df_items_batch

    def _get_relevant_tags(self, df_items_batch):
        df_items_batch['tokens'] = df_items_batch['parent_item_category_name'] + ' ' + df_items_batch[
            'parent_item_name_en'] + ' ' + \
                                   df_items_batch['parent_item_description_en']

        items_multiplied = df_items_batch['tokens'].repeat(len(self.tags)).tolist()
        df = self.k2k(
            entity1=self.tags * len(df_items_batch),
            entity2=items_multiplied
        )

        # Filter based on the hyperparameters
        filtered_df = df[df['similarity'] >= self.sim_threshold]
        LOG.info(f"Num of filtered pairs: {len(filtered_df)}")

        # Sort by keyword2 and similarity, and then get the top 3 keyword1 for each keyword2
        top_tags = (filtered_df.sort_values(['keyword2', 'similarity'], ascending=[True, False])
                    .groupby('keyword2')
                    .head(self.tags_per_items))
        top_tags = top_tags.rename(columns={'keyword1': 'tag'})
        LOG.info(f"Num of top tags: {len(top_tags)}")

        df_tags = pd.merge(
            left=df_items_batch,
            right=top_tags,
            left_on=['tokens'],
            right_on=['keyword2'],
            how='left'
        )
        LOG.info(f"Num of items with tags: {len(df_tags)}")

        df_tags = df_tags[['parent_item_id', 'parent_item_name_en', 'tag', 'similarity']].dropna().reset_index(
            drop=True)
        LOG.info(f"Num of items with tags after dropping NAs: {len(df_tags)}")

        return df_tags

    @staticmethod
    def push_to_bq(df_output):
        bq = BigQuery()
        bq.write(df_output, 'tlb-data-dev.item_tagging.ese_item_tags', if_exist_action='append')

    def __call__(self):

        chain_list = self._get_unique_chain_list()
        chain_batches = self._batch_chain_list(chain_list, batch_size=2)
        LOG.info(f"Num of chains: {len(chain_list)}")
        LOG.info(f"Num of batches: {len(chain_batches)}")
        LOG.info(f"Size of each batch: {len(chain_batches[0])}")

        for i, chain_batch in enumerate(chain_batches):
            if i < 10:
                LOG.info(f"##### Processing [batch: {i}] #####")
                df_items_batch = self._retrieve_chain_batch_items(chain_batch)
                LOG.info(f"Num of items: {len(df_items_batch)}")
                df_tags = self._get_relevant_tags(df_items_batch)
                self.push_to_bq(df_tags)
                LOG.info(f"Successfully saved [batch: {i}]")

if __name__ == '__main__':
    ese_tagger = ModelPredictor()
    ese_tagger()
