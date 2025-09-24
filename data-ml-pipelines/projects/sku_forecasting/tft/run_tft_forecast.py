import structlog
from base.v0 import mlutils
import numpy as np
import pandas as pd
from base.v0.db_utils import BigQuery
from projects.sku_forecasting.common.utils import get_training_data, add_holiday_features, add_time_features
from projects.sku_forecasting.common.query_predict_actual import get_predict_actual_query
from datetime import datetime, timedelta
import mlflow
import os
import warnings
from dotenv import load_dotenv


warnings.simplefilter(action='ignore', category=FutureWarning)
load_dotenv(override=True)
LOG: structlog.stdlib.BoundLogger = structlog.get_logger()


class TftForecast:
    def __init__(self):
        exp_name = os.environ['MLFLOW_EXPERIMENT_NAME']
        self.MODEL_NAME = exp_name
        LOG.info(f"Tracking uri: {mlflow.get_tracking_uri()}")
        LOG.info(f"Experiment id: {exp_name}")
        self.model_input = {}
        self.tft_parameters = None

    def generate_final_data(self, model_prediction, start_date):

        data = []
        tft_model = self.mlflow_model.unwrap_python_model().model
        predicted = np.array(model_prediction.output.prediction)

        for i in range(len(self.tft_parameters['quantiles'])):
            data.append(predicted[:, :, i].flatten())

        cols = [f'q_{q}'.replace('.', '') for q in self.tft_parameters['quantiles']]

        # Actual and Pred
        df_realization = pd.DataFrame(data).T
        df_realization.columns = cols

        # Metadata
        categorical_groups = tft_model.hparams['output_transformer'].groups
        groups_tensor = model_prediction.x['groups']

        category_mapping_dict = {}
        # Create a mapping dictionary for each categorical group

        for category in categorical_groups:
            cat_to_id = tft_model.hparams['embedding_labels'][category]
            category_mapping_dict[f'id_to_{category}'] = {v: k for k, v in cat_to_id.items()}

        df_metadata = pd.DataFrame(groups_tensor, columns=categorical_groups)
        for col in df_metadata.columns:
            df_metadata[col] = df_metadata[col].map(category_mapping_dict[f'id_to_{col}'])

        # Final output
        decoder_length = self.tft_parameters['max_prediction_length']
        df_output = df_metadata.loc[df_metadata.index.repeat(decoder_length)].reset_index(drop=True)

        # Generate the date range for 30 days starting from '2024-03-01'
        dates = pd.date_range(start=start_date, periods=decoder_length, freq='D')
        repeated_dates = np.tile(dates, len(df_output) // decoder_length)
        repeated_dates = pd.DatetimeIndex(repeated_dates)
        df_output.insert(0, 'forecast_date', repeated_dates)

        # Concatenate the metadata and the realization
        df_output = pd.concat([df_output, df_realization], axis=1)
        return df_output

    @staticmethod
    def push_to_bq(df_output):
        # dynamically define the schema
        schema = []
        for col in df_output.columns:
            if col in ['forecast_date']:
                schema.append({'name': col, 'type': 'DATE'})
            elif col.startswith('q'):
                schema.append({'name': col, 'type': 'FLOAT'})
            else:
                schema.append({'name': col, 'type': 'STRING'})
        bq = BigQuery()
        bq.write(
            df_output,
            'data_platform_sku_forecasting.tft_forecast',
            'replace',
            schema=schema
        )

    def write_predict_actual_table(self):
        query = get_predict_actual_query()
        bq = BigQuery()
        df_predict_actual = bq.read(query)
        bq.write(
            df_predict_actual,
            'data_platform_sku_forecasting.tft_forecast_dashboard',
            'replace'
        )

    def prepare_encoder_data(self, start_date):
        LOG.info("Preparing encoder data...")
        # Get the end date for the encoder data
        encoder_end_date = (datetime.strptime(start_date,'%Y-%m-%d') - timedelta(days=1)).strftime('%Y-%m-%d')

        encoder_data = get_training_data(
            end_date=encoder_end_date,
            history_duration=self.tft_parameters['max_encoder_length'] - 1,
            unique_categories=self.unique_categories,
        )
        LOG.info("Encoder data:")
        LOG.info(encoder_data.order_date.min())
        LOG.info(encoder_data.order_date.max())
        LOG.info(f"{encoder_data.shape}")
        LOG.info("Done preparing encoder data!")

        return encoder_data

    def prepare_decoder_data(self, encoder_data):

        LOG.info("Preparing decoder data...")
        main_columns = [col for col in encoder_data.columns if 'holiday' not in col.lower()]
        decoder_data = encoder_data[main_columns]
        decoder_data = (decoder_data.groupby(['sku', 'warehouse_id'])
                        .tail(self.tft_parameters['max_prediction_length']).reset_index(drop=True)
                        .groupby(['sku', 'warehouse_id'])
                        .apply(self.shift_dates)).reset_index(drop=True)

        LOG.info("Appending time features to decoder data...")
        decoder_data = add_time_features(decoder_data).reset_index(drop=True)

        LOG.info("Appending holiday features to decoder data...")
        decoder_data = add_holiday_features(decoder_data)

        LOG.info("\nDecoder data:")
        LOG.info(decoder_data.order_date.min())
        LOG.info(decoder_data.order_date.max())
        LOG.info(decoder_data.shape)
        LOG.info("Done preparing decoder data!")

        return decoder_data

    def construct_inference_data(self, encoder_data, decoder_data):
        inference_data = pd.concat([encoder_data, decoder_data], ignore_index=True)
        inference_data['time_idx'] = inference_data['time_idx'].astype(int)
        inference_data = self.special_tft_preprocessing(inference_data)

        for column in inference_data.select_dtypes(include=[np.number]).columns:
            if column != 'time_idx':
                inference_data[column] = inference_data[column].astype(np.float32)

        LOG.info("\nInference data:")
        LOG.info(inference_data.order_date.min())
        LOG.info(inference_data.order_date.max())
        LOG.info(inference_data.shape)
        LOG.info("Done preparing decoder data!")
        LOG.info("Done constructing inference data!")
        return inference_data

    def special_tft_preprocessing(self, data):
        data[self.tft_parameters['special_days']] = data[self.tft_parameters['special_days']].apply(
            lambda x: x.map({0: "-", 1: x.name})
        ).astype("category")
        data['parent_category'] = data['parent_category'].fillna('no_parent_category').astype("category")
        data['master_category'] = data['master_category'].fillna('no_master_category').astype("category")
        return data

    def shift_dates(self, df):
        df['order_date'] = df['order_date'] + pd.DateOffset(days=30)
        df['time_idx'] = range(
            df["time_idx"].max() + 1,
            df["time_idx"].max() + 1 + self.tft_parameters['max_prediction_length'],
            )
        df['sales'] = 0
        return df

    @staticmethod
    def assert_date(start_date):
        if not start_date:
            start_date = datetime.now().strftime('%Y-%m-%d')
            LOG.info(f"Start date not provided. Using today's date: {start_date}")
        else:
            assert datetime.strptime(start_date, '%Y-%m-%d') <= datetime.now(), "Start date cannot be in the future"
        return start_date
    def __call__(self, start_date=None):

        start_date = self.assert_date(start_date)

        LOG.info("Retrieving the model from mlflow server...")
        self.mlflow_model = mlutils.load_registered_model(
            model_name='sku_forecasting_tft',
            alias='best_model',
        )

        self.tft_parameters = self.mlflow_model.unwrap_python_model().tft_parameters
        self.unique_categories = self.mlflow_model.unwrap_python_model().unique_categories

        LOG.info("Constructing encoder input...")
        encoder_data = self.prepare_encoder_data(start_date)

        LOG.info("Constructing decoder input...")
        decoder_data = self.prepare_decoder_data(encoder_data)

        LOG.info("Constructing inference data...")
        inference_data = self.construct_inference_data(encoder_data, decoder_data)
        self.model_input = {
            'inference_data': inference_data,
        }

        LOG.info("Predicting using the model...")
        predictions = self.mlflow_model.predict(self.model_input)
        df_output = self.generate_final_data(
            model_prediction=predictions,
            start_date=start_date
        )

        LOG.info("Writing output to BigQuery...")
        self.push_to_bq(df_output)

        LOG.info("Writing predict vs actual table...")
        self.write_predict_actual_table()
        LOG.info("Done!")

        return


if __name__ == '__main__':
    forecast = TftForecast()
    forecast()
