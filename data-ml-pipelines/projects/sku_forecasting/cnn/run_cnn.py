import mlflow 
import structlog
import pandas as pd 
import pickle 
import csv
import numpy as np
import cloudpickle
import sys 
from base.v0 import mlutils
from base.v0.db_utils import BigQuery
import os
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from tqdm.notebook import tqdm 
from projects.sku_forecasting.common.utils import metrics_evaluation, get_training_data
from datetime import datetime, timedelta
from projects.sku_forecasting.common.utils import metrics_evaluation
import datetime as dt
import pandas as pd
import numpy as np
from base.v0.db_utils import BigQuery
from dotenv import load_dotenv
load_dotenv(override=True)
LOG: structlog.stdlib.BoundLogger = structlog.get_logger()


class DataGenerator:
    def __init__(self, df, batch_size):
        self.df = df
        self.batch_size = batch_size
        self.total_batches = self.calculate_total_batches()

    def calculate_total_batches(self):
        unique_combinations = self.df[['sku', 'warehouse_id']].drop_duplicates()
        total_batches = 0
        for index, row in unique_combinations.iterrows():
             total_batches += 1
        return total_batches

    def __len__(self):
        return self.total_batches

    def split_data_X_Y_generator(self, sample):
        columns_to_drop_2 = ['order_date', 'sku', 'warehouse_id']
        input_window_size = 60
        output_window_size = 30
        X1_all = []
        X2_all = []
        y_all = []
        
        for i in range(0, len(sample) - input_window_size - output_window_size + 1, 30):
            input_start_index = i
            input_end_index = i + input_window_size
            output_start_index = input_end_index
            output_end_index = output_start_index + output_window_size
            
            input_window_last_month = sample.iloc[input_start_index:input_end_index].drop(columns_to_drop_2, axis=1).values
            input_window_this_month = sample.iloc[output_start_index:output_end_index].drop(columns_to_drop_2, axis=1).drop(['sales'], axis=1).values
            output_window = sample.iloc[output_start_index:output_end_index]['sales'].values
            
            X1_all.append(input_window_last_month)
            X2_all.append(input_window_this_month)
            y_all.append(output_window)
            
            if len(X1_all) == self.batch_size:
                X1 = np.array(X1_all, dtype=np.float32)
                X2 = np.array(X2_all, dtype=np.float32)
                y = np.array(y_all, dtype=np.float32)
                yield ((X1, X2), y)
                X1_all, X2_all, y_all = [], [], []

        if len(X1_all) > 0:
            X1 = np.array(X1_all, dtype=np.float32)
            X2 = np.array(X2_all, dtype=np.float32)
            y = np.array(y_all, dtype=np.float32)
            yield ((X1, X2), y)

    def combined_generator(self):
        unique_combinations = self.df[['sku', 'warehouse_id']].drop_duplicates()
        for _, row in unique_combinations.iterrows():
            sku = row['sku']
            warehouse_id = row['warehouse_id']
            sample = self.df[(self.df['sku'] == sku) & (self.df['warehouse_id'] == warehouse_id)]
            yield from self.split_data_X_Y_generator(sample)



class CnnForecast:
    def __init__(self):
        self.model_input = {}
        exp_name = os.environ['MLFLOW_EXPERIMENT_NAME']
        self.MODEL_NAME = exp_name
        LOG.info(f"Tracking uri: {mlflow.get_tracking_uri()}")
        LOG.info(f"Experiment id: {exp_name}")
    def preprocess_data(self,data):
        # label encoding category data
        columns_to_label_encode= ['parent_category','master_category']
        encoder = LabelEncoder()
        for col in columns_to_label_encode:
            data[col] = encoder.fit_transform(data[col])
        # reformatting date 
        data['order_date']=pd.to_datetime(data['order_date'])

        # dropping textual columns 
        columns_to_drop = ['country_code', 'day_name']
        data.drop(columns_to_drop, axis=1, inplace=True)
        return data 
    

    def predict_in_production(self, production_data):
        #with open('/Users/mariam.gaafar/Desktop/Github/data-ml-pipelines/projects/sku_forecasting/cnn/sku_forecasting_cnn.pkl', 'rb') as f:
        #    model = cloudpickle.load(f)

        LOG.info("Retrieving the model from mlflow server...")
        model = mlutils.load_registered_model(
            model_name='sku_forecasting_cnn',
            alias='best_model',
        )
        output_signature = (
            (tf.TensorSpec(shape=(None, 60, 19), dtype=tf.float32), tf.TensorSpec(shape=(None, 30, 18), dtype=tf.float32)),
            tf.TensorSpec(shape=(None, 30), dtype=tf.float32)
        )

        # Create the dataset using the generator
        test_g = DataGenerator(production_data, batch_size=32)
        test_dataset = tf.data.Dataset.from_generator(
            test_g.combined_generator,
            output_signature=output_signature
        ).prefetch(tf.data.AUTOTUNE)  # Prefetching for better performance

        # Get the total number of steps for the progress bar

        total_steps = len(test_g)
        print("Totl setepes ",total_steps)
        # Initialize the progress bar
        # Generate predictions for the test generator
        test_predictions = []
        test_actual_values = []

        for step, (inputs, targets) in enumerate(test_dataset):
            print(inputs)
            input_dict= dict()
            input_dict['input']= inputs
            preds = model.predict(input_dict)
            test_predictions.append(preds)
            test_actual_values.append(targets)
            if step >= total_steps - 1:
                break
        test_predictions = np.concatenate(test_predictions)
        test_actual_values = np.concatenate(test_actual_values)

        metrics = metrics_evaluation(
            actual=test_actual_values,
            predicted=test_predictions
        )

        return metrics, test_predictions

    
    
    def __call__(self, data ):
        preprocessed_data= self.preprocess_data(data )
        print("Preprocessing done successfully ")

        metrics,predictions =self.predict_in_production(preprocessed_data)
        
        print("Predictions ran  successfully ")

        return  metrics,predictions




    
if __name__ == "__main__":
    data = get_training_data(sku_pattern='900152', warehouse_pattern='',history_duration=60, is_prediction=True)
    data_with_holidays = data.copy()
    
    
    forecast = CnnForecast()
    metrics,predictions = forecast(data_with_holidays)
    
    
    integer_predictions = pd.DataFrame(np.round(predictions).astype(int))
    forecasts = pd.DataFrame(integer_predictions)
    today = datetime.today().date()
    date_list = [(today + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(30)]
    df = pd.DataFrame(integer_predictions)

    # Assign the date strings as column names
    df.columns = date_list


    df['sku'] = data_with_holidays[['sku','warehouse_id']].drop_duplicates()['sku'].reset_index(drop=True)
    df['warehouse_id'] = data_with_holidays[['sku','warehouse_id']].drop_duplicates()['warehouse_id'].reset_index(drop=True)


    df_melted = pd.melt(df, id_vars=['sku', 'warehouse_id'], var_name='order_date', value_name='forecasts')


    val_data = data_with_holidays[['order_date', 'sku', 'warehouse_id', 'sales']]

    # Assuming 'today' is a given date, e.g., May 20, 2024
    today = datetime(2024, 5, 26).date()
    # Filter the DataFrame before converting to string
    val_data = val_data[(val_data['order_date'].dt.date > today - timedelta(days=15)) & (val_data['order_date'].dt.date <= today)]

    # Convert the 'order_date' column to string format
    val_data['order_date'] = val_data['order_date'].dt.strftime('%Y-%m-%d')
    cnn_out_forecast =pd.merge(df_melted,val_data, how='outer')



    # Convert 'order_date' column to datetime
    cnn_out_forecast['order_date'] = pd.to_datetime(cnn_out_forecast['order_date'], errors='coerce')

    # Check the data info to ensure the conversion
    print(cnn_out_forecast.info())

    # Define the schema
    schema = []
    for col in cnn_out_forecast.columns:
        if col == 'order_date':
            schema.append({'name': col, 'type': 'DATE'})
        elif col in ['sales', 'forecasts']:
            schema.append({'name': col, 'type': 'FLOAT'})
        else:
            schema.append({'name': col, 'type': 'STRING'})

    # Initialize BigQuery connection
    bq = BigQuery()

    # Write the data to BigQuery
    bq.write(
        cnn_out_forecast,
        'data_platform_sku_forecasting.cnn_forecast_debug',
        'replace',
        schema=schema
    )
