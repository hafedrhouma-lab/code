import os
import pandas as pd
import numpy as np
import mlflow
import pickle
import structlog
import cloudpickle
from model_wrapper import ModelWrapper
from keras.layers import Dense, Flatten
from keras.models import Sequential
from keras.layers import Conv1D,MaxPooling1D
import itertools
from sklearn.preprocessing import StandardScaler
from projects.sku_forecasting.common.utils import get_training_data, metrics_evaluation
from dotenv import load_dotenv
from sklearn.preprocessing import LabelEncoder
from datetime import datetime, timedelta, timezone
from keras.models import Model
from keras.layers import Input, Conv1D, MaxPooling1D, Flatten, concatenate, Dropout, BatchNormalization, Dense
from keras.optimizers import Adam
import datetime as  dt 
import random
from keras.callbacks import ReduceLROnPlateau
import tensorflow as tf

load_dotenv(override=True)


LOG: structlog.stdlib.BoundLogger = structlog.get_logger()

class DataGenerator:
    def __init__(self, df, batch_size):
        self.df = df
        self.batch_size = batch_size

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

        # Yield the last batch if it's not empty
        if len(X1_all) > 0:
            X1 = np.array(X1_all, dtype=np.float32)
            X2 = np.array(X2_all, dtype=np.float32)
            y = np.array(y_all, dtype=np.float32)
            yield ((X1, X2), y)

    def combined_generator(self):
        unique_combinations = self.df[['sku', 'warehouse_id']].drop_duplicates()
        for index, row in unique_combinations.iterrows():
            sku = row['sku']
            warehouse_id = row['warehouse_id']
            sample = self.df[(self.df['sku'] == sku) & (self.df['warehouse_id'] == warehouse_id)]
            yield from self.split_data_X_Y_generator(sample)


class ModelTrainer:
    def __init__(self):
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
    
    def train_test_valid_split(self,data):
        datetime_str = '24/03/24'
        datetime_object = datetime.strptime(datetime_str, '%d/%m/%y')
        start_date = datetime_object - dt.timedelta(days=(450))
        train_end_date = start_date + timedelta(days=419)
        train_end = train_end_date.replace(tzinfo=timezone.utc)
        train_df = copy_of_data[copy_of_data['order_date'] <= train_end]

        # Last 30 days for testing
        test_start_date = start_date + timedelta(days=360)
        test_start = test_start_date.replace(tzinfo=timezone.utc)
        test_val_df = copy_of_data[copy_of_data['order_date'] >= test_start]

        # Assuming 'df' is your dataframe containing the table
        # Replace 'df' with the actual name of your dataframe

        # Get unique SKU and warehouse pairs
        unique_sku_warehouse_pairs = test_val_df[['sku', 'warehouse_id']].drop_duplicates()

        # Shuffle the unique pairs randomly
        random.shuffle(unique_sku_warehouse_pairs.values)

        # Define the number of pairs for each table
        num_pairs_table1 = len(unique_sku_warehouse_pairs)//2 
        num_pairs_table2 = len(unique_sku_warehouse_pairs)//2

        # Select random pairs for table 1 and table 2
        sku_warehouse_pairs_test = unique_sku_warehouse_pairs[:num_pairs_table1]
        sku_warehouse_pairs_validation = unique_sku_warehouse_pairs[num_pairs_table1:num_pairs_table1+num_pairs_table2]

        # Filter the dataframe based on sku-warehouse pairs for table 1
        test_df = test_val_df[(test_val_df['sku'].isin(sku_warehouse_pairs_test['sku'])) & 
                    (test_val_df['warehouse_id'].isin(sku_warehouse_pairs_test['warehouse_id']))]

        # Filter the dataframe based on sku-warehouse pairs for table 2
        validation_df = test_val_df[(test_val_df['sku'].isin(sku_warehouse_pairs_validation['sku'])) & 
                    (test_val_df['warehouse_id'].isin(sku_warehouse_pairs_validation['warehouse_id']))]

        # Reset index for both tables
        test_df.reset_index(drop=True, inplace=True)
        validation_df.reset_index(drop=True, inplace=True)

        return train_df, validation_df , test_df 
    


    def train_test_valid_split_generators(self, data):
        train_df, test_df , valid_df = self.train_test_valid_split(data)
        # Define the output signature for the dataset
        output_signature = (
            (tf.TensorSpec(shape=(None, 60, 19), dtype=tf.float32), tf.TensorSpec(shape=(None, 30, 18), dtype=tf.float32)),
            tf.TensorSpec(shape=(None, 30), dtype=tf.float32)
        )

        # Create the dataset using the generator
        train_generator = DataGenerator(train_df, batch_size=32)
        train_dataset = tf.data.Dataset.from_generator(
            train_generator.combined_generator,
            output_signature=output_signature
        )
        test_generator = DataGenerator(test_df, batch_size=32)
        test_dataset = tf.data.Dataset.from_generator(
            test_generator.combined_generator,
            output_signature=output_signature
        )
        valid_generator = DataGenerator(valid_df, batch_size=32)
        test_dataset = tf.data.Dataset.from_generator(
            valid_generator.combined_generator,
            output_signature=output_signature
        )
    
        return train_dataset, test_dataset, test_dataset

                

    def build_cnn(self):
        # Define the shape of your inputs
        n_features1 = 19  # Number of features in the first time series input
        n_features2 = 18  # Number of features in the second time series input
        n_timesteps1 = 60  # Number of time steps in the first time series input
        n_timesteps2 = 30  # Number of time steps in the second time series input

        # First time series input model
        input_ts1 = Input(shape=(n_timesteps1, n_features1))
        conv1 = Conv1D(filters=32, kernel_size=3, activation='relu')(input_ts1)
        pool1 = MaxPooling1D(pool_size=2)(conv1)
        flat1 = Flatten()(pool1)

        # Second time series input model
        input_ts2 = Input(shape=(n_timesteps2, n_features2))
        conv2 = Conv1D(filters=32, kernel_size=3, activation='relu')(input_ts2)
        pool2 = MaxPooling1D(pool_size=2)(conv2)
        conv2_extra = Conv1D(filters=32, kernel_size=3, activation='relu')(pool2)
        pool2_extra = MaxPooling1D(pool_size=2)(conv2_extra)
        flat2_extra = Flatten()(pool2_extra)

        # Combine the outputs of the two branches
        merged = concatenate([flat1, flat2_extra])

        # Add dropout and batch normalization for regularization
        merged = Dropout(0.7)(merged)
        merged = BatchNormalization()(merged)

        # Add a fully connected layer and an output layer
        dense1 = Dense(128, activation='relu')(merged)
        dense1 = Dropout(0.7)(dense1)
        output = Dense(30, activation='relu')(dense1)

        # Create the model
        model = Model(inputs=[input_ts1, input_ts2], outputs=output)

        # Instantiate the Adam optimizer with the desired learning rate
        adam_optimizer = Adam(learning_rate=0.000001)

        # Compile the model with Mean Squared Error loss and Adam optimizer
        model.compile(optimizer=adam_optimizer, loss='mse')

        # Print the model summary
        print(model.summary())
        return model


    def run_cnn(self,model,train_generator, validation_generator, test_generator):


        # Adding a learning rate scheduler to adjust learning rate during training
        #reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.000001)
        with mlflow.start_run() as self.run:  # Do not change this line
     
            history = model.fit(
                train_generator,
                epochs=1,
                steps_per_epoch=5000,
                validation_data=validation_generator,
                validation_steps=1
            )


            # Generate predictions for the test generator
            test_predictions = model.predict(test_generator,steps=1000)

            # Collect actual values from the test generator
            steps=0
            test_actual_values = []
            for inputs, targets in test_generator:
                steps+=1
                if(steps<=1000):
                    test_actual_values.append(targets)
                else:
                    break
            test_actual_values = np.concatenate(test_actual_values)

            metrics = metrics_evaluation(
                    actual=test_actual_values,
                    predicted=test_predictions
                )

            mlflow.log_metrics(metrics)
            mlflow.log_metric("success_metric", metrics['success_metric'])

            print(metrics)
            LOG.info(f"Validation metrics: {metrics}")
            # Create a temporary file to save the serialized model
            with open(f"{self.MODEL_NAME}.pkl", "wb") as f:
                f.write(cloudpickle.dumps(model))



            mlflow.pyfunc.log_model(
                artifact_path='model',
                python_model=ModelWrapper(),
                artifacts={
                    "model": f"{self.MODEL_NAME}.pkl"
                    # add more artifacts as needed
                },
                code_path=[
                    "../../../base/v0/mlclass.py",
                    "../../../base/v0/mlutils.py",
                    "./model_wrapper.py"
                ]
            )
            LOG.info("Model logged to mlflow server")



    def __call__(self, data ):
        preprocessed_data= self.preprocess_data(data )
        LOG.info("Preprocessing done successfully ")
    
        train_generator, validation_generator, test_generator = self.train_test_valid_split_generators(preprocessed_data)
        LOG.info("Generators produced  successfully ")

        model= self.build_cnn()
        LOG.info("Model built  successfully ")

        self.run_cnn(model,train_generator, validation_generator, test_generator )
        LOG.info("Model ran  successfully ")

        return  

    
if __name__ == "__main__":
    data_with_holidays = get_training_data(sku_pattern='', warehouse_pattern='')
    copy_of_data = data_with_holidays.copy()
    trainer = ModelTrainer()
    trainer(copy_of_data)