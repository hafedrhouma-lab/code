import numpy as np
import pandas as pd
import mlflow
import structlog
from model_wrapper import ModelWrapper
from projects.sku_forecasting.common.utils import get_training_data, metrics_evaluation
from pytorch_forecasting import GroupNormalizer, TemporalFusionTransformer, TimeSeriesDataSet
from lightning.pytorch.callbacks import EarlyStopping
from pytorch_forecasting.metrics import RMSE, QuantileLoss
import os
from base.v0.file_tools import collect_files, temp_open_file, temp_dir
from pathlib import Path
import cloudpickle
import lightning.pytorch as pl
from lightning.pytorch.tuner import Tuner
import warnings
from dotenv import load_dotenv


warnings.simplefilter(action='ignore', category=FutureWarning)
load_dotenv(override=True)
LOG: structlog.stdlib.BoundLogger = structlog.get_logger()


class ModelTrainer(ModelWrapper):
    def __init__(self):
        super().__init__()

        exp_name = os.environ['MLFLOW_EXPERIMENT_NAME']
        self.MODEL_NAME = exp_name
        LOG.info(f"Tracking uri: {mlflow.get_tracking_uri()}")
        LOG.info(f"Experiment id: {exp_name}")

        self.date_column = 'order_date'
        self.target_column = 'sales'
        self.unique_categories = None
        self.tft_parameters = {
            'group_ids': ['country_code', 'sku', 'warehouse_id', 'parent_category', 'master_category'],
            'static_categoricals': ['country_code', 'sku', 'warehouse_id', 'parent_category', 'master_category'],
            'time_varying_known_categoricals': [
                "special_days",
                # "day_name",
                "day_of_week",
                "week_number",
                "month",
                "day_number"
            ],
            'time_varying_known_reals': ["time_idx"],
            'time_varying_unknown_reals': ['sales'],
            'min_encoder_length': 90,
            'max_encoder_length': 90,
            'min_prediction_length': 30,
            'max_prediction_length': 30,
            'hidden_size': 32,  # most important hyperparameter apart from learning rate
            'attention_head_size': 4,
            'dropout': 0.2,  # between 0.1 and 0.3 are good values
            'hidden_continuous_size': 8,
            'batch_size': 128,
            'epochs': 100,
            'learning_rate': None,
            'gradient_clip_val': 0.1,
            'optimizer': None,
            'loss': 'QuantileLoss',
            # 'loss': 'RMSE',
            'quantiles': [0.1, 0.4, 0.5, 0.6, 0.9],
            'special_days': [],
        }

        self.trainTimeSeriesData = None
        self.valTimeSeriesData = None
        self.testTimeSeriesData = None

    def prepare_time_series_data(self, data):
        training_cutoff = data["time_idx"].max() - self.tft_parameters['max_prediction_length']
        self.validation_start_date = data[data["time_idx"] == training_cutoff + 1][self.date_column].min()
        LOG.info(f"training cutoff: {training_cutoff}")

        self.trainTimeSeriesData = TimeSeriesDataSet(
            data[lambda x: x.time_idx <= training_cutoff],
            time_idx="time_idx",
            target=self.target_column,
            group_ids=self.tft_parameters['group_ids'],
            min_encoder_length=self.tft_parameters['max_encoder_length'],
            max_encoder_length=self.tft_parameters['max_encoder_length'],
            min_prediction_length=30,  # self.tft_parameters['max_prediction_length'],
            max_prediction_length=self.tft_parameters['max_prediction_length'],
            static_categoricals=self.tft_parameters['static_categoricals'],
            time_varying_known_categoricals=self.tft_parameters['time_varying_known_categoricals'],
            time_varying_known_reals=["time_idx"],
            time_varying_unknown_categoricals=[],
            time_varying_unknown_reals=self.tft_parameters['time_varying_unknown_reals'],
            variable_groups={"special_days": self.tft_parameters['special_days']},
            # group of categorical variables can be treated as one variable
            target_normalizer=GroupNormalizer(
                groups=self.tft_parameters['static_categoricals'], transformation="softplus"
            ),
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True,
        )
        self.valTimeSeriesData = TimeSeriesDataSet.from_dataset(self.trainTimeSeriesData, data, predict=True, stop_randomization=True)

        # create dataloaders for model
        train_dataloader = self.trainTimeSeriesData.to_dataloader(train=True, batch_size=self.tft_parameters['batch_size'], num_workers=15, persistent_workers=True)
        val_dataloader = self.valTimeSeriesData.to_dataloader(train=False, batch_size=self.tft_parameters['batch_size'], num_workers=15, persistent_workers=True)
        return train_dataloader, val_dataloader

    def create_model(self):

        if self.tft_parameters['loss'] == 'RMSE':
            loss = RMSE()
            output_size = 1
            self.tft_parameters['optimizer'] = "AdamW"
        elif self.tft_parameters['loss'] == 'QuantileLoss':
            loss = QuantileLoss(self.tft_parameters['quantiles'])
            output_size = len(self.tft_parameters['quantiles'])
            self.tft_parameters['optimizer'] = "Ranger"
        else:
            raise ValueError("Loss function not supported")

        if self.tft_parameters['learning_rate'] is None:
            LOG.info("Creating a dummy model to find the best learning rate")
            model = TemporalFusionTransformer.from_dataset(
                self.trainTimeSeriesData,
                learning_rate=0.34,

                hidden_size=self.tft_parameters['hidden_size'],
                attention_head_size=self.tft_parameters['attention_head_size'],
                dropout=self.tft_parameters['dropout'],
                hidden_continuous_size=self.tft_parameters['hidden_continuous_size'],
                output_size=output_size,
                loss=loss,
                optimizer=self.tft_parameters['optimizer'],
            )
        else:
            LOG.info("creating a tft model")
            model = TemporalFusionTransformer.from_dataset(
                self.trainTimeSeriesData,
                learning_rate=self.tft_parameters['learning_rate'],
                hidden_size=self.tft_parameters['hidden_size'],
                attention_head_size=self.tft_parameters['attention_head_size'],
                dropout=self.tft_parameters['dropout'],
                hidden_continuous_size=self.tft_parameters['hidden_continuous_size'],
                log_interval=10,
                output_size=output_size,
                loss=loss,
                optimizer=self.tft_parameters['optimizer'],
                reduce_on_plateau_patience=4,
            )

        LOG.info(f"Number of parameters in network: {model.size() / 1e3:.1f}k")
        return model

    def special_tft_preprocessing(self, data):
        self.tft_parameters['special_days'] = [col for col in data.columns if col.startswith("Holiday_")]
        data[self.tft_parameters['special_days']] = data[self.tft_parameters['special_days']].apply(lambda x: x.map({0: "-", 1: x.name})).astype("category")
        data['parent_category'] = data['parent_category'].fillna('no_parent_category').astype("category")
        data['master_category'] = data['master_category'].fillna('no_master_category').astype("category")

        return data

    def get_best_learning_rate(self, train_dataloader, val_dataloader):
        # set random seed for reproducibility
        pl.seed_everything(42)
        lr_trainer = pl.Trainer(
            accelerator="cpu",
            gradient_clip_val=self.tft_parameters['gradient_clip_val'],
            # num_sanity_val_steps=0,  # Disable sanity check
        )

        model = self.create_model()
        res = Tuner(lr_trainer).lr_find(
            model,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader,
            max_lr=10.0,
            min_lr=1e-6,
        )
        self.tft_parameters['learning_rate'] = res.suggestion()
        LOG.info(f"suggested learning rate: {self.tft_parameters['learning_rate']}")
        return self.tft_parameters['learning_rate']

    def generate_final_data(self, model_prediction, tft_model):
        data = []
        predicted = np.array(model_prediction.output.prediction)
        actual = np.array(model_prediction.y[0])

        for i in range(len(self.tft_parameters['quantiles'])):
            data.append(predicted[:, :, i].flatten())
        data.append(actual.flatten())

        cols = [f'q_{q}'.replace('.', '') for q in self.tft_parameters['quantiles']]
        cols += ["Actual"]

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
        dates = pd.date_range(start=self.validation_start_date, periods=decoder_length, freq='D')
        repeated_dates = np.tile(dates, len(df_output) // decoder_length)
        repeated_dates = pd.DatetimeIndex(repeated_dates)
        df_output.insert(0, 'forecast_date', repeated_dates)

        # Concatenate the metadata and the realization
        df_output = pd.concat([df_output, df_realization], axis=1)
        return df_output

    def train_model(self, train_dataloader, val_dataloader):
        with mlflow.start_run() as run:  # Do not change this line
            # configure network and trainer

            early_stop_callback = EarlyStopping(
                monitor="val_loss",
                min_delta=1e-4,
                patience=15,
                verbose=False,
                mode="min"
            )
            tft_trainer = pl.Trainer(
                max_epochs=self.tft_parameters['epochs'],
                accelerator="cpu",
                enable_model_summary=False,
                gradient_clip_val=self.tft_parameters['gradient_clip_val'],
                limit_train_batches=self.tft_parameters['batch_size'],
                # num_sanity_val_steps=0,  # Disable sanity check
                callbacks=[early_stop_callback],
            )

            model = self.create_model()
            tft_trainer.fit(
                model,
                train_dataloaders=train_dataloader,
                val_dataloaders=val_dataloader,
            )
            best_model_path = tft_trainer.checkpoint_callback.best_model_path
            LOG.info(f"Best model path: {best_model_path}")

            best_tft_model = TemporalFusionTransformer.load_from_checkpoint(best_model_path)

            # Log the model parameters
            mlflow.log_params(self.tft_parameters)

            # Log the metrics
            LOG.info("-- Evaluating the model --")
            val_predictions_raw = best_tft_model.predict(
                val_dataloader,
                return_x=True,
                return_y=True,
                mode="raw",
                trainer_kwargs=dict(accelerator="cpu")
            )
            actual = np.array(val_predictions_raw.y[0])
            median_index = val_predictions_raw.output.prediction.shape[2] // 2
            predicted = np.array(val_predictions_raw.output.prediction[:, :, median_index])
            df_output = self.generate_final_data(val_predictions_raw, best_tft_model)

            metrics = metrics_evaluation(
                    actual=actual,
                    predicted=predicted
                )

            mlflow.log_metrics(metrics)
            LOG.info(f"Validation metrics: {metrics}")

            mlflow.set_tags(
                {
                    'unique_skus': self.unique_categories.shape[0]
                }
            )
            # Save the model using mlflow's pyfunc
            model_pkl_file = f"{self.MODEL_NAME}.pkl"
            unique_categories_artifact = "unique_categories.csv"
            tft_parameters_artifact = "tft_parameters.pkl"
            validation_output_artifact = "validation_output.csv"

            self.unique_categories.to_csv(unique_categories_artifact, index=False)
            df_output.to_csv(validation_output_artifact, index=False)
            with open(tft_parameters_artifact, "wb") as f:
                f.write(cloudpickle.dumps(self.tft_parameters))

            code_paths = collect_files(
                base=Path(__file__).parent,
                src_dirs=["../../../base", "../common"],
                src_files=["model_wrapper.py"]
            )

            with temp_open_file(model_pkl_file, "wb") as f:
                f.write(cloudpickle.dumps(model))
                ## The code_paths variable is used to specify the paths to the files that are needed to load the model
                with temp_dir(base="./", paths=code_paths) as dir_path:
                # Save the model using mlflow's pyfunc
                    mlflow.pyfunc.log_model(
                        artifact_path='model',
                        python_model=ModelWrapper(),
                        artifacts={
                            "model": model_pkl_file,
                            "unique_categories": unique_categories_artifact,
                            "tft_parameters": tft_parameters_artifact,
                            "validation_output": validation_output_artifact,
                        },
                        code_path=code_paths
                    )
                    LOG.info("Model logged to mlflow server")
            os.remove(unique_categories_artifact)
            os.remove(tft_parameters_artifact)
            os.remove(validation_output_artifact)
            return best_tft_model

    def grid_search(self, raw_data):
        import gc
        for min_encoder_length in [60, 90]:
                for hidden_size in [16, 32]:
                    for attention_head_size in [2, 4]:
                        for hidden_continous_size in [4, 8]:
                            for batch_size in [256]:
                                self.tft_parameters['min_encoder_length'] = min_encoder_length
                                self.tft_parameters['hidden_size'] = hidden_size
                                self.tft_parameters['attention_head_size'] = attention_head_size
                                self.tft_parameters['hidden_continous_size'] = hidden_continous_size
                                self.tft_parameters['batch_size'] = batch_size
                                LOG.info(f"Training with parameters: {self.tft_parameters}")

                                train_dataloader, val_dataloader = self.prepare_time_series_data(raw_data)

                                LOG.info("-- Finding the best learning rate --")
                                self.tft_parameters['learning_rate'] = self.get_best_learning_rate(train_dataloader,
                                                                                                   val_dataloader)

                                LOG.info("-- Training the model --")
                                self.train_model(train_dataloader, val_dataloader)
                                gc.collect()
    def __call__(self, mode='train'):
        '''
        Args:
            mode: str
                'grid_search' or 'train'
        '''
        LOG.info("-- Retrieving the training data --")
        yesterday_str = (pd.Timestamp.today().normalize() - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        raw_data = get_training_data(
            # sku_pattern=90243,
            end_date=yesterday_str,
            history_duration=400,
            # warehouse_pattern='14807687-3b46-4ce6-b669-21f32d111652'
        )
        LOG.info(f"number of unique warehouse id: {raw_data['warehouse_id'].nunique()}")
        LOG.info(f"number of unique sku id: {raw_data['sku'].nunique()}")

        LOG.info("-- tft model preprocessing --")
        raw_data = self.special_tft_preprocessing(raw_data)

        self.unique_categories = pd.DataFrame(list(set(zip(raw_data['country_code'], raw_data['sku'], raw_data['warehouse_id']))),
                                   columns=['country_code', 'sku', 'warehouse_id'])
        self.unique_categories['country_code'] = self.unique_categories['country_code'].str.lower()

        if mode == 'grid_search':
            self.grid_search(raw_data)

        elif mode == 'train':
            train_dataloader, val_dataloader = self.prepare_time_series_data(raw_data)

            LOG.info("-- Finding the best learning rate --")
            self.tft_parameters['learning_rate'] = self.get_best_learning_rate(train_dataloader, val_dataloader)

            LOG.info("-- Training the model --")
            self.train_model(train_dataloader, val_dataloader)
        else:
            LOG.info("Invalid mode. Please use either 'grid_search' or 'train'")

        return


if __name__ == "__main__":
    trainer = ModelTrainer()
    trainer(mode='grid_search')

