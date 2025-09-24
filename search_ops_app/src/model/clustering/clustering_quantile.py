"""Definition of Quantile classifier module"""
import pandas as pd
import numpy as np
from . import IModel
from src import PROJECT
from src.utils.log import Logger


class QuantileModel(IModel):
    """model to clusterize based on
    quantile distribution"""
    MANUAL_QUANTILE_VALUE = 10.0

    def __init__(self, quantile_columns: list):
        self.quantile_columns = quantile_columns
        self.labels = {}
        self.model_name = "quantile_classifier"
        self.model_fname = f"{PROJECT}-{self.model_name}"
        self._logger = Logger(self.model_name).get_logger()

    def fit(self):
        pass

    def predict(
            self,
            data_frame: pd.DataFrame,
            n_quantile: int = 3) -> pd.DataFrame:
        self._logger.info(f"Data Size: {data_frame.shape[0]}")
        data_frame_clustered = data_frame.copy()
        for col in self.quantile_columns:
            data_frame_clustered[
                col+"_quantile"
            ] = pd.qcut(
                    data_frame[col],
                    q=n_quantile,
                    duplicates="drop",
                    precision=0,
                    retbins=False
            ).astype(str)

            if len(
                    np.unique(
                        data_frame_clustered[col+"_quantile"]
                    )
            ) == 1:
                data_frame_clustered.loc[
                    data_frame[col] <=
                    self.MANUAL_QUANTILE_VALUE, col+"_quantile"
                ] = f"(0.0, {self.MANUAL_QUANTILE_VALUE}]"

                data_frame_clustered.loc[
                    data_frame[col] >
                    self.MANUAL_QUANTILE_VALUE, col+"_quantile"
                ] = f"({self.MANUAL_QUANTILE_VALUE}, 100.0]"

            self.labels[col+"_quantile"] = list(
                np.unique(
                    data_frame_clustered[col+"_quantile"]
                )
            )

        self._logger.info("Prediction is done")
        return data_frame_clustered

    def save(self, version):
        pass

    def load(self, version):
        pass
