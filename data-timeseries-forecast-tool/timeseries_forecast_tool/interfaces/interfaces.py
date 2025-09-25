from abc import ABC, abstractmethod
import pandas as pd


class IModelDaily(ABC):
    @abstractmethod
    def prepare(self, timeseries_df: pd.DataFrame):
        """prepare on timeseries"""
        raise NotImplementedError

    @abstractmethod
    def fit(self, timeseries_df: pd.DataFrame):
        """Fit model on timeseries dataframe"""
        raise NotImplementedError

    @abstractmethod
    def predict(self, timeseries_df: pd.DataFrame):
        """Predict on timeseries dataframe"""
        raise NotImplementedError
