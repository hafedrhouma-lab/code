from abc import ABC, abstractmethod
import pandas as pd


class DataFetcherInterface(ABC):
    @abstractmethod
    def fetch_data(self, source: str, description: str) -> pd.DataFrame:
        """Fetch data using the provided query."""
        pass