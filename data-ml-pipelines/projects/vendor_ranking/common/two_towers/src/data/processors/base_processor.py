from abc import ABC, abstractmethod
import pandas as pd
from pathlib import Path
import yaml


class BaseProcessor(ABC):
    def __init__(self, df: pd.DataFrame):
        self.df = df

    @abstractmethod
    def process(self):
        """Process the data and return the result."""
        pass


def process_data(df, data_type):
    float64_columns = df.select_dtypes(include=["float64"]).columns
    df[float64_columns] = df[float64_columns].astype("float32")
    in64_columns = df.select_dtypes(include=["int64"]).columns
    df[in64_columns] = df[in64_columns].astype("int32")

    variables_type_path = Path(__file__).parent.resolve() / f"features_type/{data_type}.yaml"
    with open(variables_type_path, "r") as file:
        variables_type = yaml.safe_load(file)

    for col_info in variables_type["column_types"]:
        col_name = col_info["column_name"]
        dtype = col_info["dtype"]
        df[col_name] = df[col_name].astype(dtype)

    return df