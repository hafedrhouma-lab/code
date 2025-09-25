import pandas as pd


def time_split(df: pd.DataFrame, date_col: str, cutoff: str):
    dt = pd.to_datetime(df[date_col], errors="coerce")
    mask = dt < pd.to_datetime(cutoff)
    return df.loc[mask].reset_index(drop=True), df.loc[~mask].reset_index(drop=True)
