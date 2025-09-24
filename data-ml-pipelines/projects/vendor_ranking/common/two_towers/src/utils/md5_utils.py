from typing import Optional, Iterable
from hashlib import md5
import pandas as pd


def get_md5_from_series(input_iterable: Iterable) -> str:
    """
    Create a MD5 hash from an Iterable, typically a row from a Pandas ``DataFrame``, but can be any
    Iterable object instance such as a list, tuple or Pandas ``Series``.

    Args:
        input_iterable: Typically a Pandas ``DataFrame`` row, but can be any Pandas ``Series``.

    Returns:
        MD5 hash created from the input values.
    """
    # convert all values to string, concantenate, and encode so can hash
    full_str = "".join(map(str, input_iterable)).encode("utf-8")

    # create a md5 hash from the complete string
    md5_hash = md5(full_str).hexdigest()

    return md5_hash


def get_md5_series_from_dataframe(
    input_dataframe: pd.DataFrame, columns: Optional[Iterable[str]] = None
) -> pd.Series:
    """
    Create a Pandas ``Series`` of MD5 hashses for every row in a Pandas ``DataFrame``.

    Args:
        input_dataframe: Pandas ``DataFrame`` to be create MD5 hashes for.
        columns: If only wanting to use specific columns to calculate the hash, specify these here.

    Returns:
        MD5 hashes, one for every row in the input Pandas ``DataFrame``.
    """

    # if columns specified, filter to just these columns
    in_df = (
        input_dataframe.iloc[:, list(columns)]
        if columns is not None
        else input_dataframe
    )

    # create md5 hash per row
    md5_hashes = in_df.apply(lambda row: get_md5_from_series(row), axis=1)

    return md5_hashes
