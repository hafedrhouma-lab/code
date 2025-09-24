import structlog
from collections import defaultdict


LOG: "structlog.stdlib.BoundLogger" = structlog.getLogger(__name__)


def sample_rows_by_group(df, group_col, sample_size, replace=False):
    """
    Samples a specified number of rows from each group in a DataFrame.
    If the group size is less than the sample_size, the group is returned as is.

    Args:
        df (pd.DataFrame): The input DataFrame.
        group_col (str): The column to group by.
        sample_size (int): The number of rows to sample from each group.
        replace (bool): Whether to sample with replacement. Default is False.

    Returns:
        pd.DataFrame: A DataFrame with the sampled rows.
    """
    return df.groupby(group_col, group_keys=False).apply(
        lambda group: group.sample(n=min(len(group), sample_size), replace=replace)
    ).reset_index(drop=True)


def group_data_by_table(tables_and_data):
    """
    Groups data frames by table name.

    Args:
        tables_and_data (list): A list of tuples (table_name, data_frame).

    Returns:
        dict: A dictionary where keys are table names and values are lists of data frames.
    """
    grouped_data = defaultdict(list)
    for table_name, data_frame in tables_and_data:
        grouped_data[table_name].append(data_frame)
    return grouped_data
