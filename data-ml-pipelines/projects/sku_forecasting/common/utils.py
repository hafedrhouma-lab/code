import numpy as np
import pandas as pd 
from datetime import datetime
import datetime as dt
import structlog
from .query_v2 import write_query
from base.v0.db_utils import BigQuery


LOG: structlog.stdlib.BoundLogger = structlog.get_logger()


def get_training_data(
        sku_pattern='',
        warehouse_pattern='',
        end_date=None,
        history_duration=450,
        max_zero_threshold=0.2,
        unique_categories=None,
        is_prediction = False 
):

    """
    Get the training data for the model.
    Args:
        sku_pattern (str): The pattern to match the SKU.
        warehouse_pattern (str): The pattern to match the warehouse.
        end_date (str): The end date for the training data. If None, the current date is used.
        history_duration (int): The number of days to consider in the past.
        max_zero_threshold (float): The maximum percentage of zero sales to consider.
        unique_categories (dataframe): dataframe with the unique sku and warehouse eligible to process

    Returns:
        pd.DataFrame: The training data.
    """
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    else:
        end_date = datetime.strptime(end_date, '%Y-%m-%d').strftime('%Y-%m-%d')

    start_date = datetime.strptime(end_date, '%Y-%m-%d') - dt.timedelta(days=history_duration)
    start_date = start_date.strftime('%Y-%m-%d')


    # loading data
    query = write_query(
        start_date=start_date,
        end_date=end_date,
        sku_pattern=sku_pattern,
        warehouse_pattern=warehouse_pattern,
    )
    bq = BigQuery()
    LOG.info(f"Start date: {start_date}")
    LOG.info(f"End date:   {end_date}")
    if sku_pattern:
        LOG.info(f"SKU pattern: {sku_pattern}")
    if warehouse_pattern:
        LOG.info(f"Warehouse pattern: {warehouse_pattern}")

    LOG.info(f"Duration: {history_duration} days...")
    data_with_master_cat = bq.read(query)
    LOG.info("Data Loaded successfully!")
    LOG.info(f'Raw data - size: {data_with_master_cat.shape}')

    if len(data_with_master_cat) == 0:
        LOG.info("No data found for the given filters")
        return pd.DataFrame()

    if unique_categories is not None:
        LOG.info(f"Unique categories provided - size: {unique_categories.shape}")
        # convert all columns to string to avoid any issues with merging
        unique_categories = unique_categories.astype(str)
        data_with_master_cat[['country_code', 'sku', 'warehouse_id']] = data_with_master_cat[
            ['country_code', 'sku', 'warehouse_id']].astype(str)
        # LOG.info(unique_categories.dtypes)
        # LOG.info(data_with_master_cat.dtypes)
        # LOG.info(data_with_master_cat.loc[data_with_master_cat['sku'] == '902430'])
        df_filled = pd.merge(
            left=data_with_master_cat,
            right=unique_categories,
            on=['country_code', 'sku', 'warehouse_id'],
            how='inner'
        ).reset_index(drop=True)
        LOG.info(f'Merged data (unique categories) - size: {df_filled.shape}')
    else:
        df_filled = data_with_master_cat.copy()

    if (is_prediction):
        end_date = datetime.strptime(end_date, '%Y-%m-%d') + dt.timedelta(days=30)
    df_filled['order_date'] = pd.to_datetime(df_filled['order_date'])
    df_filled = df_filled.sort_values(['order_date', 'country_code', 'sku', 'warehouse_id']).set_index('order_date')

    # Fill missing days
    LOG.info("Filling missing days...")
    df_filled = df_filled.groupby(['country_code', 'sku', 'warehouse_id'], as_index=False).apply(
        lambda x: fill_missing_days(x, start_date, end_date, max_zero_threshold)
    )

    LOG.info(f'Filled data - size: {df_filled.shape}')

    df_filled_time = add_time_features(df_filled).reset_index(drop=True)
    LOG.info(f'Data with time - size: {df_filled_time.shape}')

    data_with_holidays = add_holiday_features(df_filled_time)
    LOG.info(f'Data with holidays - size: {data_with_holidays.shape}')

    # Ensure time_idx is int
    data_with_holidays['time_idx'] = data_with_holidays['time_idx'].astype(int)
    # debug_data(data_with_master_cat, df_filled, data_with_holidays)
    return data_with_holidays

def add_holiday_features(df):
    bq = BigQuery()
    Holidays_query = write_holidays_query()
    df_holidays = bq.read(Holidays_query)

    df['order_date']=pd.to_datetime(df['order_date'], utc=True)
    df['country_code']=df['country_code'].str.upper()
    df_holiday_encoded = pd.get_dummies(df_holidays, columns=['Holiday'], dtype='int')
    df_holiday_encoded = df_holiday_encoded.groupby(['Date', 'Country'], as_index=False).agg(np.bitwise_or.reduce)
    holiday_columns = [col for col in df_holiday_encoded.columns if col.startswith('Holiday_')]
    df_holiday_encoded[holiday_columns] = df_holiday_encoded[holiday_columns].astype(int)

    merged_df = pd.merge(df, df_holiday_encoded, how='left', left_on=['order_date', 'country_code'],
                         right_on=['Date', 'Country']).drop(['Date', 'Country'], axis=1).reset_index(drop=True)

    merged_df.loc[:, holiday_columns] = merged_df[holiday_columns].fillna(0)

    return merged_df


def check_type(x):
    if isinstance(x, int):
        return 'int'
    elif isinstance(x, str):
        return 'str'
    else:
        return 'other'


def percent_zero(group):
    return np.sum(group['sales']== 0)/ len(group ) * 100


def debug_data(input_data , filled_data, final_output):
    type_counts_input = input_data['sku'].apply(check_type).value_counts()
    type_counts_filled = filled_data['sku'].apply(check_type).value_counts()
    type_counts_output = final_output['sku'].apply(check_type).value_counts()
    # Checks weather we have both integers and string SKUs (if we have all str we succeeded)
    LOG.debug(f"Type counts input: {type_counts_input}")
    LOG.debug(f"Type counts filled: {type_counts_filled}")
    LOG.debug(f"Type counts output: {type_counts_output}" )
    # Checks if the shapes are consisrent or inconsistent (if filled shape is equal to final shape we succeeded)
    LOG.debug(f"Input data shape: {input_data.shape}")
    LOG.debug(f"Filled data shape: {filled_data.shape}")
    LOG.debug(f"output data shape: {final_output.shape}")
    # Checks if the dates are consistent or inconsistent (if the dates are unified across all datasets we succeeded )
    LOG.debug(f"Mininmum filled date {min(filled_data['order_date'])}")
    LOG.debug(f"Maximum filled  date {max(filled_data['order_date'])}")
    LOG.debug(f"Mininmum output date {min(final_output['order_date'])}")
    LOG.debug(f"Maximum output date {max(final_output['order_date'])}")
    # Checks if all sku warehouses have the same number of time points (450 min and max means we succeeded )
    LOG.debug(f"Number of points per sku,warehouse after filling min max {min(filled_data.groupby(['sku','warehouse_id']).count()['sales']) , max(filled_data.groupby(['sku','warehouse_id']).count()['sales'])}")
    LOG.debug(f"Number of points per sku,warehouse  after output min max {min(final_output.groupby(['sku','warehouse_id']).count()['sales']) , max(final_output.groupby(['sku','warehouse_id']).count()['sales'])}")

    # Checking column names and dtypes in input and output 
    LOG.debug(f"Columns input: {input_data.columns}")
    LOG.debug(f"Columns filled: {filled_data.columns}")
    LOG.debug(f"Columns output: {final_output.columns}")

    #  Checks if filtering by sales is working properly
    percentage_zero_filled = filled_data.groupby(['sku','warehouse_id']).apply(percent_zero).reset_index(name='percent_zero')
    below_threshold = len(percentage_zero_filled[percentage_zero_filled['percent_zero']>90])
    percentage_zero_output = final_output.groupby(['sku','warehouse_id']).apply(percent_zero).reset_index(name='percent_zero')
    below_threshold_output = len(percentage_zero_filled[percentage_zero_filled['percent_zero']>90])
    LOG.info(f"Points below threshold in filled: {below_threshold}")
    LOG.info(f"Points below threshold in output: {below_threshold_output}")
    LOG.info(f" data dtypes: {final_output.dtypes}")


def fill_missing_days(df, start_date, end_date, threshold=0.2):

    fill_forward_columns = [
        'order_item_price_lc',
        'parent_category',
        'master_category',
        'country_code',
        'sku',
        'warehouse_id'
    ]
    fill_zero_columns = ['sales']

    # Generate a complete date range for this group
    date_range = pd.date_range(start=start_date, end=end_date)
    df = df.reindex(date_range)
    df[fill_forward_columns] = df[fill_forward_columns].ffill()
    df[fill_forward_columns] = df[fill_forward_columns].bfill()

    for col in fill_zero_columns:
        df[col] = df[col].fillna(0)
    
    df = df.reset_index().rename(columns={'index': 'order_date'})
    df = df.reset_index().rename(columns={'index': 'time_idx'})

    
    non_zero_count = (df['sales'] != 0).sum()
    percentage_non_zero = (non_zero_count / len(df))

    if percentage_non_zero >= threshold:

        return df
    else:
        return
    
def add_time_features(df):
    
    df['day_name'] = df['order_date'].dt.day_name()
    df['day_of_week'] = df['order_date'].dt.dayofweek
    df['week_number'] = df['order_date'].dt.isocalendar().week
    df['month'] = df['order_date'].dt.month
    df['day_number'] = df['order_date'].dt.day
    
    df['month'] = df['month'].astype(str).astype("category")
    df['week_number'] = df['week_number'].astype(str).astype("category")
    df['day_number'] = df['day_number'].astype(str).astype("category")
    df['day_of_week'] =df['day_of_week'].astype(str).astype("category")
    return df

def write_holidays_query():
    Holidays_query=f'''

    SELECT * FROM `tlb-data-dev.data_platform_sku_forecasting.country_holiday` 
    WHERE Country='KW'

    '''
    return Holidays_query

def metrics_evaluation(actual, predicted):
    metrics_dict = {}
    actual = actual.flatten()
    predicted = predicted.flatten()
    # predicted[np.abs(predicted) < 0.1] = 0

    metrics_dict['mape'] = np.mean(np.abs(actual - predicted) / actual) * 100
    metrics_dict['mdape'] = np.median(np.abs(predicted - actual) / np.maximum(actual, 1)) * 100

    # success metric: percentage of predictions within 10% of actual
    diffs = np.abs(predicted - actual)
    errors = np.divide(diffs, actual, out=np.full_like(diffs, np.inf), where=actual != 0)
    # Apply condition 1: if diffs <= 1, errors should be 0
    errors[diffs <= 1] = 0
    # Apply condition 2: if actual = 0 and diffs > 1, errors should be 1 (included)
    errors[(actual == 0) & (diffs > 1)] = 1
    metrics_dict['success_metric'] = np.round(np.mean(errors <= 0.1) * 100, 200)

    return metrics_dict

