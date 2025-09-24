from datetime import datetime, timedelta


def get_date_range(end_date, days_back):
    """Return start and end date strings for a range ending at evaluation_date."""
    date_start_obj = datetime.strptime(end_date, '%Y-%m-%d')
    end_date = date_start_obj - timedelta(days=1)
    start_date = date_start_obj - timedelta(days=days_back)
    return start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')


def get_date_plus_days(start_date, days_forward):
    """Return a future date string with a range ending at evaluation_date."""
    date_start_obj = datetime.strptime(start_date, '%Y-%m-%d')
    future_date = date_start_obj + timedelta(days=days_forward)
    return future_date.strftime('%Y-%m-%d')


def get_date_minus_days(end_date, days_backward):
    """Return a future date string with a range ending at evaluation_date."""
    date_end_obj = datetime.strptime(end_date, '%Y-%m-%d')
    past_date = date_end_obj - timedelta(days=days_backward)
    return past_date.strftime('%Y-%m-%d')