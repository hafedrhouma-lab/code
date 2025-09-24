from datetime import datetime, timedelta


def get_date_range(evaluation_date, days_back):
    """Return start and end date strings for a range ending at evaluation_date."""
    date_start_obj = datetime.strptime(evaluation_date, '%Y-%m-%d')
    end_date = date_start_obj - timedelta(days=1)
    start_date = date_start_obj - timedelta(days=days_back)
    return start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')
