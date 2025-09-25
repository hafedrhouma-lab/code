from datetime import datetime

from timeseries_forecast_tool._helpers import DatesIterator


def test_date_iterator():
    n_days = 100
    dates_iterator = DatesIterator(datetime(2022, 2, 21), n_days)
    iterable_dates = iter(dates_iterator)
    first_date = next(iterable_dates)
    assert first_date == datetime(2022, 2, 22)

    date_value = None
    for date_value in iterable_dates:
        continue
    assert date_value == datetime(2022, 6, 1)
