import pytest
from unittest import mock
from src.data import fetch
from datetime import datetime, timedelta


@pytest.fixture
def date_start():
    return (
            datetime.today() - timedelta(60)
    ).strftime('%Y-%m-%d')


@pytest.fixture
def get_fetch_search_object():
    """Text fixture of get zendesk client"""
    with mock.patch.object(
            fetch,
            "FetchQueriesPerformances",
    ) as patched:
        yield patched


def test_fetch_search_perf_class(
        get_fetch_search_object,
        date_start):
    """Test for get search get_by_country/city"""
    fetch_instance = fetch.FetchQueriesPerformances("search_query")
    fetch_instance.get(
        {
            'date_start': date_start,
            'os': 'Android',
            'language': 'english',
            'search_type': 'item_search',
            'country': 'Bahrain',
            'meta': {}
        }
    )
    get_fetch_search_object.assert_called_once()
