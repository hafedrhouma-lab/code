# from datetime import datetime
#
# import pytest
# import pytz
#
# from nba.data import get_day_interval, time_of_day, COUNTRY_TIMEZONE, map_banners
#
#
# def test_time_of_day():
#     assert time_of_day(0) == "midnight"
#     assert time_of_day(5) == "midnight"
#     assert time_of_day(6) == "morning"
#     assert time_of_day(11) == "morning"
#     assert time_of_day(12) == "afternoon"
#     assert time_of_day(17) == "afternoon"
#     assert time_of_day(18) == "evening"
#     assert time_of_day(23) == "evening"
#     with pytest.raises(ValueError):
#         time_of_day(-1)


# def test_get_day_interval():
#     day_interval = get_day_interval("eg")
#     time_now = datetime.now().astimezone(COUNTRY_TIMEZONE.get(9, pytz.utc))
#     assert day_interval == time_of_day(time_now.hour)


# def test_map_banners():
#     assert map_banners([0, 1, 2, 3, 4]) == [
#         "tpro_sub",
#         "tpro_non_sub",
#         "food",
#         "tmart",
#         "grocery",
#     ]
#     assert map_banners([]) == []
#     assert map_banners(None) == []
