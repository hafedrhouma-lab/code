# import pytest
#
# from tests.apis.common import fetch_request
#
#
# @pytest.mark.parametrize("fetch_requests_paths", ["nba"], indirect=True)
# @pytest.mark.asyncio
# async def test_hero_banners(nba_client, fetch_requests_paths):
#     dirs = fetch_requests_paths
#     for directory in dirs:
#         request, response = fetch_request(directory)
#         api_response = await nba_client.get(
#             f"/home/v1/{request['country_code']}/customer/{request['customer_id']}/hero-banners"
#         )
#
#         assert api_response.status_code == 200
#         assert api_response.json()["variants"] == response["variants"]


# @pytest.mark.asyncio
# async def test_hero_banners_invalid_input(nba_client):
#     invalid_country_code = "invalid_country_code"
#     customer_id = 1
#     api_response = await nba_client.get(
#         f"/home/v1/{invalid_country_code}/customer/{customer_id}/hero-banners"
#     )
#
#     assert api_response.status_code == 400
