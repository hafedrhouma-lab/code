import polars as pl

from vendor_ranking.input import split_by_vendor_availability


def test_partition_vendors():
    data = {
        "chain_id": [679355, 1165, 33849, 680511],
        "vendor_id": [679355, 1165, 33849, 680511],
        "delivery_fee": [0.0, 0.0, 0.0, 0.0],
        "delivery_time": [31.0, 27.0, 26.0, 23.0],
        "vendor_rating": [3.0, 4.4, 0.0, 0.0],
        "status": ["0", "1", "0", "1"],
        "min_order_amount": [0.0, 0.0, 0.0, 0.0],
        "has_promotion": [False, False, False, False],
    }
    df = pl.DataFrame(data)
    available, busy = split_by_vendor_availability(df)

    assert available[0]["status"][0] == "0"
    assert busy[0]["status"][0] == "1"
