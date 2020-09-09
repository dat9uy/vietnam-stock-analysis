import pytest
import pytz
from datetime import datetime, timedelta
from vietnam_stock_analysis.stock_reader import _get_date_string


now = datetime.now(pytz.utc)


@pytest.mark.parametrize(
    "days_from_now, from_date, to_date, expected_date",
    [
        (None, None, None, ((now - timedelta(30)).strftime("%d/%m/%Y"), now.strftime("%d/%m/%Y"))),
        (6, None, None, ((now - timedelta(6)).strftime("%d/%m/%Y"), now.strftime("%d/%m/%Y"))),
        (None, "1/9/2020", None, ("1/9/2020", now.strftime("%d/%m/%Y"))),
    ],
)
def test_get_date_string_successful(
    days_from_now: int, from_date: str, to_date: str, expected_date: tuple
):
    date_tuple = _get_date_string(days_from_now, from_date, to_date)

    assert date_tuple == expected_date


@pytest.mark.parametrize("days_from_now, from_date, to_date", [(None, None, "1/9/2020")])
def test_get_date_string_without_from_date_failed(days_from_now: int, from_date: str, to_date: str):
    with pytest.raises(ValueError):
        _get_date_string(days_from_now, from_date, to_date)
