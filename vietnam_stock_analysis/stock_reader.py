from datetime import datetime, timedelta
from typing import List, Optional, Tuple

import investpy
import pandas as pd
import pytz
from pandas.core.frame import DataFrame

from pandera import check_output

from ._dataframe_schema import stock_data_schema


def _get_date_string(
    days_from_now: int = None, from_date: str = None, to_date: str = None
) -> Optional[Tuple[str, str]]:
    """
    Function hỗ trợ lấy date dạng string dùng cho function.

    Trong thư viện investpy date được format ở dạng %d/%m/%Y.
    Function này hỗ trợ lấy date đúng format bằng cách bổ sung 2 tính năng
    so với mặc định:
        - Lấy date từ hiện tại lùi về days_from_now
        - Khi không nhập to_date thì mặc định là date hiện tại

    Nếu không nhập parameters thì mặc định lấy 30 ngày trở về trước
    tính từ thời điểm hiện tại

    Parameters
    ----------
    days_from_now : int, optional
        Số ngày tính từ thời điểm hiện tại trở về trước
    from_date : str, optional
        Thời điểm đầu
    to_date : str, optional
        Thời điểm cuối

    Returns
    -------
    Optional[Tuple[str, str]]
        Tuple gồm thời điểm đầu và cuối

    Raises
    ------
    ValueError
        Khi không nhập from_date
    """
    now = datetime.now(pytz.utc)

    if not (days_from_now or from_date or to_date):
        days_from_now = 30

    if days_from_now:
        to_date = now.strftime("%d/%m/%Y")
        from_date = (now - timedelta(days=days_from_now)).strftime("%d/%m/%Y")

    elif not from_date and to_date:
        raise ValueError("You must provided from_date")

    elif from_date and not to_date:
        to_date = now.strftime("%d/%m/%Y")

    return from_date, to_date


@check_output(stock_data_schema)
def ticker_data_reader(
    name: str, *, days_from_now: str = None, from_date: str = None, to_date: str = None
) -> DataFrame:  # pragma: no cover
    """Đọc dữ liệu ticker từ investing.com"""
    name = name.upper()
    from_date, to_date = _get_date_string(days_from_now, from_date, to_date)

    df = investpy.get_stock_historical_data(
        stock=name, country="vietnam", from_date=from_date, to_date=to_date
    )
    df["name"] = name

    return df


@check_output(stock_data_schema)
def ticker_group_data_reader(
    ticker_group: List[str], *, days_from_now: str = None, from_date: str = None, to_date=None
) -> DataFrame:  # pragma: no cover
    """Đọc dữ liệu ticker theo nhóm"""
    group_df = pd.DataFrame()

    for ticker_name in ticker_group:
        ticker_df = ticker_data_reader(
            name=ticker_name, days_from_now=days_from_now, from_date=from_date, to_date=to_date
        )

        group_df = group_df.append(ticker_df, sort=True)

    return group_df


class TickerGroup:  # pragma: no cover
    """Class gộp nhóm ticker và các tính toán tóm tắt cho nhóm"""

    def __init__(
        self,
        ticker_group: List[str],
        *,
        days_from_now: int = None,
        from_date: str = None,
        to_date: str = None,
    ) -> None:
        self.raw_data = ticker_group_data_reader(
            ticker_group=ticker_group,
            days_from_now=days_from_now,
            from_date=from_date,
            to_date=to_date,
        )

    @property
    def summary_info(self) -> DataFrame:
        return self.raw_data.groupby("name").describe().T

    @property
    def portfolio(self) -> DataFrame:
        return self.raw_data.reset_index().groupby("Date").sum()

    def __repr__(self) -> str:
        return (
            f"{self.raw_data['name'].unique().tolist()} - "
            f"({self.raw_data.index.min().strftime('%d/%m/%Y')} - "
            f"{self.raw_data.index.max().strftime('%d/%m/%Y')})"
        )
