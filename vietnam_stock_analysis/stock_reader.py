from datetime import datetime, timedelta
from typing import List, Optional, Tuple

import investpy
import pandas as pd
import pytz
from pandas.core.frame import DataFrame


def get_current_date_string(
        days_left: int = None, from_date: str = None, to_date: str = None
) -> Optional[Tuple[str, str]]:
    if days_left:
        now = datetime.now(pytz.utc)
        to_date = now.strftime("%d/%m/%Y")
        from_date = (now - timedelta(days_left)).strftime("%d/%m/%Y")

    elif not from_date or not to_date:
        raise ValueError("You must provided from_date AND to_date")

    return from_date, to_date


class TickerDataReader:
    def __init__(
            self, name: str, *, days_from_now: int = None, from_date: str = None, to_date: str = None
    ) -> None:
        self.name = name.upper()

        if not (days_from_now or from_date or to_date):
            self.from_date, self.to_date = get_current_date_string(days_left=30)

        else:
            self.from_date, self.to_date = get_current_date_string(
                days_from_now, from_date, to_date
            )

    @property
    def recent_data(self) -> DataFrame:
        df = investpy.get_stock_recent_data(stock=self.name, country="vietnam")
        df["name"] = self.name
        return df

    @property
    def historical_data(self) -> DataFrame:
        df = investpy.get_stock_historical_data(
            stock=self.name, country="vietnam", from_date=self.from_date, to_date=self.to_date
        )
        df["name"] = self.name
        return df

    def __repr__(self) -> str:
        return f"{self.name} {[self.from_date, self.to_date]}"


class TickerGroupDataReader:
    def __init__(
            self,
            ticker_group: List[str],
            *,
            days_from_now: int = None,
            from_date: str = None,
            to_date: str = None,
    ) -> None:
        self.ticker_group = [ticker.upper() for ticker in ticker_group]

        if not (days_from_now or from_date or to_date):
            self.from_date, self.to_date = get_current_date_string(days_left=30)

        else:
            self.from_date, self.to_date = get_current_date_string(
                days_from_now, from_date, to_date
            )

    @property
    def raw_data(self) -> DataFrame:
        group_df = pd.DataFrame()

        for ticker_name in self.ticker_group:
            ticker_df = TickerDataReader(
                name=ticker_name, from_date=self.from_date, to_date=self.to_date
            ).historical_data

            group_df = group_df.append(ticker_df, sort=True)

        return group_df

    @property
    def summary_info(self) -> DataFrame:
        return self.raw_data.groupby("name").describe().T

    @property
    def portfolio(self) -> DataFrame:
        return self.raw_data.reset_index().groupby("Date").sum()

    def __repr__(self) -> str:
        return f"{self.ticker_group} - {[self.from_date, self.to_date]}"
