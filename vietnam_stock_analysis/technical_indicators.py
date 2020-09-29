from abc import ABC, abstractmethod

import plotly.graph_objs as go
from pandas.core.frame import DataFrame
from pandas.core.series import Series

from vietnam_stock_analysis.domain.symbol import StockRecords


class BaseIndicator(ABC):
    """
    Mỗi Indicator bao gồm result là kết quả tính và plot là đồ thị
    """

    def __init__(self, stock_records: StockRecords):
        self._data_obj = stock_records
        self._df = stock_records.data
        self.name = stock_records.symbol
        super(BaseIndicator, self).__init__()

    @property
    @abstractmethod
    def result(self):
        pass

    @abstractmethod
    def plot(self, **kwargs):
        pass


class Candlestick(BaseIndicator):
    @property
    def result(self) -> DataFrame:
        return self._df

    def plot(self, **kwargs):
        return go.Candlestick(
            x=self._df.index,
            open=self._df["Open"],
            close=self._df["Close"],
            high=self._df["High"],
            low=self._df["Low"],
            name=f"{self.name} Price",
            **kwargs,
        )


class MA(BaseIndicator):
    def __init__(self, stock_records: StockRecords, moving_days: int = 21):
        super().__init__(stock_records)
        self.moving_days = moving_days

    @property
    def result(self) -> Series:
        if len(self._df) < 2 * self.moving_days:
            raise ValueError("Data too short")

        return self._df["Close"].rolling(window=self.moving_days).mean()

    def plot(self, **kwargs):
        return go.Scatter(
            x=self.result.index,
            y=self.result,
            name=f"{self.name}_MA({self.moving_days})",
            **kwargs,
        )


class EMA(BaseIndicator):
    # def __init__(self, df: DataFrame, moving_days: int = 21):
    #     super().__init__(df)
    #     self.moving_days = moving_days
    def __init__(self, stock_records: StockRecords, moving_days: int = 21):
        super().__init__(stock_records)
        self.moving_days = moving_days

    @property
    def result(self) -> Series:
        if len(self._df) < 2 * self.moving_days:
            raise ValueError("Data too short")

        return (
            self._df["Close"]
                .ewm(span=self.moving_days, min_periods=self.moving_days, adjust=False)
                .mean()
        )

    def plot(self, **kwargs):
        return go.Scatter(
            x=self.result.index,
            y=self.result,
            name=f"{self.name}_EMA({self.moving_days})",
            **kwargs,
        )


class MACD(BaseIndicator):
    def __init__(self, stock_records: StockRecords, low: int = 12, high: int = 24, exp: int = 0):
        super().__init__(stock_records)
        self.low = low
        self.high = high
        self.exp = exp

    @property
    def result(self) -> Series:
        macd: Series = (
                EMA(self._data_obj, moving_days=self.low).result
                - EMA(self._data_obj, moving_days=self.high).result
        )
        if self.exp == 0:
            return macd
        else:
            return macd.ewm(span=self.exp, adjust=False).mean()

    def plot(self, **kwargs):
        return go.Scatter(
            x=self.result.index, y=self.result, name=f"{self.name}_MACD({self.exp})", **kwargs
        )
