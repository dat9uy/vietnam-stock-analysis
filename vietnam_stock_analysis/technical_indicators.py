from abc import ABC, abstractmethod

import plotly.graph_objs as go
from pandas.core.frame import DataFrame
from pandas.core.series import Series
from pandera.decorators import check_input

from ._dataframe_schema import stock_data_schema


class BaseIndicator(ABC):
    """
    Mỗi Indicator bao gồm result là kết quả tính và plot là đồ thị
    """
    @check_input(stock_data_schema)
    def __init__(self, df: DataFrame):
        self._data = df
        self.name = self._data["name"].unique()[0]
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
        return self._data

    def plot(self, **kwargs):
        return go.Candlestick(
            x=self._data.index,
            open=self._data["Open"],
            close=self._data["Close"],
            high=self._data["High"],
            low=self._data["Low"],
            name=f"{self.name} Price",
            **kwargs,
        )


class MA(BaseIndicator):
    def __init__(self, df: DataFrame, moving_days: int = 21):
        super().__init__(df)
        self.moving_days = moving_days

    @property
    def result(self) -> Series:
        if len(self._data) < 2 * self.moving_days:
            raise ValueError("Data too short")

        return self._data["Close"].rolling(window=self.moving_days).mean()

    def plot(self, **kwargs):
        return go.Scatter(
            x=self.result.index,
            y=self.result,
            name=f"{self.name}_MA({self.moving_days})",
            **kwargs,
        )


class EMA(BaseIndicator):
    def __init__(self, df: DataFrame, moving_days: int = 21):
        super().__init__(df)
        self.moving_days = moving_days

    @property
    def result(self) -> Series:
        if len(self._data) < 2 * self.moving_days:
            raise ValueError("Data too short")

        return (
            self._data["Close"]
            .ewm(span=self.moving_days, min_periods=self.moving_days, adjust=False)
            .mean()
        )

    def plot(self, **kwargs):
        return go.Scatter(
            x=self.result.index,
            y=self.result,
            name=f"{self._data['name'].unique()[0]}_EMA({self.moving_days})",
            **kwargs,
        )


class MACD(BaseIndicator):
    def __init__(self, df: DataFrame, low: int = 12, high: int = 24, exp: int = 0):
        super().__init__(df)
        self.low = low
        self.high = high
        self.exp = exp

    @property
    def result(self) -> Series:
        macd: Series = (
            EMA(self._data, moving_days=self.low).result
            - EMA(self._data, moving_days=self.high).result
        )
        if self.exp == 0:
            return macd
        else:
            return macd.ewm(span=self.exp, adjust=False).mean()

    def plot(self, **kwargs):
        return go.Scatter(
            x=self.result.index, y=self.result, name=f"{self.name}_MACD({self.exp})", **kwargs
        )
