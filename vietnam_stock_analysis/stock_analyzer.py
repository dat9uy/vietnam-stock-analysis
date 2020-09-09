import math
from typing import Any, Dict, Optional

import numpy as np
from pandas.core.frame import DataFrame
from pandas.core.series import Series


class StockAnalyzer:
    """Tính toán các chỉ số tài chính cho phân tích kỹ thuật"""

    def __init__(self, df: DataFrame) -> None:
        self._data = df

    @property
    def _max_periods(self) -> int:
        """Số ngày tối đa sử dụng trong tính toán"""
        return self._data.shape[0]

    @property
    def close(self) -> Series:
        """Giá đóng cửa"""
        return self._data["Close"]

    @property
    def close_pct_change(self) -> Series:
        """Thay đổi phần trăm giá cuối ngày"""
        return self.close.pct_change()

    @property
    def last_close(self) -> float:
        """Giá đóng cửa ngày cuối cùng"""
        return self._data.last("1D")["Close"].iat[0]

    @property
    def last_high(self) -> float:
        """Giá cao nhất ngày cuối cùng"""
        return self._data.last("1D")["High"].iat[0]

    @property
    def last_low(self):
        """Giá thấp nhất ngày cuối cùng"""
        return self._data.last("1D")["Low"].iat[0]

    @property
    def pivot_point(self):
        """Tính pivot point để tìm mức hỗ trợ/kháng cự"""
        return (self.last_close + self.last_high + self.last_low) / 3

    @staticmethod
    def port_return(df: DataFrame) -> float:
        """Lợi nhuận thu được"""
        start, end = df["Close"][0], df["Close"][-1]
        return (end - start) / start

    def cummulative_returns(self) -> Series:
        """Lợi nhuận lũy kế"""
        return np.add(self.close_pct_change, 1).cumprod()

    def is_bear_market(self) -> bool:
        """Giả định bear market khi lãi suất trong 2 tháng gần nhất giảm 20% hoặc hơn"""
        return self.port_return(self._data.last("2M")) <= -0.2

    def is_bull_market(self) -> bool:
        """Giả định bull market khi lãi suất trong 2 tháng gần nhất tăng 20% hoặc hơn"""
        return self.port_return(self._data.last("2M")) >= 0.2

    def resistance(self, level: int = 1) -> Optional[float]:
        """Mức kháng cự"""
        if level == 1:
            res = (2 * self.pivot_point) - self.last_low
        elif level == 2:
            res = self.pivot_point + (self.last_high - self.last_low)
        elif level == 3:
            res = self.last_high + 2 * (self.pivot_point - self.last_low)
        else:
            raise ValueError("Not a valid level. Must be 1, 2, or 3")

        return res

    def support(self, level: int = 1) -> Optional[float]:
        """Mức hỗ trợ"""
        if level == 1:
            sup = (2 * self.pivot_point) - self.last_high
        elif level == 2:
            sup = self.pivot_point - (self.last_high - self.last_low)
        elif level == 3:
            sup = self.last_low - 2 * (self.last_high - self.pivot_point)
        else:
            raise ValueError("Not a valid level. Must be 1, 2, or 3")

        return sup

    def daily_std(self, periods: int = 252) -> float:
        """Tính độ lệch chuẩn tỉ lệ phần trăm giá đóng cửa

        Parameters
        ----------
        periods : int, optional
            Số ngày dùng để tính toán, mặc định là 252 (số tối đa) tương ứng với 1 năm

        Returns
        -------
        float
            Độ lệch chuẩn
        """
        return self.close_pct_change[min(periods, self._max_periods) * -1 :].std()

    def annualized_volatility(self):
        """Biến động năm"""
        return self.daily_std() * math.sqrt(252)

    def volatility(self, periods=252) -> Series:
        """Biến động"""
        periods = min(periods, self._max_periods)

        return self.close.rolling(periods).std() / math.sqrt(periods)

    def corr_with(self, other: DataFrame) -> Series:
        """Tính hệ số tương quan"""
        return self._data.corrwith(other)

    def cv(self) -> float:
        """coefficient of variation"""
        return self.close.std() / self.close.mean()

    def qcd(self) -> float:
        """quantile coefficient of dispersion"""
        q1, q3 = self.close.quantile([0.25, 0.75])
        return (q3 - q1) / (q3 + q1)

    def beta(self, index: DataFrame) -> float:
        """Tính hệ số beta

        Parameters
        ----------
        index : DataFrame
            DataFrame của index

        Returns
        -------
        float
            Hệ số beta
        """
        index_change = index["Close"].pct_change()
        beta = self.close_pct_change.cov(index_change) / index_change.var()

        return beta

    def alpha(self, index: DataFrame, r_f: float):
        """Tính hệ số alpha

        Parameters
        ----------
        index : DataFrame
            DataFrame của index
        r_f : float
            Lãi suất phi rủi ro (The risk free rate of return)

        Returns
        -------
        float
            Hệ số alpha
        """
        r_f /= 100
        r_m = self.port_return(index)
        beta = self.beta(index)
        r = self.port_return(self._data)

        alpha = r - r_f - beta * (r_m - r_f)

        return alpha

    def sharpe_ratio(self, r_f: float) -> float:
        """Tính sharpe ratio

        Parameters
        ----------
        r_f : float
            Lãi suất phi rủi ro (The risk free rate of return)

        Returns
        -------
        float
            Sharpe Ratio
        """
        return (
            self.cummulative_returns().last("1D").iat[0] - r_f
        ) / self.cummulative_returns().std()


class AssetGroupAnalyzer:
    """Phân tích cổ phiếu theo nhóm"""

    def __init__(self, df: DataFrame, group_col: str = "name") -> None:
        """Gộp nhóm cổ phiếu để phân tích"""
        self._data = df

        if group_col not in self._data.columns:
            raise ValueError(f"`group_col` column {group_col} not in dataframe")

        self.group_col = group_col
        self._analyzers = self._composition_handler

    @property
    def _composition_handler(self) -> Dict[str, StockAnalyzer]:
        """Tạo một dict map từng mã cổ phiếu vào df phân tích tương ứng"""
        return {
            group: StockAnalyzer(ticker_df)
            for group, ticker_df in self._data.groupby(self.group_col)
        }

    def analyze(self, func_name: str, **kwargs) -> Dict[str, Any]:
        """Chạy function của StockAnalyzer trên cả nhóm

        Parameters
        ----------
        func_name : str
            Tên function của StockAnalyzer

        Returns
        -------
        Dict[str, Any]
            Một dict map mỗi mã vào kết quả tính toán từ function

        Raises
        ------
        ValueError
            Khi nhập function không có trong StockAnalyzer
        """
        if not hasattr(StockAnalyzer, func_name):
            raise ValueError(f"StockAnalyzer has no {func_name} method.")
        if not kwargs:
            kwargs = {}

        return {
            group: getattr(StockAnalyzer, func_name)(analyzer, **kwargs)
            for group, analyzer in self._analyzers.items()
        }
