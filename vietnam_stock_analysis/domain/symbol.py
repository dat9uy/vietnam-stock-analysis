from typing import Tuple
from pydantic import BaseModel
from enum import Enum

from pandas.core.frame import DataFrame


class SymbolType(str, Enum):
    STOCK = "stock"
    INDEX = "index"


class Symbol(BaseModel):
    symbol: str
    type: SymbolType


class SymbolRecords(Symbol):
    date: Tuple[str, str]
    data: DataFrame

    def __repr__(self):
        return f"{self.symbol} - {self.type} | {self.date}"

    class Config:
        arbitrary_types_allowed = True


class StockRecords(SymbolRecords):
    type = SymbolType.STOCK
