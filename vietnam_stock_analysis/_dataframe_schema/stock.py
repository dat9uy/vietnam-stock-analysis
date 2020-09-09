import pandera as pa
from pandera import Column, DataFrameSchema, Check, Index

stock_data_schema = DataFrameSchema(
    {
        "Open": Column(pa.Float),
        "Close": Column(pa.Float),
        "Low": Column(pa.Float),
        "High": Column(pa.Float),
        "Volume": Column(pa.Int),
        "Currency": Column(pa.String),
        "name": Column(pa.String),
    },
    index=Index(pa.DateTime, name="Date"),
    strict=True,
)
