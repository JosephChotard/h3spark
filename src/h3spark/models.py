from enum import Enum

from pyspark.sql import types as T

__all__ = ["AreaUnit", "BOUNDARY_TYPE", "COORDINATE_TYPE", "LengthUnit"]

COORDINATE_TYPE = T.StructType(
    [
        T.StructField("lat", T.DoubleType(), False),
        T.StructField("lon", T.DoubleType(), False),
    ]
)


BOUNDARY_TYPE = T.ArrayType(COORDINATE_TYPE)


class AreaUnit(Enum):
    KM2 = "km^2"
    M2 = "m^2"
    RADS2 = "rads^2"


class LengthUnit(Enum):
    KM = "km"
    M = "m"
    RADS = "rads"
