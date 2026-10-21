from dataclasses import dataclass
from enum import Enum

from pyspark.sql import types as T


@dataclass
class Coordinates:
    lat: float
    lon: float

    def from_tuple(self, tuple):
        return Coordinates(lat=tuple[0], lon=tuple[1])


COORDINATE_TYPE = T.StructType(
    [
        T.StructField("lat", T.FloatType(), False),
        T.StructField("lon", T.FloatType(), False),
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
