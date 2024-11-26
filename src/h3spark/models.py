from dataclasses import dataclass

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


H3_CELL_DECIMAL_TYPE = T.DecimalType(20, 0)
