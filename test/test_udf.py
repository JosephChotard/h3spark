import os
import sys
import unittest
from decimal import Decimal

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

import src.h3spark as h3spark

os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

latitude = 30.76973533630371
longitude = -91.45850372314453
integer = 1
double = 0.5
point = '{"type": "Point", "coordinates": [-89.998354, 29.8988]}'
line = '{"type": "LineString", "coordinates": [[-89.99927146300001, 29.90139583899997], [-89.99921418299999, 29.90139420899999], [-89.99903129900002, 29.90138951699998], [-89.99900807, 29.90142210300002], [-89.99898608000001, 29.90138835699997], [-89.99875118300002, 29.90138410499998], [-89.99872961, 29.90141686999999], [-89.99871085699999, 29.90138346399999], [-89.99837947499998, 29.90137720600001], [-89.99835869700001, 29.90140975100002], [-89.99834035200001, 29.901376191], [-89.998234115, 29.90137350700002], [-89.998218017, 29.90137313499997], [-89.99819830400003, 29.90137344499999], [-89.99787396300002, 29.90139402699998], [-89.99785696700002, 29.90142557899998], [-89.99783514199999, 29.90139429700002]]}'
polygon = '{"type": "Polygon", "coordinates": [[[-89.998354, 29.8988], [-89.99807, 29.8988], [-89.99807, 29.898628], [-89.998354, 29.898628], [-89.998354, 29.8988]]]}'
h3_cells = ["81447ffffffffff", "81267ffffffffff", "8148bffffffffff", "81483ffffffffff"]
h3_edge = "131447ffffffffff"
unit = "km^2"

test_arg_map = {
    "i": integer,
    "j": integer,
    "k": integer,
    "x": integer,
    "resolution": integer,
    "res": integer,
    "lat": latitude,
    "lng": longitude,
    "point1": (latitude, longitude),
    "point2": (latitude, longitude),
    "h": h3_cells[0],
    "h_int": Decimal(582169416674836479),
    "hexes": h3_cells,
    "h1": h3_cells[1],
    "h2": h3_cells[2],
    "origin": h3_cells[2],
    "destination": h3_cells[3],
    "start": h3_cells[1],
    "end": h3_cells[2],
    "e": h3_edge,
    "edge": h3_edge,
    "geo_json": True,
    "geo_json_conformant": True,
    "geojson": polygon,
}


class MyUDFTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spark = SparkSession.builder.getOrCreate()
        cls.masterDf = cls.spark.createDataFrame([test_arg_map])

    @classmethod
    def tearDownClass(cls):
        cls.spark.stop()

    def get_df(self):
        return self.masterDf

    def test_str_to_int(self):
        test_df = self.get_df()
        test_df = test_df.withColumn("result", h3spark.str_to_int(F.col("h")))
        results = test_df.collect()
        self.assertEqual(results[0]["result"], Decimal(582169416674836479))

    def test_int_to_str(self):
        test_df = self.get_df()
        test_df = test_df.withColumn("result", h3spark.int_to_str(F.col("h_int")))
        results = test_df.collect()
        self.assertEqual(results[0]["result"], "81447ffffffffff")

    def test_get_num_cells(self):
        test_df = self.get_df()
        test_df = test_df.withColumn("result", h3spark.get_num_cells(F.lit(7)))
        results = test_df.collect()
        # Checked with https://h3geo.org/docs/core-library/restable
        self.assertEqual(results[0]["result"], 98_825_162)

    def test_average_hexagon_area(self):
        test_df = self.get_df()
        test_df = test_df.withColumn(
            "result",
            h3spark.average_hexagon_area(F.lit(7), F.lit(h3spark.AreaUnit.KM2.value)),
        )
        results = test_df.collect()
        # Checked with https://h3geo.org/docs/core-library/restable
        self.assertAlmostEqual(results[0]["result"], 5.161293360, 6)

    # Not going to test all the native h3 calls

    def test_cell_to_latlng_str(self):
        test_df = self.get_df()
        test_df = test_df.withColumn("result", h3spark.cell_to_latlng(F.col("h")))
        results = test_df.collect()
        self.assertEqual(
            results[0]["result"].asDict(),
            {"lat": latitude, "lon": longitude},
        )

    def test_cell_to_latlng_int(self):
        test_df = self.get_df()
        test_df = test_df.withColumn("result", h3spark.cell_to_latlng(F.col("h_int")))
        results = test_df.collect()
        self.assertEqual(
            results[0]["result"].asDict(),
            {"lat": latitude, "lon": longitude},
        )

    def test_latlng_to_cell(self):
        test_df = self.get_df()
        test_df = test_df.withColumn(
            "result",
            h3spark.latlng_to_cell(F.lit(latitude), F.lit(longitude), F.lit(1)),
        )
        results = test_df.collect()
        self.assertEqual(results[0]["result"], h3_cells[0])

    def test_cell_to_boundary(self):
        test_df = self.get_df()
        test_df = test_df.withColumn("result", h3spark.cell_to_boundary(F.col("h")))
        results = test_df.collect()
        self.assertEqual(
            [r.asDict() for r in results[0]["result"]],
            [
                {"lat": 26.426477432250977, "lon": -89.80770874023438},
                {"lat": 29.759286880493164, "lon": -86.5457763671875},
                {"lat": 34.059837341308594, "lon": -88.06588745117188},
                {"lat": 35.09010696411133, "lon": -93.26521301269531},
                {"lat": 31.619529724121094, "lon": -96.66288757324219},
                {"lat": 27.264005661010742, "lon": -94.74356842041016},
            ],
        )

    def test_grid_disk(self):
        test_df = self.get_df()
        test_df = test_df.withColumn("result", h3spark.grid_disk(F.col("h"), F.lit(1)))
        results = test_df.collect()
        self.assertEqual(
            sorted(results[0]["result"]),
            sorted(
                [
                    "81447ffffffffff",
                    "81443ffffffffff",
                    "8144fffffffffff",
                    "81267ffffffffff",
                    "8126fffffffffff",
                    "8148bffffffffff",
                    "81457ffffffffff",
                ]
            ),
        )

    def test_h3shape_to_cells(self):
        test_df = self.get_df()
        test_df = test_df.withColumn(
            "result", h3spark.h3shape_to_cells(F.lit(polygon), F.lit(13))
        )
        results = test_df.collect()
        self.assertEqual(
            sorted(results[0]["result"]),
            sorted(
                [
                    "8d444651ad5a07f",
                    "8d444651ad5a0ff",
                    "8d444651ad5a2bf",
                    "8d444651ad5a3bf",
                    "8d444651ad5a4ff",
                    "8d444651ad5a63f",
                    "8d444651ad5a67f",
                    "8d444651ad5a6bf",
                    "8d444651ad5a6ff",
                    "8d444651ad5a73f",
                    "8d444651ad5a77f",
                    "8d444651ad5a7bf",
                ]
            ),
        )


if __name__ == "__main__":
    unittest.main()
