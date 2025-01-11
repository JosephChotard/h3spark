import os
import sys
import unittest

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

import src.h3spark as h3spark

os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

latitude = 30.76973533630371
longitude = -91.45850372314453
polygon = """
{
    "type": "Polygon",
    "coordinates": [
        [
            [
                -89.998354,
                29.8988
            ],
            [
                -89.99807,
                29.8988
            ],
            [
                -89.99807,
                29.898628
            ],
            [
                -89.998354,
                29.898628
            ],
            [
                -89.998354,
                29.8988
            ]
        ]
    ]
}"""
h3_cell = "81447ffffffffff"
h3_cell_int = 582169416674836479

test_arg_map = {
    "lat": latitude,
    "lng": longitude,
    "h": h3_cell,
    "h_int": h3_cell_int,
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
        self.assertEqual(results[0]["result"], h3_cell_int)

    def test_int_to_str(self):
        test_df = self.get_df()
        test_df = test_df.withColumn("result", h3spark.int_to_str(F.col("h_int")))
        results = test_df.collect()
        self.assertEqual(results[0]["result"], "81447ffffffffff")

    def test_latlng_to_cell(self):
        test_df = self.get_df()
        test_df = test_df.withColumn(
            "result",
            h3spark.latlng_to_cell(F.lit(latitude), F.lit(longitude), F.lit(1)),
        )
        results = test_df.collect()
        self.assertEqual(results[0]["result"], h3_cell_int)

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
                    636208648398676095,
                    636208648398676223,
                    636208648398676671,
                    636208648398676927,
                    636208648398677247,
                    636208648398677567,
                    636208648398677631,
                    636208648398677695,
                    636208648398677759,
                    636208648398677823,
                    636208648398677887,
                    636208648398677951,
                ]
            ),
        )

    def test_local_ij_to_cell(self):
        test_df = self.get_df()
        test_df = test_df.withColumn(
            "result",
            h3spark.local_ij_to_cell(F.lit("85283473fffffff"), F.lit(0), F.lit(0)),
        )
        results = test_df.collect()
        self.assertEqual(results[0]["result"], 599682438955794431)

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
                    581641651093503999,
                    581650447186526207,
                    582165018628325375,
                    582169416674836479,
                    582178212767858687,
                    582187008860880895,
                    582244183465525247,
                ]
            ),
        )


if __name__ == "__main__":
    unittest.main()
