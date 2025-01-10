# Test that the env variable can change udfs to use int or hex


import os
import sys
import unittest
from decimal import Decimal
from test.test_udf import polygon

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

import src.h3spark as h3spark

latitude = 30.76973533630371
longitude = -91.45850372314453


os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable


class IntUdfTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spark = SparkSession.builder.getOrCreate()
        cls.masterDf = cls.spark.createDataFrame([{"a": "b"}])

    @classmethod
    def tearDownClass(cls):
        cls.spark.stop()

    def get_df(self):
        return self.masterDf

    def test_latlng_to_cell(self):
        test_df = self.get_df()
        test_df = test_df.withColumn(
            "result",
            h3spark.latlng_to_cell_decimal(F.lit(latitude), F.lit(longitude), F.lit(1)),
        )
        results = test_df.collect()
        self.assertEqual(results[0]["result"], Decimal(582169416674836479))

    def test_h3shape_to_cells_decimal(self):
        test_df = self.get_df()
        test_df = test_df.withColumn(
            "result", h3spark.h3shape_to_cells_decimal(F.lit(polygon), F.lit(13))
        )
        results = test_df.collect()
        self.assertEqual(
            sorted(results[0]["result"]),
            sorted(
                [
                    Decimal("636208648398676095"),
                    Decimal("636208648398676223"),
                    Decimal("636208648398676671"),
                    Decimal("636208648398676927"),
                    Decimal("636208648398677247"),
                    Decimal("636208648398677567"),
                    Decimal("636208648398677631"),
                    Decimal("636208648398677695"),
                    Decimal("636208648398677759"),
                    Decimal("636208648398677823"),
                    Decimal("636208648398677887"),
                    Decimal("636208648398677951"),
                ]
            ),
        )

    def test_local_ij_to_cell(self):
        test_df = self.get_df()
        test_df = test_df.withColumn(
            "result",
            h3spark.local_ij_to_cell_decimal(
                F.lit("85283473fffffff"), F.lit(0), F.lit(0)
            ),
        )
        results = test_df.collect()
        self.assertEqual(results[0]["result"], Decimal("599682438955794431"))


if __name__ == "__main__":
    unittest.main()
