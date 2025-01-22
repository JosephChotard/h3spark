import os
import sys
import unittest

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

import src.h3spark as h3spark
import src.h3spark.convenience as h3spark_c

from .test_udf import polygon

os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable


class ConvienienceOpTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spark = SparkSession.builder.getOrCreate()
        hexes_15 = [
            644549256467709952,
            644543985138668136,
            644507327923224576,
            645738780946333696,
            647779474527485952,
        ]
        cls.masterDf = cls.spark.createDataFrame(
            [
                {
                    "h3_int_15": hex,
                }
                for hex in hexes_15
            ]
        )

    @classmethod
    def tearDownClass(cls):
        cls.spark.stop()

    def get_df(self):
        return self.masterDf

    def test_cover_and_compact(self):
        test_df = self.get_df()
        test_df = test_df.withColumn("geometry", F.lit(polygon))
        test_df = test_df.withColumn(
            "compacted_length",
            F.size(h3spark_c.h3shape_to_cells_compacted(F.col("geometry"), F.lit(15))),
        ).withColumn(
            "uncompacted_length",
            F.size(h3spark.h3shape_to_cells(F.col("geometry"), F.lit(15))),
        )

        results = test_df.collect()
        assert results[0]["compacted_length"] == 122
        assert results[0]["uncompacted_length"] == 554


if __name__ == "__main__":
    unittest.main()
