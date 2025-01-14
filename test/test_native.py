import os
import sys
import unittest

import h3.api.numpy_int as h3
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

import src.h3spark as h3spark
import src.h3spark.native as h3spark_n

os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable


class NativeOpTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spark = SparkSession.builder.getOrCreate()
        cls.masterDf = cls.spark.createDataFrame(
            [
                {
                    "h3_int": 599513261267746815,
                    "h3": "851e6227fffffff",
                    "resolution": 5,
                    "h3_int_2": 586002863965077503,
                    "h3_int_14": 640045656840339463,
                    "h3_int_15": 644549256467709952,
                },
                {
                    "h3_int": 640040385511297647,
                    "h3": "8e1e156ec4e126f",
                    "resolution": 14,
                    "h3_int_2": 585997366406938623,
                    "h3_int_14": 640040385511297647,
                    "h3_int_15": 644543985138668136,
                },
                {
                    "h3_int": 585961082523222015,
                    "h3": "821c07fffffffff",
                    "resolution": 2,
                    "h3_int_2": 585961082523222015,
                    "h3_int_14": 640003728295854087,
                    "h3_int_15": 644507327923224576,
                },
            ]
        )

    @classmethod
    def tearDownClass(cls):
        cls.spark.stop()

    def get_df(self):
        return self.masterDf

    def test_get_resolution(self):
        test_df = self.get_df()
        test_df = test_df.withColumn(
            "result", h3spark_n.get_resolution(F.col("h3_int"))
        )
        results = test_df.collect()
        for res in results:
            self.assertEqual(res["result"], res["resolution"])

    def test_get_fixed_parent_one(self):
        test_df = self.get_df()
        test_df = test_df.withColumn(
            "result", h3spark_n.go_to_parent_fixed(F.col("h3_int_15"), 15, 14)
        )
        results = test_df.collect()
        for res in results:
            self.assertEqual(res["result"], res["h3_int_14"])

    def test_get_fixed_parent_many(self):
        test_df = self.get_df()
        test_df = test_df.withColumn(
            "result", h3spark_n.go_to_parent_fixed(F.col("h3_int_15"), 15, 2)
        )
        results = test_df.collect()
        for res in results:
            self.assertEqual(res["result"], res["h3_int_2"])


if __name__ == "__main__":
    unittest.main()
