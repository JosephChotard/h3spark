import os
import sys
import unittest

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

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
                    "h3_int": 599513261267746800,
                    "h3": "851e6227fffffff",
                    "resolution": 5,
                },
                {
                    "h3_int": 640040385511297647,
                    "h3": "8e1e156ec4e126f",
                    "resolution": 14,
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


if __name__ == "__main__":
    unittest.main()
