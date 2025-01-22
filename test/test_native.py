import os
import sys
import unittest

import h3.api.numpy_int as h3
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

import src.h3spark.native as h3spark_n

os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable


class NativeOpTests(unittest.TestCase):
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
                    "h3_int_15_str": h3.int_to_str(hex),
                    "h3_int_14": h3.cell_to_parent(hex, 14),
                    "h3_int_2": h3.cell_to_parent(hex, 2),
                    "h3_base_cell": h3.get_base_cell_number(hex),
                    "is_pentagon": h3.is_pentagon(hex),
                }
                for hex in hexes_15
            ]
        )

    @classmethod
    def tearDownClass(cls):
        cls.spark.stop()

    def get_df(self):
        return self.masterDf

    def test_get_resolution(self):
        test_df = self.get_df()
        test_df = (
            test_df.withColumn(
                "result_15", h3spark_n.get_resolution(F.col("h3_int_15"))
            )
            .withColumn("result_14", h3spark_n.get_resolution(F.col("h3_int_14")))
            .withColumn("result_2", h3spark_n.get_resolution(F.col("h3_int_2")))
        )
        results = test_df.collect()
        for res in results:
            self.assertEqual(res["result_15"], 15)
            self.assertEqual(res["result_14"], 14)
            self.assertEqual(res["result_2"], 2)

    def test_get_fixed_parent_one(self):
        test_df = self.get_df()
        test_df = test_df.withColumn(
            "result", h3spark_n.cell_to_parent_fixed(F.col("h3_int_15"), 15, 14)
        )
        results = test_df.collect()
        for res in results:
            self.assertEqual(res["result"], res["h3_int_14"])

    def test_get_fixed_parent_many(self):
        test_df = self.get_df()
        test_df = test_df.withColumn(
            "result", h3spark_n.cell_to_parent_fixed(F.col("h3_int_15"), 15, 2)
        )
        results = test_df.collect()
        for res in results:
            self.assertEqual(res["result"], res["h3_int_2"])

    def test_has_child_at_resolution(self):
        test_df = self.get_df()
        test_df = (
            test_df.withColumn(
                "result_1", h3spark_n.has_child_at_resolution(F.col("h3_int_15"), 3)
            )
            .withColumn(
                "result_2", h3spark_n.has_child_at_resolution(F.col("h3_int_14"), 1)
            )
            .withColumn(
                "result_3", h3spark_n.has_child_at_resolution(F.col("h3_int_2"), 13)
            )
        )
        results = test_df.collect()
        for res in results:
            self.assertEqual(res["result_1"], False)
            self.assertEqual(res["result_2"], False)
            self.assertEqual(res["result_3"], True)

    def test_get_base_cell(self):
        test_df = self.get_df()
        test_df = test_df.withColumn(
            "result", h3spark_n.get_base_cell(F.col("h3_int_14"))
        )
        results = test_df.collect()
        for res in results:
            self.assertEqual(res["result"], res["h3_base_cell"])

    def test_is_pentagon(self):
        test_df = self.get_df()
        test_df = test_df.withColumn(
            "result", h3spark_n.is_pentagon(F.col("h3_int_15"))
        )
        results = test_df.collect()
        for res in results:
            self.assertEqual(res["result"], res["is_pentagon"])

    def test_cell_to_children_size(self):
        test_df = self.get_df()
        test_df = (
            test_df.withColumn(
                "result_14_15",
                h3spark_n.cell_to_children_size(F.col("h3_int_14"), F.lit(15)),
            )
            .withColumn(
                "result_14_14",
                h3spark_n.cell_to_children_size(F.col("h3_int_14"), F.lit(14)),
            )
            .withColumn(
                "result_2_15",
                h3spark_n.cell_to_children_size(F.col("h3_int_2"), F.lit(15)),
            )
            .withColumn(
                "result_2_4",
                h3spark_n.cell_to_children_size(F.col("h3_int_2"), F.lit(4)),
            )
        )
        results = test_df.collect()
        for res in results:
            self.assertEqual(
                res["result_14_15"], h3.cell_to_children_size(res["h3_int_14"], 15)
            )
            self.assertEqual(
                res["result_14_14"], h3.cell_to_children_size(res["h3_int_14"], 14)
            )
            self.assertEqual(
                res["result_2_15"], h3.cell_to_children_size(res["h3_int_2"], 15)
            )
            self.assertEqual(
                res["result_2_4"], h3.cell_to_children_size(res["h3_int_2"], 4)
            )

    def test_cell_to_children_size_throws_invalid(self):
        test_df = self.get_df()
        test_df = test_df.withColumn(
            "result_14_13",
            h3spark_n.cell_to_children_size(F.col("h3_int_14"), F.lit(13), True),
        )
        with self.assertRaises(Exception):
            test_df.collect()

    def test_str_to_int(self):
        test_df = self.get_df()
        test_df = test_df.withColumn(
            "result",
            h3spark_n.str_to_int(F.col("h3_int_15_str")),
        )
        results = test_df.collect()
        for res in results:
            self.assertEqual(res["result"], res["h3_int_15"])

    def test_int_to_str(self):
        test_df = self.get_df()
        test_df = test_df.withColumn(
            "result",
            h3spark_n.int_to_str(F.col("h3_int_15")),
        )
        results = test_df.collect()
        for res in results:
            self.assertEqual(res["result"], res["h3_int_15_str"])


if __name__ == "__main__":
    unittest.main()
