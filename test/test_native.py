import os
import sys
import unittest

import h3.api.numpy_int as h3
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

import src.h3spark.native as h3spark_n
from src.h3spark.models import AreaUnit, LengthUnit

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

    def test_cell_to_parent(self):
        test_df = self.get_df()
        test_df = (
            test_df.withColumn(
                "result_15_12", h3spark_n.cell_to_parent(F.col("h3_int_15"), 12)
            )
            .withColumn("result_15_1", h3spark_n.cell_to_parent(F.col("h3_int_15"), 1))
            .withColumn(
                "result_14_12", h3spark_n.cell_to_parent(F.col("h3_int_14"), 12)
            )
            .withColumn("result_2_0", h3spark_n.cell_to_parent(F.col("h3_int_2"), 0))
        )
        results = test_df.collect()
        for res in results:
            self.assertEqual(
                res["result_15_12"], h3.cell_to_parent(res["h3_int_15"], 12)
            )
            self.assertEqual(res["result_15_1"], h3.cell_to_parent(res["h3_int_15"], 1))
            self.assertEqual(
                res["result_14_12"], h3.cell_to_parent(res["h3_int_14"], 12)
            )
            self.assertEqual(res["result_2_0"], h3.cell_to_parent(res["h3_int_2"], 0))

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

    def test_has_child_at_max_resolution(self):
        test_df = self.get_df().withColumn(
            "result", h3spark_n.has_child_at_resolution(F.col("h3_int_14"), 15)
        )
        for res in test_df.collect():
            self.assertEqual(res["result"], True)

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

    def test_ischildof(self):
        test_df = self.get_df()
        test_df = (
            test_df.withColumn(
                "result_1", h3spark_n.is_childof(F.col("h3_int_14"), F.col("h3_int_15"))
            )
            .withColumn(
                "result_2", h3spark_n.is_childof(F.col("h3_int_2"), F.col("h3_int_15"))
            )
            .withColumn(
                "result_3", h3spark_n.is_childof(F.col("h3_int_15"), F.col("h3_int_15"))
            )
            .withColumn(
                "result_4", h3spark_n.is_childof(F.col("h3_int_15"), F.col("h3_int_2"))
            )
            .withColumn(
                "result_5", h3spark_n.is_childof(F.col("h3_int_15"), F.col("h3_int_14"))
            )
        )
        results = test_df.collect()
        for res in results:
            self.assertEqual(res["result_1"], False)
            self.assertEqual(res["result_2"], False)
            self.assertEqual(res["result_3"], True)
            self.assertEqual(res["result_4"], True)
            self.assertEqual(res["result_5"], True)

    def test_min_child(self):
        test_df = self.get_df()
        test_df = (
            test_df.withColumn(
                "result_14_15", h3spark_n.minchild(F.col("h3_int_14"), F.lit(15))
            )
            .withColumn(
                "result_15_15", h3spark_n.minchild(F.col("h3_int_15"), F.lit(15))
            )
            .withColumn("result_2_7", h3spark_n.minchild(F.col("h3_int_2"), F.lit(7)))
        )
        results = test_df.collect()
        for res in results:
            self.assertEqual(
                res["result_14_15"], h3.child_pos_to_cell(res["h3_int_14"], 15, 0)
            )
            self.assertEqual(
                res["result_15_15"], h3.child_pos_to_cell(res["h3_int_15"], 15, 0)
            )
            self.assertEqual(
                res["result_2_7"], h3.child_pos_to_cell(res["h3_int_2"], 7, 0)
            )

    def test_max_child(self):
        test_df = self.get_df()
        test_df = (
            test_df.withColumn(
                "result_14_15", h3spark_n.maxchild(F.col("h3_int_14"), F.lit(15))
            )
            .withColumn(
                "result_15_15", h3spark_n.maxchild(F.col("h3_int_15"), F.lit(15))
            )
            .withColumn("result_2_7", h3spark_n.maxchild(F.col("h3_int_2"), F.lit(7)))
        )
        results = test_df.collect()
        for res in results:
            self.assertEqual(
                res["result_14_15"],
                h3.child_pos_to_cell(
                    res["h3_int_14"],
                    15,
                    h3.cell_to_children_size(res["h3_int_14"], 15) - 1,
                ),
            )
            self.assertEqual(
                res["result_15_15"],
                h3.child_pos_to_cell(
                    res["h3_int_15"],
                    15,
                    h3.cell_to_children_size(res["h3_int_15"], 15) - 1,
                ),
            )
            self.assertEqual(
                res["result_2_7"],
                h3.child_pos_to_cell(
                    res["h3_int_2"], 7, h3.cell_to_children_size(res["h3_int_2"], 7) - 1
                ),
            )

    def get_mixed_df(self):
        """Hexagons and pentagons at several resolutions"""
        cells = [
            h3.latlng_to_cell(lat, lng, res)
            for lat, lng in [
                (0, 0),
                (37.77, -122.42),
                (-33.87, 151.21),
                (64.15, -21.94),
            ]
            for res in [0, 1, 5, 9, 14, 15]
        ] + [p for res in [0, 1, 5, 9, 14, 15] for p in h3.get_pentagons(res).tolist()]
        return self.spark.createDataFrame([(int(c),) for c in cells], ["cell"])

    def test_is_pentagon_mixed(self):
        test_df = self.get_mixed_df().withColumn(
            "result", h3spark_n.is_pentagon(F.col("cell"))
        )
        for res in test_df.collect():
            self.assertEqual(res["result"], h3.is_pentagon(res["cell"]))

    def test_cell_to_center_child(self):
        test_df = self.get_mixed_df().withColumn(
            "result", h3spark_n.cell_to_center_child(F.col("cell"), 15)
        )
        for res in test_df.collect():
            self.assertEqual(res["result"], h3.cell_to_center_child(res["cell"], 15))

    def test_min_max_child_mixed(self):
        test_df = (
            self.get_mixed_df()
            .withColumn("min", h3spark_n.minchild(F.col("cell"), 15))
            .withColumn("max", h3spark_n.maxchild(F.col("cell"), 15))
        )
        for res in test_df.collect():
            size = h3.cell_to_children_size(res["cell"], 15)
            self.assertEqual(res["min"], h3.child_pos_to_cell(res["cell"], 15, 0))
            self.assertEqual(
                res["max"], h3.child_pos_to_cell(res["cell"], 15, size - 1)
            )

    def test_is_res_class_III(self):
        test_df = self.get_mixed_df().withColumn(
            "result", h3spark_n.is_res_class_III(F.col("cell"))
        )
        for res in test_df.collect():
            self.assertEqual(res["result"], h3.is_res_class_III(res["cell"]))

    def test_per_resolution_constants(self):
        test_df = self.spark.range(16).select(
            F.col("id").alias("res"),
            h3spark_n.get_num_cells(F.col("id")).alias("num_cells"),
            h3spark_n.average_hexagon_area(F.col("id")).alias("area_km2"),
            h3spark_n.average_hexagon_area(F.col("id"), "m^2").alias("area_m2"),
            h3spark_n.average_hexagon_edge_length(F.col("id")).alias("edge_km"),
            h3spark_n.average_hexagon_edge_length(F.col("id"), LengthUnit.M).alias(
                "edge_m"
            ),
            h3spark_n.get_pentagons(F.col("id")).alias("pentagons"),
        )
        for res in test_df.collect():
            r = res["res"]
            self.assertEqual(res["num_cells"], h3.get_num_cells(r))
            self.assertEqual(
                res["area_km2"], h3.average_hexagon_area(r, AreaUnit.KM2.value)
            )
            self.assertEqual(res["area_m2"], h3.average_hexagon_area(r, "m^2"))
            self.assertEqual(res["edge_km"], h3.average_hexagon_edge_length(r, "km"))
            self.assertEqual(res["edge_m"], h3.average_hexagon_edge_length(r, "m"))
            self.assertEqual(res["pentagons"], h3.get_pentagons(r).tolist())

    def test_get_res0_cells(self):
        result = self.get_df().select(h3spark_n.get_res0_cells()).first()[0]
        self.assertEqual(result, h3.get_res0_cells().tolist())

    def test_column_names_needing_quotes(self):
        test_df = self.get_mixed_df().withColumnRenamed("cell", "my cell")
        cell = F.col("`my cell`")
        test_df = test_df.select(
            cell.alias("cell"),
            h3spark_n.is_pentagon(cell).alias("is_pentagon"),
            h3spark_n.minchild(cell, 15).alias("min"),
            h3spark_n.maxchild(cell, 15).alias("max"),
            h3spark_n.cell_to_parent(cell, 0).alias("parent"),
            h3spark_n.cell_to_children_size(cell, F.lit(15)).alias("size"),
        )
        for res in test_df.collect():
            self.assertEqual(res["is_pentagon"], h3.is_pentagon(res["cell"]))
            self.assertEqual(res["parent"], h3.cell_to_parent(res["cell"], 0))
            self.assertEqual(res["size"], h3.cell_to_children_size(res["cell"], 15))


if __name__ == "__main__":
    unittest.main()
