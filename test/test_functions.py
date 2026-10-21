"""Behaviour the generic registry tests can't express"""

import json

import h3.api.numpy_int as h3
import pytest
from pyspark.sql import functions as F

import h3spark
import h3spark.convenience as h3spark_c
import h3spark.hex as h3hex
import h3spark.native as h3spark_n
import h3spark.udf as h3spark_u
from h3spark import REGISTRY

POLYGON = """{
    "type": "Polygon",
    "coordinates": [[
        [-89.998354, 29.8988], [-89.99807, 29.8988], [-89.99807, 29.898628],
        [-89.998354, 29.898628], [-89.998354, 29.8988]
    ]]
}"""
RING = json.loads(POLYGON)["coordinates"][0]
CELL = 582169416674836479
CELL_STR = "81447ffffffffff"


@pytest.fixture(scope="module")
def df(spark):
    return spark.createDataFrame(
        [(CELL, CELL_STR, POLYGON)], ["cell", "cell_str", "shape"]
    )


def test_shapes(df):
    lat_lng_ring = F.array(*[F.array(F.lit(lat), F.lit(lng)) for lng, lat in RING])
    row = df.select(
        F.size(h3spark.h3shape_to_cells("shape", 15)).alias("cells"),
        F.size(h3spark_c.h3shape_to_cells_compacted("shape", 15)).alias("compacted"),
        F.size(h3spark.h3shape_to_cells(lat_lng_ring, 15)).alias("ring_cells"),
    ).first()
    assert (row["cells"], row["compacted"], row["ring_cells"]) == (554, 122, 554)


def test_udfs_accept_hex_string_cells(df):
    row = df.select(
        h3spark.grid_disk("cell_str", 1).alias("disk"),
        h3spark_u.cell_to_parent("cell_str", 0).alias("parent"),
        h3spark.cell_to_parent("cell_str", 0).alias("native_parent"),
    ).first()
    assert row["disk"] == h3.grid_disk(CELL, 1).tolist()
    assert row["parent"] == h3.cell_to_parent(CELL, 0)
    # Native functions need long cells: hex strings cast to null
    assert row["native_parent"] is None


def test_validate_resolution(df):
    valid = df.select(
        h3spark.cell_to_children_size("cell", 1, validate_resolution=True)
    )
    assert valid.first()[0] == 1
    assert df.select(h3spark.cell_to_children_size("cell", 0)).first()[0] is None
    with pytest.raises(Exception, match="Child resolution"):
        df.select(h3spark.cell_to_children_size("cell", 0, True)).collect()


def test_units(df):
    with pytest.raises(TypeError):
        h3spark.cell_area("cell", F.lit("km^2"))
    with pytest.raises(TypeError):
        h3spark.average_hexagon_area(7, F.lit("km^2"))
    with pytest.raises(ValueError):
        h3spark.cell_area("cell", "furlongs^2")
    row = df.select(
        h3spark.cell_area("cell", h3spark.AreaUnit.M2), h3spark.cell_area("cell", "m^2")
    ).first()
    assert row[0] == row[1] == h3.cell_area(CELL, "m^2")


def test_column_argument_types(df):
    with pytest.raises(TypeError):
        h3spark.get_resolution(123)
    row = df.select(
        h3spark.get_resolution("cell"),
        h3spark.get_resolution(F.col("cell")),
        h3spark.cell_to_parent("cell", F.lit(0)),
        h3spark.cell_to_parent("cell", 0),
    ).first()
    assert row[0] == row[1] == 1
    assert row[2] == row[3] == h3.cell_to_parent(CELL, 0)


def test_large_child_pos(spark):
    res0 = int(h3.get_res0_cells()[0])
    child = int(h3.child_pos_to_cell(res0, 15, h3.cell_to_children_size(res0, 15) - 1))
    pos = spark.range(1).select(h3spark.cell_to_child_pos(F.lit(child), 0)).first()[0]
    assert pos == h3.cell_to_child_pos(child, 0) > 2**31


def test_coordinate_type():
    assert (
        REGISTRY["cell_to_latlng"].return_type.simpleString()
        == "struct<lat:double,lon:double>"
    )


def test_modules():
    assert {e.name for e in REGISTRY.values() if e.native}.issubset(h3spark_n.__all__)
    assert (
        h3spark_n.minchild is h3spark.min_child
        and h3spark_n.get_base_cell is h3spark.get_base_cell_number
    )
    assert set(h3spark_u.__all__) == set(REGISTRY)
    assert set(h3hex.__all__) == set(REGISTRY) - {"str_to_int", "int_to_str"}
    assert h3spark.grid_disk is h3spark_u.grid_disk
    assert h3spark.cell_to_parent is h3spark_n.cell_to_parent
