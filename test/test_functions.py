"""Behaviour the generic registry tests can't express"""

import json
import random
import re

import h3.api.numpy_int as h3
import pyspark
import pytest
from pyspark.sql import functions as F

import h3spark
import h3spark.convenience as h3spark_c
import h3spark.hex as h3hex
import h3spark.native as h3spark_n
import h3spark.udf as h3spark_u
from h3spark import REGISTRY
from h3spark._registry import cell_check, with_checks

from .conftest import physical_plan

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
MAX_RES = 15


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
    ).first()
    assert row["disk"] == h3.grid_disk(CELL, 1).tolist()
    assert row["parent"] == h3.cell_to_parent(CELL, 0)


NATIVE_CALLS = {
    "get_resolution": lambda cell: h3spark.get_resolution(cell),
    "cell_to_parent": lambda cell: h3spark.cell_to_parent(cell, 0),
    "is_childof": lambda cell: h3spark_n.is_childof(cell, cell),
}


@pytest.mark.parametrize("name", NATIVE_CALLS)
@pytest.mark.parametrize("cell_type", ["bigint", "int"])
def test_native_accepts_integral_cells(spark, name, cell_type):
    df = spark.createDataFrame([(1,), (None,)], f"cell {cell_type}")
    df.select(NATIVE_CALLS[name]("cell")).collect()


@pytest.mark.parametrize("name", NATIVE_CALLS)
@pytest.mark.parametrize("cell_type", ["string", "double", "boolean"])
def test_native_rejects_other_cell_types_when_planning(spark, name, cell_type):
    # No rows: the error comes from analysing the plan, not from processing data
    df = spark.createDataFrame([], f"cell {cell_type}")
    hint = f"h3spark.hex.{name} or h3spark.str_to_int"
    message = f"{name} expects bigint cells (for hex string cells use {hint})"
    with pytest.raises(Exception, match=re.escape(message)):
        df.select(NATIVE_CALLS[name]("cell")).schema


def test_array_cell_type_check(spark):
    def checked(df):
        cells = F.col("cells")
        return df.select(with_checks(cells, [cell_check(cells, "f", False, True)]))

    assert checked(spark.createDataFrame([([1],)], "cells array<int>")).first()[0] == [
        1
    ]
    message = (
        "f expects array<bigint> cells (for hex string cells use h3spark.str_to_int)"
    )
    with pytest.raises(Exception, match=re.escape(message)):
        checked(spark.createDataFrame([], "cells array<string>")).schema


def test_cell_type_check_is_optimized_away(df):
    plan = physical_plan(df.select(h3spark.cell_to_parent("cell", 0)))
    assert "expects" not in plan and "CASE" not in plan


@pytest.mark.skipif(
    pyspark.__version__ >= "4",
    reason="Spark 4 Connect columns subclass Column, the Spark Connect test run covers them",
)
def test_connect_columns(monkeypatch):
    # Spark Connect columns don't subclass Column in pyspark 3.5
    connect = pytest.importorskip("pyspark.sql.connect.functions", exc_type=ImportError)
    from pyspark.sql.connect.column import Column as ConnectColumn

    # Set by Spark Connect sessions: pyspark.sql.functions then builds Spark Connect columns
    monkeypatch.setenv("SPARK_CONNECT_MODE_ENABLED", "1")
    for cell in ["cell", connect.col("cell")]:
        results = [
            h3spark.cell_to_parent(cell, connect.lit(0)),
            h3hex.cell_to_parent(cell, 0),
        ]
        assert all(isinstance(r, ConnectColumn) for r in results)


def test_is_valid_cell(spark):
    rng = random.Random(0)
    cells = [int(p) for r in range(MAX_RES + 1) for p in h3.get_pentagons(r)]
    cells += [int(h3.cell_to_center_child(CELL, r)) for r in range(1, MAX_RES + 1)]
    values = [*cells, -1, 0, *(rng.getrandbits(63) for _ in range(500))]
    for c in cells:
        values.append(c ^ (1 << rng.randrange(63)))  # any single bit flip
        values.append(c ^ (rng.randint(1, 7) << (3 * rng.randrange(15))))  # one digit
        if h3.is_pentagon(c) and h3.get_resolution(c):
            # Leading digit 1 is the deleted subsequence of a pentagon
            values.append(c | 1 << 3 * (MAX_RES - rng.randint(1, h3.get_resolution(c))))
    expected = [v >= 0 and bool(h3.is_valid_cell(v)) for v in values]
    assert 0 < sum(expected) < len(values)
    rows = [(v, h3.int_to_str(v) if v >= 0 else "zz") for v in values]
    df = spark.createDataFrame(rows, "cell long, cell_str string")
    result = df.select(
        h3spark.is_valid_cell("cell"), h3hex.is_valid_cell("cell_str")
    ).collect()
    assert [r[0] for r in result] == [r[1] for r in result] == expected
    junk = spark.createDataFrame([("zz",), ("",), ("f" * 17,)], "cell_str string")
    assert all(not r[0] for r in junk.select(h3hex.is_valid_cell("cell_str")).collect())


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
