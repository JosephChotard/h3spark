"""
Generic tests over every registered function, on random valid cells across all resolutions plus every pentagon and
res 0 cell:

- the default, UDF (and so native, when there is one) implementations equal the h3-py reference
- h3spark.hex.<name> equals the long version with cells converted to hex strings, with string output types
- result types match the declared return type, and hex versions of native functions run without Python UDFs
- every version names its result like a UDF call, e.g. cell_to_parent(cell, res)

Adding a function only needs its @h3fn definition (and an argument generator below if it uses a new parameter name).
"""

import itertools
import json
import math
import random
from enum import Enum

import h3.api.numpy_int as h3
import pytest
from pyspark.sql import functions as F

import h3spark.hex as h3hex
from h3spark import REGISTRY

from .conftest import physical_plan

MAX_RES = 15
N_RANDOM_CELLS = 3000


def random_cell(rng: random.Random) -> int:
    lat = math.degrees(math.asin(rng.uniform(-1, 1)))  # uniform on the sphere
    lng = rng.uniform(-180, 180)
    return int(h3.latlng_to_cell(lat, lng, rng.randint(0, MAX_RES)))


def random_descendant(rng: random.Random, cell: int) -> int:
    res = rng.randint(h3.get_resolution(cell), MAX_RES)
    return int(
        h3.child_pos_to_cell(
            cell, res, rng.randrange(h3.cell_to_children_size(cell, res))
        )
    )


def build_cells() -> list[int]:
    rng = random.Random(42)
    pentagons = [int(p) for r in range(MAX_RES + 1) for p in h3.get_pentagons(r)]
    return (
        [random_cell(rng) for _ in range(N_RANDOM_CELLS)]
        + pentagons
        # Pentagons' descendants have their own rules (no digit 1 subsequence)
        + [random_descendant(rng, p) for p in pentagons]
        + [int(c) for c in h3.get_res0_cells()]
    )


def generate_args(rng: random.Random, cell: int, small: bool) -> dict:
    """
    Arguments by parameter name, derived from a random cell. Invalid combinations (e.g. a parent resolution above the
    cell's) are fine: rows where the h3-py reference raises are skipped. `small` keeps array results small.
    """
    res = h3.get_resolution(cell)
    other = int(rng.choice(h3.grid_disk(cell, 2)))
    parent = int(h3.cell_to_parent(rng.choice([cell, other]), rng.randint(0, res)))
    any_res = (
        min(max(res + rng.randint(-3, 2), 0), MAX_RES)
        if small
        else rng.randint(0, MAX_RES)
    )
    try:
        i, j = h3.cell_to_local_ij(other, cell)
    except Exception:
        i, j = 0, 0
    lat, lng = h3.cell_to_latlng(cell)
    return {
        **dict.fromkeys(["cell", "h3_int", "child", "cell1", "start"], cell),
        **dict.fromkeys(["cell2", "end", "origin"], other),
        **dict.fromkeys(
            ["res", "resolution", "res_parent", "child_resolution"], any_res
        ),
        "parent": parent,
        "res_child": res,
        "child_pos": int(h3.cell_to_child_pos(cell, h3.get_resolution(parent))),
        "h3_str": rng.choice([str.lower, str.upper])(h3.int_to_str(cell)),
        "k": rng.randint(0, 2),
        "i": int(i),
        "j": int(j),
        "lat": lat,
        "lng": lng,
        "cells": h3.cell_to_children(cell, min(res + 1, MAX_RES)).tolist(),
        "shape": json.dumps(h3.cells_to_h3shape([cell]).__geo_interface__),
    }


CELLS = build_cells()
_rng = random.Random(0)
ARGS = {
    small: [generate_args(_rng, c, small) for c in CELLS] for small in (False, True)
}


def static_values(entry, param: str) -> list:
    default = entry.signature.parameters[param].default
    if isinstance(default, Enum):
        return [member.value for member in type(default)]
    if isinstance(default, bool):
        return [False, True]
    raise NotImplementedError(f"No test values for {entry.name}({param})")


CASES = [
    pytest.param(
        name, dict(combo), id="-".join([name, *map(str, dict(combo).values())])
    )
    for name, entry in REGISTRY.items()
    for combo in itertools.product(
        *[[(p, v) for v in static_values(entry, p)] for p in entry.static]
    )
]


def normalize(value):
    if isinstance(value, (list, tuple)):  # also Rows
        return tuple(normalize(v) for v in value)
    return value


def to_hex(value):
    if value is None:
        return None
    if isinstance(value, list):
        return [h3.int_to_str(v) for v in value]
    return h3.int_to_str(value)


def test_cells_cover_all_resolutions_and_pentagons():
    assert {h3.get_resolution(c) for c in CELLS} == set(range(MAX_RES + 1))
    assert sum(map(h3.is_pentagon, CELLS)) >= 12 * (MAX_RES + 1)


@pytest.mark.parametrize("name, static", CASES)
def test_function(spark, name, static):
    entry = REGISTRY[name]
    params = [p for p in entry.signature.parameters if p not in entry.static]
    rows, expected = [], []
    for args in ARGS[entry.returns == "cells"]:
        values = [args[p] for p in params]
        try:
            expected.append(normalize(entry.reference(*values, **static)))
        except Exception:
            continue
        hex_values = [
            to_hex(v) if p in entry.cells + entry.cell_lists else v
            for p, v in zip(params, values)
        ]
        rows.append([len(rows), *values, *hex_values])

    if not rows:
        # h3 rejects the static arguments for every row, e.g. unit="rads^2" for average_hexagon_area
        with pytest.raises(Exception):
            spark.range(1).select(
                entry.default(*([F.lit(0)] * len(params)), **static)
            ).collect()
        return
    assert len(rows) > 100

    df = spark.createDataFrame(rows, ["idx", *params, *[f"{p}_hex" for p in params]])
    cols = [F.col(p) for p in params]
    hex_cols = [F.col(f"{p}_hex") for p in params]
    result = df.select(
        "idx",
        entry.default(*cols, **static).alias("default"),
        entry.udf(*cols, **static).alias("udf"),
        *(
            [getattr(h3hex, name)(*hex_cols, **static).alias("hex")]
            if entry.hex
            else []
        ),
    )

    implementations = [entry.default, entry.udf] + (
        [getattr(h3hex, name)] if entry.hex else []
    )
    for implementation in implementations:
        unnamed = df.select(implementation(*map(F.col, params), **static))
        assert unnamed.columns == [f"{name}({', '.join(params)})"]

    types = {f.name: f.dataType.simpleString() for f in result.schema.fields}
    assert types["default"] == types["udf"] == entry.return_type.simpleString()
    if entry.hex:
        expected_hex_type = {"cell": "string", "cells": "array<string>"}
        assert types["hex"] == expected_hex_type.get(entry.returns, types["default"])
        if entry.native:
            plan = physical_plan(df.select(getattr(h3hex, name)(*hex_cols, **static)))
            assert "Python" not in plan

    for row in result.collect():
        assert normalize(row["default"]) == expected[row["idx"]]
        assert normalize(row["udf"]) == expected[row["idx"]]
        if entry.hex:
            long_value = row["default"]
            if entry.returns:
                long_value = to_hex(long_value)
            assert row["hex"] == long_value
