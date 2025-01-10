import json
from decimal import Decimal
from enum import Enum
from typing import List, Union

import h3.api.numpy_int as h3
from pyspark.sql import functions as F
from pyspark.sql import types as T

from .models import BOUNDARY_TYPE, COORDINATE_TYPE, H3_CELL_DECIMAL_TYPE
from .utils import H3CellInput, H3Shape, to_h3_cell, to_h3_shape

# Skipping functions relating to h3 edges and vertices for now


@F.udf(T.DecimalType(20, 0))
def str_to_int(h3_str):
    return Decimal(h3.str_to_int(h3_str))


@F.udf(T.StringType())
def int_to_str(h3_int: Union[str, int]):
    """
    Note: We allow user to pass in a string as Spark does not support unsigned 64-bit integers.
    """
    if isinstance(h3_int, str):
        h3_int = int(h3_int)
    return h3.int_to_str(h3_int)


@F.udf(T.IntegerType())
def get_num_cells(res: int):
    return h3.get_num_cells(res)


class AreaUnit(Enum):
    KM2 = "km^2"
    M2 = "m^2"
    RADS2 = "rads^2"


@F.udf(T.FloatType())
def average_hexagon_area(res: int, unit: Union[AreaUnit, str] = AreaUnit.KM2):
    if isinstance(unit, str):
        unit = AreaUnit(unit)
    return h3.average_hexagon_area(res, unit.value)


class LengthUnit(Enum):
    KM = "km"
    M = "m"
    RADS = "rads"


@F.udf(T.FloatType())
def average_hexagon_edge_length(res: int, unit: Union[LengthUnit, str] = LengthUnit.KM):
    if isinstance(unit, str):
        unit = LengthUnit(unit)
    return h3.average_hexagon_edge_length(res, unit.value)


@F.udf(H3_CELL_DECIMAL_TYPE)
def latlng_to_cell_decimal(lat: float, lng: float, res: int):
    return Decimal(h3.latlng_to_cell(lat, lng, res))


@F.udf(T.StringType())
def latlng_to_cell(lat: float, lng: float, res: int):
    return h3.int_to_str(h3.latlng_to_cell(lat, lng, res))


@F.udf(COORDINATE_TYPE)
def cell_to_latlng(cell: H3CellInput):
    return h3.cell_to_latlng(to_h3_cell(cell))


@F.udf(T.ShortType())
def get_resolution(cell: H3CellInput):
    return h3.get_resolution(to_h3_cell(cell))


@F.udf(H3_CELL_DECIMAL_TYPE)
def cell_to_parent_decimal(cell: H3CellInput, res: int):
    return Decimal(h3.cell_to_parent(to_h3_cell(cell), res))


@F.udf(T.StringType())
def cell_to_parent(cell: H3CellInput, res: int):
    return h3.int_to_str(h3.cell_to_parent(to_h3_cell(cell), res))


@F.udf(T.IntegerType())
def grid_distance(cell1: H3CellInput, cell2: H3CellInput):
    return h3.grid_distance(to_h3_cell(cell1), to_h3_cell(cell2))


@F.udf(BOUNDARY_TYPE)
def cell_to_boundary(cell: H3CellInput):
    return h3.cell_to_boundary(to_h3_cell(cell))


@F.udf(T.ArrayType(H3_CELL_DECIMAL_TYPE))
def grid_disk_decimal(cell: H3CellInput, k: int):
    return [Decimal(c) for c in h3.grid_disk(to_h3_cell(cell), k)]


@F.udf(T.ArrayType(T.StringType()))
def grid_disk(cell: H3CellInput, k: int):
    return [h3.int_to_str(c) for c in h3.grid_disk(to_h3_cell(cell), k)]


@F.udf(T.ArrayType(H3_CELL_DECIMAL_TYPE))
def grid_ring_decimal(cell: H3CellInput, k: int):
    return [Decimal(c) for c in h3.grid_ring(to_h3_cell(cell), k)]


@F.udf(T.ArrayType(T.StringType()))
def grid_ring(cell: H3CellInput, k: int):
    return [h3.int_to_str(c) for c in h3.grid_ring(to_h3_cell(cell), k)]


@F.udf(T.IntegerType())
def cell_to_children_size(cell: H3CellInput, res: int):
    return h3.cell_to_children_size(to_h3_cell(cell), res)


@F.udf(T.ArrayType(H3_CELL_DECIMAL_TYPE))
def cell_to_children_decimal(cell: H3CellInput, res: int):
    return [Decimal(c) for c in h3.cell_to_children(to_h3_cell(cell), res)]


@F.udf(T.ArrayType(T.StringType()))
def cell_to_children(cell: H3CellInput, res: int):
    return [h3.int_to_str(c) for c in h3.cell_to_children(to_h3_cell(cell), res)]


@F.udf(T.IntegerType())
def cell_to_child_pos(child: H3CellInput, res_parent: int):
    return h3.cell_to_child_pos(to_h3_cell(child), res_parent)


@F.udf(H3_CELL_DECIMAL_TYPE)
def child_pos_to_cell_decimal(parent: H3CellInput, res_child: int, child_pos: int):
    return Decimal(h3.child_pos_to_cell(to_h3_cell(parent), res_child, child_pos))


@F.udf(T.StringType())
def child_pos_to_cell(parent: H3CellInput, res_child: int, child_pos: int):
    return h3.int_to_str(h3.child_pos_to_cell(to_h3_cell(parent), res_child, child_pos))


@F.udf(T.ArrayType(H3_CELL_DECIMAL_TYPE))
def compact_cells_decimal(cells: List[H3CellInput]):
    return [Decimal(c) for c in h3.compact_cells(cells)]


@F.udf(T.ArrayType(T.StringType()))
def compact_cells(cells: List[H3CellInput]):
    return [h3.int_to_str(c) for c in h3.compact_cells(cells)]


@F.udf(T.ArrayType(H3_CELL_DECIMAL_TYPE))
def uncompact_cells_decimal(cells: List[H3CellInput], res: int):
    return [Decimal(c) for c in h3.uncompact_cells(cells, res)]


@F.udf(T.ArrayType(T.StringType()))
def uncompact_cells(cells: List[H3CellInput], res: int):
    return [h3.int_to_str(c) for c in h3.uncompact_cells(cells, res)]


@F.udf(T.ArrayType(H3_CELL_DECIMAL_TYPE))
def h3shape_to_cells_decimal(shape: H3Shape, res: int):
    return [Decimal(c) for c in h3.h3shape_to_cells(to_h3_shape(shape), res)]


@F.udf(T.ArrayType(T.StringType()))
def h3shape_to_cells(shape: H3Shape, res: int):
    return [h3.int_to_str(c) for c in h3.h3shape_to_cells(to_h3_shape(shape), res)]


@F.udf(T.StringType())
def cells_to_h3shape(cells: List[H3CellInput]):
    res = h3.cells_to_h3shape([to_h3_cell(c) for c in cells])
    return json.dumps(res.__geo_interface__)


@F.udf(T.BooleanType())
def is_pentagon(cell: H3CellInput):
    return h3.is_pentagon(to_h3_cell(cell))


@F.udf(T.IntegerType())
def get_base_cell_number(cell: H3CellInput):
    return h3.get_base_cell_number(to_h3_cell(cell))


@F.udf(T.BooleanType())
def are_neighbor_cells(cell1: H3CellInput, cell2: H3CellInput):
    return h3.are_neighbor_cells(to_h3_cell(cell1), to_h3_cell(cell2))


@F.udf(T.ArrayType(H3_CELL_DECIMAL_TYPE))
def grid_path_cells_decimal(start: H3CellInput, end: H3CellInput):
    return [Decimal(c) for c in h3.grid_path_cells(to_h3_cell(start), to_h3_cell(end))]


@F.udf(T.ArrayType(T.StringType()))
def grid_path_cells(start: H3CellInput, end: H3CellInput):
    return [
        h3.int_to_str(c) for c in h3.grid_path_cells(to_h3_cell(start), to_h3_cell(end))
    ]


@F.udf(T.BooleanType())
def is_res_class_III(cell: H3CellInput):
    return h3.is_res_class_III(to_h3_cell(cell))


@F.udf(T.ArrayType(H3_CELL_DECIMAL_TYPE))
def get_pentagons_decimal(res: int):
    return [Decimal(c) for c in h3.get_pentagons(res)]


@F.udf(T.ArrayType(T.StringType()))
def get_pentagons(res: int):
    return [h3.int_to_str(c) for c in h3.get_pentagons(res)]


@F.udf(T.ArrayType(H3_CELL_DECIMAL_TYPE))
def get_res0_cells_decimal():
    return [Decimal(c) for c in h3.get_res0_cells()]


@F.udf(T.ArrayType(T.StringType()))
def get_res0_cells():
    return [h3.int_to_str(c) for c in h3.get_res0_cells()]


@F.udf(H3_CELL_DECIMAL_TYPE)
def cell_to_center_child_decimal(cell: H3CellInput, res: int):
    return Decimal(h3.cell_to_center_child(to_h3_cell(cell), res))


@F.udf(T.StringType())
def cell_to_center_child(cell: H3CellInput, res: int):
    return h3.int_to_str(h3.cell_to_center_child(to_h3_cell(cell), res))


@F.udf(T.ArrayType(T.IntegerType()))
def get_icosahedron_faces(cell: H3CellInput):
    return h3.get_icosahedron_faces(to_h3_cell(cell))


@F.udf(T.ArrayType(T.IntegerType()))
def cell_to_local_ij(cell: H3CellInput):
    return h3.cell_to_local_ij(to_h3_cell(cell))


@F.udf(H3_CELL_DECIMAL_TYPE)
def local_ij_to_cell_decimal(origin: H3CellInput, i: int, j: int):
    return Decimal(h3.local_ij_to_cell(to_h3_cell(origin), i, j))


@F.udf(T.StringType())
def local_ij_to_cell(origin: H3CellInput, i: int, j: int):
    return h3.int_to_str(h3.local_ij_to_cell(to_h3_cell(origin), i, j))


@F.udf(T.FloatType())
def cell_area(cell: H3CellInput, unit: Union[AreaUnit, str] = AreaUnit.KM2):
    if isinstance(unit, str):
        unit = AreaUnit(unit)
    return h3.cell_area(to_h3_cell(cell), unit.value)
