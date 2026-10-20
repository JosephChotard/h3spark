import json
from typing import List, Union

import h3.api.numpy_int as h3
from pyspark.sql import functions as F
from pyspark.sql import types as T

from .models import BOUNDARY_TYPE, COORDINATE_TYPE, AreaUnit, LengthUnit
from .utils import H3CellInput, H3Shape, to_h3_cell, to_h3_shape

# Skipping functions relating to h3 edges and vertices for now


@F.udf(T.LongType(), useArrow=True)
def str_to_int(h3_str):
    return h3.str_to_int(h3_str)


@F.udf(T.StringType(), useArrow=True)
def int_to_str(h3_int: Union[str, int]):
    """
    Note: We allow user to pass in a string as Spark does not support unsigned 64-bit integers.
    """
    if isinstance(h3_int, str):
        h3_int = int(h3_int)
    return h3.int_to_str(h3_int)


@F.udf(T.LongType(), useArrow=True)
def get_num_cells(res: int):
    return h3.get_num_cells(res)


@F.udf(T.FloatType(), useArrow=True)
def average_hexagon_area(res: int, unit: Union[AreaUnit, str] = AreaUnit.KM2):
    if isinstance(unit, str):
        unit = AreaUnit(unit)
    return h3.average_hexagon_area(res, unit.value)


@F.udf(T.FloatType(), useArrow=True)
def average_hexagon_edge_length(res: int, unit: Union[LengthUnit, str] = LengthUnit.KM):
    if isinstance(unit, str):
        unit = LengthUnit(unit)
    return h3.average_hexagon_edge_length(res, unit.value)


@F.udf(T.LongType(), useArrow=True)
def latlng_to_cell(lat: float, lng: float, res: int):
    return h3.latlng_to_cell(lat, lng, res)


@F.udf(COORDINATE_TYPE, useArrow=True)
def cell_to_latlng(cell: H3CellInput):
    return h3.cell_to_latlng(to_h3_cell(cell))


@F.udf(T.ShortType(), useArrow=True)
def get_resolution(cell: H3CellInput):
    return h3.get_resolution(to_h3_cell(cell))


@F.udf(T.LongType(), useArrow=True)
def cell_to_parent(cell: H3CellInput, res: int):
    return h3.cell_to_parent(to_h3_cell(cell), res)


@F.udf(T.IntegerType(), useArrow=True)
def grid_distance(cell1: H3CellInput, cell2: H3CellInput):
    return h3.grid_distance(to_h3_cell(cell1), to_h3_cell(cell2))


@F.udf(BOUNDARY_TYPE, useArrow=True)
def cell_to_boundary(cell: H3CellInput):
    return h3.cell_to_boundary(to_h3_cell(cell))


@F.udf(T.ArrayType(T.LongType()), useArrow=True)
def grid_disk(cell: H3CellInput, k: int):
    return h3.grid_disk(to_h3_cell(cell), k).tolist()


@F.udf(T.ArrayType(T.LongType()), useArrow=True)
def grid_ring(cell: H3CellInput, k: int):
    return h3.grid_ring(to_h3_cell(cell), k).tolist()


@F.udf(T.IntegerType(), useArrow=True)
def cell_to_children_size(cell: H3CellInput, res: int):
    return h3.cell_to_children_size(to_h3_cell(cell), res)


@F.udf(T.ArrayType(T.LongType()), useArrow=True)
def cell_to_children(cell: H3CellInput, res: int):
    return h3.cell_to_children(to_h3_cell(cell), res).tolist()


@F.udf(T.IntegerType(), useArrow=True)
def cell_to_child_pos(child: H3CellInput, res_parent: int):
    return h3.cell_to_child_pos(to_h3_cell(child), res_parent)


@F.udf(T.LongType(), useArrow=True)
def child_pos_to_cell(parent: H3CellInput, res_child: int, child_pos: int):
    return h3.child_pos_to_cell(to_h3_cell(parent), res_child, child_pos)


@F.udf(T.ArrayType(T.LongType()), useArrow=True)
def compact_cells(cells: List[H3CellInput]):
    return h3.compact_cells([to_h3_cell(c) for c in cells]).tolist()


@F.udf(T.ArrayType(T.LongType()), useArrow=True)
def uncompact_cells(cells: List[H3CellInput], res: int):
    return h3.uncompact_cells([to_h3_cell(c) for c in cells], res).tolist()


@F.udf(T.ArrayType(T.LongType()), useArrow=True)
def h3shape_to_cells(shape: H3Shape, res: int):
    return h3.h3shape_to_cells(to_h3_shape(shape), res).tolist()


@F.udf(T.StringType(), useArrow=True)
def cells_to_h3shape(cells: List[H3CellInput]):
    res = h3.cells_to_h3shape([to_h3_cell(c) for c in cells])
    return json.dumps(res.__geo_interface__)


@F.udf(T.BooleanType(), useArrow=True)
def is_pentagon(cell: H3CellInput):
    return h3.is_pentagon(to_h3_cell(cell))


@F.udf(T.IntegerType(), useArrow=True)
def get_base_cell_number(cell: H3CellInput):
    return h3.get_base_cell_number(to_h3_cell(cell))


@F.udf(T.BooleanType(), useArrow=True)
def are_neighbor_cells(cell1: H3CellInput, cell2: H3CellInput):
    return h3.are_neighbor_cells(to_h3_cell(cell1), to_h3_cell(cell2))


@F.udf(T.ArrayType(T.LongType()), useArrow=True)
def grid_path_cells(start: H3CellInput, end: H3CellInput):
    return h3.grid_path_cells(to_h3_cell(start), to_h3_cell(end)).tolist()


@F.udf(T.BooleanType(), useArrow=True)
def is_res_class_III(cell: H3CellInput):
    return h3.is_res_class_III(to_h3_cell(cell))


@F.udf(T.ArrayType(T.LongType()), useArrow=True)
def get_pentagons(res: int):
    return h3.get_pentagons(res).tolist()


# Arrow can't be used for UDFs without arguments
@F.udf(T.ArrayType(T.LongType()))
def get_res0_cells():
    return h3.get_res0_cells().tolist()


@F.udf(T.LongType(), useArrow=True)
def cell_to_center_child(cell: H3CellInput, res: int):
    return h3.cell_to_center_child(to_h3_cell(cell), res)


@F.udf(T.ArrayType(T.IntegerType()), useArrow=True)
def get_icosahedron_faces(cell: H3CellInput):
    return h3.get_icosahedron_faces(to_h3_cell(cell)).tolist()


@F.udf(T.ArrayType(T.IntegerType()), useArrow=True)
def cell_to_local_ij(cell: H3CellInput):
    return h3.cell_to_local_ij(to_h3_cell(cell)).tolist()


@F.udf(T.LongType(), useArrow=True)
def local_ij_to_cell(origin: H3CellInput, i: int, j: int):
    return h3.local_ij_to_cell(to_h3_cell(origin), i, j)


@F.udf(T.FloatType(), useArrow=True)
def cell_area(cell: H3CellInput, unit: Union[AreaUnit, str] = AreaUnit.KM2):
    if isinstance(unit, str):
        unit = AreaUnit(unit)
    return h3.cell_area(to_h3_cell(cell), unit.value)
