"""
PySpark wrappers around h3-py.

Every function is defined once below with ``@h3fn`` (see ``h3spark._registry``). ``h3spark.<name>`` is the default
implementation: native Spark expressions when the function has a body, otherwise a Python UDF calling h3-py.
``h3spark.native``, ``h3spark.udf`` and ``h3spark.hex`` expose the native, UDF and hex string versions.

Column arguments accept a ``Column`` or a column name; other arguments (e.g. ``res``, ``k``) also accept plain
Python values. ``unit`` arguments are plain Python values resolved when the plan is built. Native implementations
require ``LongType`` cells and do less validation than h3-py; UDFs accept long or hex string cells.
"""

import h3.api.numpy_int as h3
from pyspark.sql import functions as F
from pyspark.sql import types as T
from pyspark.sql.column import Column

from . import _bits as B
from ._registry import REGISTRY, ColumnOrName, IntOrColumn, h3fn, unit_value
from .models import BOUNDARY_TYPE, COORDINATE_TYPE, AreaUnit, LengthUnit

# Skipping functions relating to h3 edges and vertices for now

_LONG = T.LongType()
_INT = T.IntegerType()
_BOOL = T.BooleanType()
_DOUBLE = T.DoubleType()
_CELLS = T.ArrayType(T.LongType())
_area_unit = unit_value(AreaUnit)
_length_unit = unit_value(LengthUnit)


def _h3_is_childof(child: int, parent: int) -> bool:
    res = h3.get_resolution(parent)
    return res <= h3.get_resolution(child) and h3.cell_to_parent(child, res) == parent


def _h3_is_valid_cell(cell: int | None) -> bool | None:
    # h3 raises for negative ints, which a long column can hold
    return None if cell is None else cell >= 0 and bool(h3.is_valid_cell(cell))


def _h3_max_child(cell: int, res: int) -> int:
    return h3.child_pos_to_cell(cell, res, h3.cell_to_children_size(cell, res) - 1)


def _h3shape_to_cells_compacted(shape: h3.H3Shape, res: int):
    return h3.compact_cells(h3.h3shape_to_cells(shape, res))


# Native implementations


@h3fn(h3=h3.str_to_int, return_type=_LONG, returns="cell", hex=False)
def str_to_int(h3_str: ColumnOrName) -> Column:
    """Parses a hex string cell. Performs no validation: strings that aren't hex give meaningless values or null"""
    # Signed so the conversion can't overflow a long, which raises in ANSI mode
    # (the length check is on the input so Spark can still reuse the conversion when the result is used repeatedly)
    return F.conv(F.when(F.length(h3_str) <= 16, h3_str), 16, -10).cast("long")


@h3fn(h3=h3.int_to_str, return_type=T.StringType(), cells=("h3_int",), hex=False)
def int_to_str(h3_int: ColumnOrName) -> Column:
    """Formats a cell as a lowercase hex string. Performs no validation"""
    return F.lower(F.hex(h3_int))


@h3fn(h3=_h3_is_valid_cell, return_type=_BOOL, cells=("cell",))
def is_valid_cell(cell: ColumnOrName) -> Column:
    """Same checks as h3-py. Native: null for null cells"""
    return B.is_valid_cell(cell)


@h3fn(h3=h3.get_resolution, return_type=_INT, cells=("cell",))
def get_resolution(cell: ColumnOrName) -> Column:
    return B.resolution(cell)


@h3fn(h3=h3.get_base_cell_number, return_type=_INT, cells=("cell",))
def get_base_cell_number(cell: ColumnOrName) -> Column:
    return B.base_cell(cell)


@h3fn(h3=h3.is_pentagon, return_type=_BOOL, cells=("cell",))
def is_pentagon(cell: ColumnOrName) -> Column:
    return B.is_pentagon(cell)


@h3fn(h3=h3.is_res_class_III, return_type=_BOOL, cells=("cell",))
def is_res_class_III(cell: ColumnOrName) -> Column:
    return B.resolution(cell) % 2 == 1


@h3fn(h3=h3.cell_to_parent, return_type=_LONG, cells=("cell",), returns="cell")
def cell_to_parent(cell: ColumnOrName, res: IntOrColumn) -> Column:
    """Native: doesn't validate that res is at most the cell's resolution"""
    return B.parent(cell, res)


@h3fn(h3=h3.cell_to_center_child, return_type=_LONG, cells=("cell",), returns="cell")
def min_child(cell: ColumnOrName, resolution: IntOrColumn) -> Column:
    """Child with the lowest index, i.e. the center child. Native: doesn't validate the resolution"""
    masked = cell.bitwiseAND(F.bitwise_not(B.digits_between(cell, resolution)))
    return B.set_resolution(masked, resolution)


@h3fn(h3=_h3_max_child, return_type=_LONG, cells=("cell",), returns="cell")
def max_child(cell: ColumnOrName, resolution: IntOrColumn) -> Column:
    """Child with the highest index. Native: doesn't validate the resolution"""
    # Unused digits are 7 (0b111), so clearing the lowest bit of each new digit sets it to 6
    mask = B.digits_between(cell, resolution).bitwiseAND(B.MAX_CHILD_MASK)
    return B.set_resolution(cell.bitwiseAND(F.bitwise_not(mask)), resolution)


@h3fn(h3=h3.cell_to_center_child, return_type=_LONG, cells=("cell",), returns="cell")
def cell_to_center_child(cell: ColumnOrName, res: IntOrColumn) -> Column:
    """The center child has all new digits set to 0. Native: doesn't validate the resolution"""
    return min_child(cell, res)


@h3fn(
    h3=h3.cell_to_children_size,
    return_type=_LONG,
    cells=("cell",),
    static={"validate_resolution": bool},
)
def cell_to_children_size(
    cell: ColumnOrName, res: IntOrColumn, validate_resolution: bool = False
) -> Column:
    """
    Native: null if res is lower than the cell's resolution, or an error if validate_resolution is True (or ANSI mode
    is enabled)
    """
    n = res - B.resolution(cell)
    size = B.children_size(B.is_pentagon(cell), n)
    if validate_resolution:
        message = "Child resolution must be greater than or equal to parent resolution"
        size = F.when(F.assert_true(n >= 0, message).isNull(), size)
    return size


@h3fn(h3=_h3_is_childof, return_type=_BOOL, cells=("child", "parent"))
def is_childof(child: ColumnOrName, parent: ColumnOrName) -> Column:
    """True if child is parent or one of its descendants"""
    mask = F.lit((1 << B.H3_RES_OFFSET) - 1).bitwiseXOR(
        B.digits_after_res(B.resolution(parent))
    )
    return child.bitwiseXOR(parent).bitwiseAND(mask) == 0


@h3fn(
    h3=lambda cell, res: h3.get_resolution(cell) < res <= B.MAX_H3_RES,
    return_type=_BOOL,
    cells=("cell",),
)
def has_child_at_resolution(
    cell: ColumnOrName, child_resolution: IntOrColumn
) -> Column:
    return (child_resolution > B.resolution(cell)) & (child_resolution <= B.MAX_H3_RES)


@h3fn(h3=h3.cell_to_child_pos, return_type=_LONG, cells=("child",))
def cell_to_child_pos(child: ColumnOrName, res_parent: IntOrColumn) -> Column:
    """Native: null if res_parent is above the child's resolution, doesn't validate the child"""
    return B.child_pos(child, res_parent)


@h3fn(h3=h3.child_pos_to_cell, return_type=_LONG, cells=("parent",), returns="cell")
def child_pos_to_cell(
    parent: ColumnOrName, res_child: IntOrColumn, child_pos: IntOrColumn
) -> Column:
    """Native: null if child_pos or res_child is out of range"""
    return B.child_at_pos(parent, res_child, child_pos)


@h3fn(h3=h3.get_num_cells, return_type=_LONG)
def get_num_cells(res: IntOrColumn) -> Column:
    return B.lookup([h3.get_num_cells(r) for r in B.RESOLUTIONS], res)


@h3fn(h3=h3.average_hexagon_area, return_type=_DOUBLE, static={"unit": _area_unit})
def average_hexagon_area(
    res: IntOrColumn, unit: AreaUnit | str = AreaUnit.KM2
) -> Column:
    """Only km^2 and m^2 are supported by h3"""
    return B.lookup([h3.average_hexagon_area(r, unit) for r in B.RESOLUTIONS], res)


@h3fn(
    h3=h3.average_hexagon_edge_length,
    return_type=_DOUBLE,
    static={"unit": _length_unit},
)
def average_hexagon_edge_length(
    res: IntOrColumn, unit: LengthUnit | str = LengthUnit.KM
) -> Column:
    """Only km and m are supported by h3"""
    lengths = [h3.average_hexagon_edge_length(r, unit) for r in B.RESOLUTIONS]
    return B.lookup(lengths, res)


@h3fn(h3=h3.get_pentagons, return_type=_CELLS, returns="cells", hex="table")
def get_pentagons(res: IntOrColumn) -> Column:
    return B.lookup([[int(p) for p in h3.get_pentagons(r)] for r in B.RESOLUTIONS], res)


@h3fn(h3=h3.get_res0_cells, return_type=_CELLS, returns="cells", hex="table")
def get_res0_cells() -> Column:
    return B.literal_array([int(c) for c in h3.get_res0_cells()])


# Python UDFs


@h3fn(h3=h3.latlng_to_cell, return_type=_LONG, returns="cell")
def latlng_to_cell(
    lat: ColumnOrName | float, lng: ColumnOrName | float, res: IntOrColumn
) -> Column: ...


@h3fn(h3=h3.cell_to_latlng, return_type=COORDINATE_TYPE, cells=("cell",))
def cell_to_latlng(cell: ColumnOrName) -> Column: ...


@h3fn(h3=h3.grid_distance, return_type=_INT, cells=("cell1", "cell2"))
def grid_distance(cell1: ColumnOrName, cell2: ColumnOrName) -> Column: ...


@h3fn(h3=h3.cell_to_boundary, return_type=BOUNDARY_TYPE, cells=("cell",))
def cell_to_boundary(cell: ColumnOrName) -> Column: ...


@h3fn(h3=h3.grid_disk, return_type=_CELLS, cells=("cell",), returns="cells")
def grid_disk(cell: ColumnOrName, k: IntOrColumn) -> Column: ...


@h3fn(h3=h3.grid_ring, return_type=_CELLS, cells=("cell",), returns="cells")
def grid_ring(cell: ColumnOrName, k: IntOrColumn) -> Column: ...


@h3fn(h3=h3.cell_to_children, return_type=_CELLS, cells=("cell",), returns="cells")
def cell_to_children(cell: ColumnOrName, res: IntOrColumn) -> Column: ...


@h3fn(h3=h3.compact_cells, return_type=_CELLS, cell_lists=("cells",), returns="cells")
def compact_cells(cells: ColumnOrName) -> Column: ...


@h3fn(h3=h3.uncompact_cells, return_type=_CELLS, cell_lists=("cells",), returns="cells")
def uncompact_cells(cells: ColumnOrName, res: IntOrColumn) -> Column: ...


@h3fn(h3=h3.h3shape_to_cells, return_type=_CELLS, shapes=("shape",), returns="cells")
def h3shape_to_cells(shape: ColumnOrName, res: IntOrColumn) -> Column:
    """shape: GeoJSON string, or array of (lat, lng) / array of arrays of (lat, lng)"""
    ...


@h3fn(
    h3=_h3shape_to_cells_compacted,
    return_type=_CELLS,
    shapes=("shape",),
    returns="cells",
)
def h3shape_to_cells_compacted(shape: ColumnOrName, res: IntOrColumn) -> Column:
    """h3shape_to_cells followed by compact_cells, without an extra pass over the data"""
    ...


@h3fn(h3=h3.cells_to_h3shape, return_type=T.StringType(), cell_lists=("cells",))
def cells_to_h3shape(cells: ColumnOrName) -> Column:
    """GeoJSON string"""
    ...


@h3fn(h3=h3.are_neighbor_cells, return_type=_BOOL, cells=("cell1", "cell2"))
def are_neighbor_cells(cell1: ColumnOrName, cell2: ColumnOrName) -> Column: ...


@h3fn(
    h3=h3.grid_path_cells, return_type=_CELLS, cells=("start", "end"), returns="cells"
)
def grid_path_cells(start: ColumnOrName, end: ColumnOrName) -> Column: ...


@h3fn(h3=h3.get_icosahedron_faces, return_type=T.ArrayType(_INT), cells=("cell",))
def get_icosahedron_faces(cell: ColumnOrName) -> Column: ...


@h3fn(h3=h3.cell_to_local_ij, return_type=T.ArrayType(_INT), cells=("origin", "cell"))
def cell_to_local_ij(origin: ColumnOrName, cell: ColumnOrName) -> Column: ...


@h3fn(h3=h3.local_ij_to_cell, return_type=_LONG, cells=("origin",), returns="cell")
def local_ij_to_cell(
    origin: ColumnOrName, i: IntOrColumn, j: IntOrColumn
) -> Column: ...


@h3fn(
    h3=h3.cell_area, return_type=_DOUBLE, cells=("cell",), static={"unit": _area_unit}
)
def cell_area(cell: ColumnOrName, unit: AreaUnit | str = AreaUnit.KM2) -> Column: ...


__all__ = [
    "AreaUnit",
    "BOUNDARY_TYPE",
    "COORDINATE_TYPE",
    "ColumnOrName",
    "IntOrColumn",
    "LengthUnit",
    *REGISTRY,
]
