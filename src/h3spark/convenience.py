import h3.api.numpy_int as h3
from pyspark.sql import functions as F
from pyspark.sql import types as T

from . import cell_to_center_child
from .utils import H3CellInput, H3Shape, to_h3_cell, to_h3_shape


def min_child(cell_id: H3CellInput, resolution: int):
    """Prefer h3spark.native.minchild for long inputs"""
    return cell_to_center_child(cell_id, resolution)


@F.udf(T.LongType(), useArrow=True)
def max_child(cell_id: H3CellInput, resolution: int):
    """Prefer h3spark.native.maxchild for long inputs"""
    cell = to_h3_cell(cell_id)
    return h3.child_pos_to_cell(
        cell, resolution, h3.cell_to_children_size(cell, resolution) - 1
    )


@F.udf(T.ArrayType(T.LongType()), useArrow=True)
def h3shape_to_cells_compacted(shape: H3Shape, res: int):
    cells = h3.h3shape_to_cells(to_h3_shape(shape), res)
    compacted = h3.compact_cells(cells)
    return compacted.tolist()
