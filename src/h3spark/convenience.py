import h3.api.numpy_int as h3
from pyspark.sql import functions as F
from pyspark.sql import types as T

from . import cell_to_children_size, child_pos_to_cell
from .utils import H3CellInput, H3Shape, to_h3_shape


def min_child(cell_id: H3CellInput, resolution: int):
    return child_pos_to_cell(cell_id, resolution, F.lit(0))


def max_child(cell_id: H3CellInput, resolution: int):
    return child_pos_to_cell(
        cell_id, resolution, cell_to_children_size(cell_id, resolution) - 1
    )


@F.udf(T.ArrayType(T.LongType()))
def h3shape_to_cells_compacted(shape: H3Shape, res: int):
    cells = h3.h3shape_to_cells(to_h3_shape(shape), res)
    compacted = h3.compact_cells(cells)
    return compacted.tolist()
