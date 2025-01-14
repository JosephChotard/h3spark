from pyspark.sql import functions as F

from . import cell_to_children_size, child_pos_to_cell
from .utils import H3CellInput


def min_child(cell_id: H3CellInput, resolution: int):
    return child_pos_to_cell(cell_id, resolution, F.lit(0))


def max_child(cell_id: H3CellInput, resolution: int):
    return child_pos_to_cell(
        cell_id, resolution, cell_to_children_size(cell_id, resolution) - 1
    )
