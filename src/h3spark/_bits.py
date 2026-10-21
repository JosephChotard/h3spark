"""Bit layout of H3 cell indexes and helpers shared by the native implementations"""

import h3.api.numpy_int as h3
from pyspark.sql import functions as F
from pyspark.sql.column import Column

H3_RES_OFFSET = 52
H3_RES_MASK = 15 << H3_RES_OFFSET
H3_RES_MASK_NEGATIVE = ~H3_RES_MASK
MAX_H3_RES = 15
H3_PER_DIGIT_OFFSET = 3
H3_BC_OFFSET = 45
H3_BC_MASK = 127 << H3_BC_OFFSET
MAX_CHILD_MASK = 0b001001001001001001001001001001001001001001001
H3_DIGITS_MASK = (1 << H3_BC_OFFSET) - 1
RESOLUTIONS = range(MAX_H3_RES + 1)

PENTAGON_BASE_CELLS = sorted(
    h3.get_base_cell_number(int(p)) for p in h3.get_pentagons(0)
)

# Bits of every digit after resolution r, i.e. the digits that are unused (set to 7) for a cell at resolution r
DIGITS_AFTER_RES = [
    (1 << ((MAX_H3_RES - r) * H3_PER_DIGIT_OFFSET)) - 1 for r in RESOLUTIONS
]

# Number of descendants n resolutions below a hexagon / pentagon (a pentagon has 1 pentagon + 5 hexagon children)
HEXAGON_CHILDREN_SIZE = [7**n for n in RESOLUTIONS]
PENTAGON_CHILDREN_SIZE = [1 + 5 * (7**n - 1) // 6 for n in RESOLUTIONS]


def literal_array(values: list) -> Column:
    return F.array(
        *[literal_array(v) if isinstance(v, list) else F.lit(v) for v in values]
    )


def lookup(values: list, index: Column) -> Column:
    """
    Looks up a per-resolution constant. Spark constant-folds the literal array, so this avoids both UDFs and
    building SQL expression strings to shift by a column amount. Out of range indexes return null (or raise in
    ANSI mode).
    """
    return literal_array(values)[index.cast("long")]


def resolution(cell: Column) -> Column:
    return F.shiftright(cell.bitwiseAND(H3_RES_MASK), H3_RES_OFFSET).cast("int")


def base_cell(cell: Column) -> Column:
    return F.shiftright(cell.bitwiseAND(H3_BC_MASK), H3_BC_OFFSET).cast("int")


def digits_after_res(res: Column) -> Column:
    return lookup(DIGITS_AFTER_RES, res)


def digits_between(cell: Column, child_res: Column) -> Column:
    """Bits of the digits for resolutions in (resolution(cell), child_res]"""
    return digits_after_res(resolution(cell)).bitwiseXOR(digits_after_res(child_res))


def set_resolution(cell: Column, res: Column) -> Column:
    return cell.bitwiseAND(H3_RES_MASK_NEGATIVE).bitwiseOR(
        F.shiftleft(res.cast("long"), H3_RES_OFFSET)
    )


def is_pentagon(cell: Column) -> Column:
    used_digits = F.lit(H3_DIGITS_MASK).bitwiseXOR(digits_after_res(resolution(cell)))
    return base_cell(cell).isin(PENTAGON_BASE_CELLS) & (
        cell.bitwiseAND(used_digits) == 0
    )
