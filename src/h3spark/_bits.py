"""Bit layout of H3 cell indexes and helpers shared by the native implementations"""

import h3.api.numpy_int as h3
from pyspark.sql import functions as F
from pyspark.sql.column import Column

# Top 8 bits of a cell: high bit 0, mode 1 (cell), reserved bits 0
H3_HEADER_OFFSET = 56
H3_CELL_HEADER = 1 << (59 - H3_HEADER_OFFSET)
H3_RES_OFFSET = 52
H3_RES_MASK = 15 << H3_RES_OFFSET
H3_RES_MASK_NEGATIVE = ~H3_RES_MASK
MAX_H3_RES = 15
H3_PER_DIGIT_OFFSET = 3
H3_BC_OFFSET = 45
H3_BC_MASK = 127 << H3_BC_OFFSET
NUM_BASE_CELLS = 122
# Lowest bit of every digit
MAX_CHILD_MASK = 0b001001001001001001001001001001001001001001001
H3_DIGITS_MASK = (1 << H3_BC_OFFSET) - 1
RESOLUTIONS = range(MAX_H3_RES + 1)

PENTAGON_BASE_CELLS = sorted(
    h3.get_base_cell_number(int(p)) for p in h3.get_pentagons(0)
)
POW7 = [7**n for n in RESOLUTIONS]

# Bits of every digit after resolution r, i.e. the digits that are unused (set to 7) for a cell at resolution r
DIGITS_AFTER_RES = [
    (1 << ((MAX_H3_RES - r) * H3_PER_DIGIT_OFFSET)) - 1 for r in RESOLUTIONS
]

# Number of descendants n resolutions below a hexagon / pentagon (a pentagon has 1 pentagon + 5 hexagon children)
HEXAGON_CHILDREN_SIZE = [7**n for n in RESOLUTIONS]
PENTAGON_CHILDREN_SIZE = [1 + 5 * (7**n - 1) // 6 for n in RESOLUTIONS]
# Position of the n-levels-down descendants of a pentagon among its children = their base 7 digits as a number, minus
# this offset (see child_pos). For n = 0: pentagon children skip digit 1, so digit 2 is position 1.
PENTAGON_POS_OFFSET = [(7 ** (n + 1) - 1) // 6 for n in RESOLUTIONS]


def literal_array(values: list) -> Column:
    return F.array(
        *[literal_array(v) if isinstance(v, list) else F.lit(v) for v in values]
    )


def lookup(values: list, index: Column) -> Column:
    """
    Looks up a per-resolution constant. Spark constant-folds the literal array, so this avoids both UDFs and
    building SQL expression strings to shift by a column amount. Out of range indexes return null, also in ANSI mode.
    """
    return F.get(literal_array(values), index.cast("int"))


def resolution(cell: Column) -> Column:
    return F.shiftright(cell.bitwiseAND(H3_RES_MASK), H3_RES_OFFSET).cast("int")


def base_cell(cell: Column) -> Column:
    return F.shiftright(cell.bitwiseAND(H3_BC_MASK), H3_BC_OFFSET).cast("int")


def digits_after_res(res: Column) -> Column:
    return lookup(DIGITS_AFTER_RES, res)


def digits_between(cell: Column, child_res: Column) -> Column:
    """Bits of the digits for resolutions in (resolution(cell), child_res]"""
    return digits_after_res(resolution(cell)).bitwiseXOR(digits_after_res(child_res))


def parent(cell: Column, res: Column) -> Column:
    return set_resolution(cell.bitwiseOR(digits_after_res(res)), res)


def set_resolution(cell: Column, res: Column) -> Column:
    return cell.bitwiseAND(H3_RES_MASK_NEGATIVE).bitwiseOR(
        F.shiftleft(res.cast("long"), H3_RES_OFFSET)
    )


def is_pentagon(cell: Column) -> Column:
    used_digits = F.lit(H3_DIGITS_MASK).bitwiseXOR(digits_after_res(resolution(cell)))
    return base_cell(cell).isin(PENTAGON_BASE_CELLS) & (
        cell.bitwiseAND(used_digits) == 0
    )


def _count_at_least(value: Column, thresholds: list[int]) -> Column:
    return sum((value >= t).cast("int") for t in thresholds)


def _digit(cell: Column, r: int) -> Column:
    """Digit for resolution r (1-15)"""
    return F.shiftright(cell, (MAX_H3_RES - r) * H3_PER_DIGIT_OFFSET).bitwiseAND(7)


def is_valid_cell(cell: Column) -> Column:
    """Same checks as h3's isValidCell"""
    res = resolution(cell)
    unused = digits_after_res(res)
    used = cell.bitwiseAND(F.lit(H3_DIGITS_MASK).bitwiseXOR(unused))
    # The lowest bit of a digit stays set iff all 3 of its bits are, i.e. the digit is 7
    sevens = used.bitwiseAND(F.shiftright(used, 1)).bitwiseAND(F.shiftright(used, 2))
    # Pentagons have no digit 1 subsequence: their first non-zero digit can't be 1
    leading_one = F.lit(False)
    for d in range(MAX_H3_RES):
        leading_one = leading_one | used.between(1 << (3 * d), (2 << (3 * d)) - 1)
    return (
        (F.shiftright(cell, H3_HEADER_OFFSET) == H3_CELL_HEADER)
        & (base_cell(cell) < NUM_BASE_CELLS)
        & (sevens.bitwiseAND(MAX_CHILD_MASK) == 0)
        & (cell.bitwiseAND(unused) == unused)
        & ~(base_cell(cell).isin(PENTAGON_BASE_CELLS) & leading_one)
    )


def children_size(pentagon: Column, levels: Column) -> Column:
    """Number of descendants levels resolutions down, null if out of range. One lookup rather than a conditional"""
    table = HEXAGON_CHILDREN_SIZE + PENTAGON_CHILDREN_SIZE
    index = levels + pentagon.cast("int") * len(RESOLUTIONS)
    return lookup(table, index) + _null_unless(levels.between(0, MAX_H3_RES))


def _null_unless(condition: Column) -> Column:
    """0, or null if condition is false. Adding it nulls a result without putting it in a conditional branch, which
    would stop Spark from computing its repeated subexpressions only once."""
    return F.when(condition, F.lit(0))


def child_pos(child: Column, parent_res: Column) -> Column:
    """
    Position of child among the descendants of its parent at parent_res, ordered by index. For a hexagon parent that's
    the digits after parent_res read as a base 7 number. A pentagon parent has no digit 1 children while its
    descendants are pentagons, i.e. until the first non-zero digit, which shifts positions by PENTAGON_POS_OFFSET.
    """
    child_res = resolution(child)
    digits = child.bitwiseAND(
        digits_after_res(parent_res).bitwiseXOR(digits_after_res(child_res))
    )
    # Base 7 number with the digits up to child_res, divided by the (zero) digits after it. The division is exact.
    base7 = sum(
        _digit(digits, r) * POW7[MAX_H3_RES - r] for r in range(1, MAX_H3_RES + 1)
    )
    pos = (base7 / lookup(POW7, MAX_H3_RES - child_res)).cast("long")
    # Digits after the first non-zero one: the first non-zero digit is at resolution 15 - (digits after it at res 15)
    levels = (
        child_res
        - MAX_H3_RES
        + _count_at_least(digits, [8**k for k in range(1, MAX_H3_RES)])
    )
    pentagon = (is_pentagon(parent(child, parent_res)) & (digits != 0)).cast("long")
    offset = pentagon * F.coalesce(lookup(PENTAGON_POS_OFFSET, levels), F.lit(0))
    return pos - offset + _null_unless(parent_res <= child_res)


def child_at_pos(parent: Column, child_res: Column, pos: Column) -> Column:
    """Inverse of child_pos. Null if pos is out of range"""
    levels = child_res - resolution(parent)
    pentagon = is_pentagon(parent)
    size = children_size(pentagon, levels)
    pos = pos.cast("long")
    pentagon_levels = _count_at_least(pos, PENTAGON_CHILDREN_SIZE[1:])
    offset = lookup(PENTAGON_POS_OFFSET, pentagon_levels)
    base7 = pos + (pentagon & (pos > 0)).cast("long") * offset
    # Base 7 digits to 3 bit digits, with the last one at bit 0. Exact in double: base7 < 7^15 < 2^53.
    packed = sum(
        ((base7 / POW7[i]).cast("long") % 7) * (8**i) for i in range(MAX_H3_RES)
    )
    digits = packed * lookup([8 ** (MAX_H3_RES - r) for r in RESOLUTIONS], child_res)
    cleared = parent.bitwiseAND(F.bitwise_not(digits_between(parent, child_res)))
    child = set_resolution(cleared.bitwiseOR(digits), child_res)
    return child + _null_unless((pos >= 0) & (pos < size))
