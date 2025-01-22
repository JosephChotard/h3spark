from pyspark.sql import functions as F
from pyspark.sql.column import Column

H3_RES_OFFSET = 52
H3_RES_MASK = 15 << H3_RES_OFFSET
H3_RES_MASK_NEGATIVE = ~H3_RES_MASK
H3_DIGIT_MASK = 7
MAX_H3_RES = 15
H3_PER_DIGIT_OFFSET = 3
H3_BC_OFFSET = 45
H3_BC_MASK = 127 << H3_BC_OFFSET

BASE_CELL_DATA = [
    ((1, (1, 0, 0)), 0, (0, 0)),  # base cell 0
    ((2, (1, 1, 0)), 0, (0, 0)),  # base cell 1
    ((1, (0, 0, 0)), 0, (0, 0)),  # base cell 2
    ((2, (1, 0, 0)), 0, (0, 0)),  # base cell 3
    ((0, (2, 0, 0)), 1, (-1, -1)),  # base cell 4
    ((1, (1, 1, 0)), 0, (0, 0)),  # base cell 5
    ((1, (0, 0, 1)), 0, (0, 0)),  # base cell 6
    ((2, (0, 0, 0)), 0, (0, 0)),  # base cell 7
    ((0, (1, 0, 0)), 0, (0, 0)),  # base cell 8
    ((2, (0, 1, 0)), 0, (0, 0)),  # base cell 9
    ((1, (0, 1, 0)), 0, (0, 0)),  # base cell 10
    ((1, (0, 1, 1)), 0, (0, 0)),  # base cell 11
    ((3, (1, 0, 0)), 0, (0, 0)),  # base cell 12
    ((3, (1, 1, 0)), 0, (0, 0)),  # base cell 13
    ((11, (2, 0, 0)), 1, (2, 6)),  # base cell 14
    ((4, (1, 0, 0)), 0, (0, 0)),  # base cell 15
    ((0, (0, 0, 0)), 0, (0, 0)),  # base cell 16
    ((6, (0, 1, 0)), 0, (0, 0)),  # base cell 17
    ((0, (0, 0, 1)), 0, (0, 0)),  # base cell 18
    ((2, (0, 1, 1)), 0, (0, 0)),  # base cell 19
    ((7, (0, 0, 1)), 0, (0, 0)),  # base cell 20
    ((2, (0, 0, 1)), 0, (0, 0)),  # base cell 21
    ((0, (1, 1, 0)), 0, (0, 0)),  # base cell 22
    ((6, (0, 0, 1)), 0, (0, 0)),  # base cell 23
    ((10, (2, 0, 0)), 1, (1, 5)),  # base cell 24
    ((6, (0, 0, 0)), 0, (0, 0)),  # base cell 25
    ((3, (0, 0, 0)), 0, (0, 0)),  # base cell 26
    ((11, (1, 0, 0)), 0, (0, 0)),  # base cell 27
    ((4, (1, 1, 0)), 0, (0, 0)),  # base cell 28
    ((3, (0, 1, 0)), 0, (0, 0)),  # base cell 29
    ((0, (0, 1, 1)), 0, (0, 0)),  # base cell 30
    ((4, (0, 0, 0)), 0, (0, 0)),  # base cell 31
    ((5, (0, 1, 0)), 0, (0, 0)),  # base cell 32
    ((0, (0, 1, 0)), 0, (0, 0)),  # base cell 33
    ((7, (0, 1, 0)), 0, (0, 0)),  # base cell 34
    ((11, (1, 1, 0)), 0, (0, 0)),  # base cell 35
    ((7, (0, 0, 0)), 0, (0, 0)),  # base cell 36
    ((10, (1, 0, 0)), 0, (0, 0)),  # base cell 37
    ((12, (2, 0, 0)), 1, (3, 7)),  # base cell 38
    ((6, (1, 0, 1)), 0, (0, 0)),  # base cell 39
    ((7, (1, 0, 1)), 0, (0, 0)),  # base cell 40
    ((4, (0, 0, 1)), 0, (0, 0)),  # base cell 41
    ((3, (0, 0, 1)), 0, (0, 0)),  # base cell 42
    ((3, (0, 1, 1)), 0, (0, 0)),  # base cell 43
    ((4, (0, 1, 0)), 0, (0, 0)),  # base cell 44
    ((6, (1, 0, 0)), 0, (0, 0)),  # base cell 45
    ((11, (0, 0, 0)), 0, (0, 0)),  # base cell 46
    ((8, (0, 0, 1)), 0, (0, 0)),  # base cell 47
    ((5, (0, 0, 1)), 0, (0, 0)),  # base cell 48
    ((14, (2, 0, 0)), 1, (0, 9)),  # base cell 49
    ((5, (0, 0, 0)), 0, (0, 0)),  # base cell 50
    ((12, (1, 0, 0)), 0, (0, 0)),  # base cell 51
    ((10, (1, 1, 0)), 0, (0, 0)),  # base cell 52
    ((4, (0, 1, 1)), 0, (0, 0)),  # base cell 53
    ((12, (1, 1, 0)), 0, (0, 0)),  # base cell 54
    ((7, (1, 0, 0)), 0, (0, 0)),  # base cell 55
    ((11, (0, 1, 0)), 0, (0, 0)),  # base cell 56
    ((10, (0, 0, 0)), 0, (0, 0)),  # base cell 57
    ((13, (2, 0, 0)), 1, (4, 8)),  # base cell 58
    ((10, (0, 0, 1)), 0, (0, 0)),  # base cell 59
    ((11, (0, 0, 1)), 0, (0, 0)),  # base cell 60
    ((9, (0, 1, 0)), 0, (0, 0)),  # base cell 61
    ((8, (0, 1, 0)), 0, (0, 0)),  # base cell 62
    ((6, (2, 0, 0)), 1, (11, 15)),  # base cell 63
    ((8, (0, 0, 0)), 0, (0, 0)),  # base cell 64
    ((9, (0, 0, 1)), 0, (0, 0)),  # base cell 65
    ((14, (1, 0, 0)), 0, (0, 0)),  # base cell 66
    ((5, (1, 0, 1)), 0, (0, 0)),  # base cell 67
    ((16, (0, 1, 1)), 0, (0, 0)),  # base cell 68
    ((8, (1, 0, 1)), 0, (0, 0)),  # base cell 69
    ((5, (1, 0, 0)), 0, (0, 0)),  # base cell 70
    ((12, (0, 0, 0)), 0, (0, 0)),  # base cell 71
    ((7, (2, 0, 0)), 1, (12, 16)),  # base cell 72
    ((12, (0, 1, 0)), 0, (0, 0)),  # base cell 73
    ((10, (0, 1, 0)), 0, (0, 0)),  # base cell 74
    ((9, (0, 0, 0)), 0, (0, 0)),  # base cell 75
    ((13, (1, 0, 0)), 0, (0, 0)),  # base cell 76
    ((16, (0, 0, 1)), 0, (0, 0)),  # base cell 77
    ((15, (0, 1, 1)), 0, (0, 0)),  # base cell 78
    ((15, (0, 1, 0)), 0, (0, 0)),  # base cell 79
    ((16, (0, 1, 0)), 0, (0, 0)),  # base cell 80
    ((14, (1, 1, 0)), 0, (0, 0)),  # base cell 81
    ((13, (1, 1, 0)), 0, (0, 0)),  # base cell 82
    ((5, (2, 0, 0)), 1, (10, 19)),  # base cell 83
    ((8, (1, 0, 0)), 0, (0, 0)),  # base cell 84
    ((14, (0, 0, 0)), 0, (0, 0)),  # base cell 85
    ((9, (1, 0, 1)), 0, (0, 0)),  # base cell 86
    ((14, (0, 0, 1)), 0, (0, 0)),  # base cell 87
    ((17, (0, 0, 1)), 0, (0, 0)),  # base cell 88
    ((12, (0, 0, 1)), 0, (0, 0)),  # base cell 89
    ((16, (0, 0, 0)), 0, (0, 0)),  # base cell 90
    ((17, (0, 1, 1)), 0, (0, 0)),  # base cell 91
    ((15, (0, 0, 1)), 0, (0, 0)),  # base cell 92
    ((16, (1, 0, 1)), 0, (0, 0)),  # base cell 93
    ((9, (1, 0, 0)), 0, (0, 0)),  # base cell 94
    ((15, (0, 0, 0)), 0, (0, 0)),  # base cell 95
    ((13, (0, 0, 0)), 0, (0, 0)),  # base cell 96
    ((8, (2, 0, 0)), 1, (13, 17)),  # base cell 97
    ((13, (0, 1, 0)), 0, (0, 0)),  # base cell 98
    ((17, (1, 0, 1)), 0, (0, 0)),  # base cell 99
    ((19, (0, 1, 0)), 0, (0, 0)),  # base cell 100
    ((14, (0, 1, 0)), 0, (0, 0)),  # base cell 101
    ((19, (0, 1, 1)), 0, (0, 0)),  # base cell 102
    ((17, (0, 1, 0)), 0, (0, 0)),  # base cell 103
    ((13, (0, 0, 1)), 0, (0, 0)),  # base cell 104
    ((17, (0, 0, 0)), 0, (0, 0)),  # base cell 105
    ((16, (1, 0, 0)), 0, (0, 0)),  # base cell 106
    ((9, (2, 0, 0)), 1, (14, 18)),  # base cell 107
    ((15, (1, 0, 1)), 0, (0, 0)),  # base cell 108
    ((15, (1, 0, 0)), 0, (0, 0)),  # base cell 109
    ((18, (0, 1, 1)), 0, (0, 0)),  # base cell 110
    ((18, (0, 0, 1)), 0, (0, 0)),  # base cell 111
    ((19, (0, 0, 1)), 0, (0, 0)),  # base cell 112
    ((17, (1, 0, 0)), 0, (0, 0)),  # base cell 113
    ((19, (0, 0, 0)), 0, (0, 0)),  # base cell 114
    ((18, (0, 1, 0)), 0, (0, 0)),  # base cell 115
    ((18, (1, 0, 1)), 0, (0, 0)),  # base cell 116
    ((19, (2, 0, 0)), 1, (-1, -1)),  # base cell 117
    ((19, (1, 0, 0)), 0, (0, 0)),  # base cell 118
    ((18, (0, 0, 0)), 0, (0, 0)),  # base cell 119
    ((19, (1, 0, 1)), 0, (0, 0)),  # base cell 120
    ((18, (1, 0, 0)), 0, (0, 0)),  # base cell 121
]


PENTAGON_BASE_CELLS = [
    i for i in range(len(BASE_CELL_DATA)) if BASE_CELL_DATA[i][1] == 1
]


def get_resolution(col: Column) -> Column:
    return F.shiftRight(col.bitwiseAND(H3_RES_MASK), H3_RES_OFFSET)


def __set_resolution(col: Column, res: int) -> Column:
    """Should probably not be used directly"""
    return col.bitwiseAND(H3_RES_MASK_NEGATIVE).bitwiseOR(res << H3_RES_OFFSET)


def __set_index_digit(col: Column, res: int, digit: int) -> Column:
    mask_shifted = H3_DIGIT_MASK << ((MAX_H3_RES - res) * H3_PER_DIGIT_OFFSET)
    digit_shifted = digit << ((MAX_H3_RES - res) * H3_PER_DIGIT_OFFSET)

    return col.bitwiseAND(~mask_shifted).bitwiseOR(digit_shifted)


def cell_to_parent_fixed(
    col: Column, current_resolution: int, parent_resolution: int
) -> Column:
    """No validation, assume that all values of col are source_resolution + valid. Use at your own risk :)"""
    assert current_resolution >= parent_resolution

    if current_resolution == parent_resolution:
        return col

    parent = __set_resolution(col, parent_resolution)
    for i in range(parent_resolution, current_resolution):
        parent = __set_index_digit(parent, i + 1, H3_DIGIT_MASK)
    return parent


def get_base_cell(col: Column) -> Column:
    return F.shiftRight(col.bitwiseAND(H3_BC_MASK), H3_BC_OFFSET)


def has_child_at_resolution(col: Column, child_resolution: Column) -> Column:
    parent_res = get_resolution(col)
    return (child_resolution > parent_res) & (child_resolution < F.lit(MAX_H3_RES))


def __all_resolution_digits(cell: Column):
    resolution = get_resolution(cell)
    shiftAmount = ((MAX_H3_RES - resolution) * H3_PER_DIGIT_OFFSET)._jc.toString()

    # Kind of annoying but seems like the pyspark version of shiftleft|right doesn't accept expressions for the numBits
    # I'll PR pyspark when I get the chance but this should do for now
    return F.expr(f"shiftright({cell._jc.toString()}, {shiftAmount})").bitwiseAND(
        F.expr(f"shiftleft(1, {resolution._jc.toString()} * 3)") - 1
    )


def __is_base_cell_pentagon(cell: Column):
    return cell.isin(PENTAGON_BASE_CELLS)


def is_pentagon(cell: Column):
    return __is_base_cell_pentagon(get_base_cell(cell)) & (
        __all_resolution_digits(cell) == 0
    )


def cell_to_children_size(
    cell: Column, child_resolution: Column, validate_resolution: bool = False
) -> Column:
    """
    If validate_resolution is False then it might produce incorrect results if the child_resolution is less than the
    parent resolution.
    """
    n = child_resolution - get_resolution(cell)

    if validate_resolution:
        assertion = F.assert_true(
            n >= 0,
            "Child resolution must be greater than or equal to parent resolution",
        )
    else:
        assertion = F.lit(None)

    return F.floor(
        F.when(
            assertion.isNull() & is_pentagon(cell), 1 + 5 * (F.pow(7, n) - 1) / 6
        ).otherwise(F.pow(7, n))
    )


def str_to_int(col: Column) -> Column:
    """Performs no validation"""
    return F.conv(col, 16, 10).cast("long")


def int_to_str(col: Column) -> Column:
    """Performs no validation"""
    return F.lower(F.hex(col))
