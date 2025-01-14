from pyspark.sql import Column
from pyspark.sql import functions as F

H3_RES_OFFSET = 52
H3_RES_MASK = 15 << H3_RES_OFFSET
H3_RES_MASK_NEGATIVE = ~H3_RES_MASK
H3_DIGIT_MASK = 7
MAX_H3_RES = 15
H3_PER_DIGIT_OFFSET = 3



def get_resolution(col: Column) -> Column:
    return F.shiftRight(col.bitwiseAND(H3_RES_MASK), H3_RES_OFFSET)

def __set_resolution(col: Column, res: int) -> Column:
    """Should probably not be used directly"""
    return col.bitwiseAND(H3_RES_MASK_NEGATIVE).bitwiseOR(
        res << H3_RES_OFFSET)

def __set_index_digit(col: Column, res: int, digit: int) -> Column:
    mask_shifted = H3_DIGIT_MASK << ((MAX_H3_RES - res) * H3_PER_DIGIT_OFFSET)
    digit_shifted = digit << ((MAX_H3_RES - res) * H3_PER_DIGIT_OFFSET)
    
    return col.bitwiseAND(~mask_shifted).bitwiseOR(digit_shifted)


def cell_to_parent_fixed(col: Column, current_resolution: int, parent_resolution: int) -> Column:
    """No validation, assume that all values of col are source_resolution + valid. Use at your own risk :)"""
    assert current_resolution >= parent_resolution

    if current_resolution == parent_resolution:
        return col

    parent = __set_resolution(col, parent_resolution)
    for i in range(parent_resolution, current_resolution):
        parent = __set_index_digit(parent, i + 1, H3_DIGIT_MASK)
    return parent
