from pyspark.sql import Column
from pyspark.sql import functions as F

H3_RES_OFFSET = 52
H3_RES_MASK = 15 << H3_RES_OFFSET


def get_resolution(col: Column) -> Column:
    """Column must be of long type"""
    return F.shiftRight(col.bitwiseAND(H3_RES_MASK), H3_RES_OFFSET)


# TODO: Add h3ToParent, h3ToChildren, h3MaxChild, h3MinChild, etc
