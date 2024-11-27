import json
from typing import List, Tuple, Union

import h3.api.numpy_int as h3

H3CellInput = Union[str, int]


def to_h3_cell(value: H3CellInput) -> int:
    if isinstance(value, str):
        return h3.str_to_int(value)
    return value


H3Shape = Union[str, List[Tuple[float, float]], List[List[Tuple[float, float]]]]


def to_h3_shape(shape: H3Shape) -> h3.H3Shape:
    # Handle [()] to LatLngPoly (then to LatLngMultiPoly)
    # Handle [[()]] to LatLngMultiPoly
    # Handle string as dict and convert to LatLngMultiPoly
    if isinstance(shape, str):
        shape = json.loads(shape)
        return h3.geo_to_h3shape(shape)

    poly = isinstance(shape[0][0], (float, int))

    if poly:
        return h3.LatLngPoly(shape)
    else:
        return h3.LatLngMultiPoly(*[h3.LatLngPoly(s) for s in shape])
