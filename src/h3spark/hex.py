"""
Hex string versions of the h3spark API, for data that stores cells as H3 strings (e.g. ``"8928308280fffff"``).

``h3spark.hex.<name>`` mirrors ``h3spark.<name>`` but takes cells as hex strings and returns cells as lowercase hex
strings (arrays of cells become ``array<string>``). Other arguments and non-cell results are unchanged. Cells are
converted with the native ``str_to_int`` / ``int_to_str``, so natively implemented functions stay entirely in the JVM.
Like those, the conversions perform no validation.

    import h3spark.hex as h3hex
    df.select(h3hex.cell_to_parent("cell_str", 5))
"""

import functools

from pyspark.sql import functions as F
from pyspark.sql.column import Column

from . import REGISTRY, int_to_str, str_to_int
from ._registry import H3Function, to_col


def _hexify(entry: H3Function):
    func = entry.default
    if not (entry.cells or entry.cell_lists or entry.returns):
        return func

    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> Column:
        bound = entry.signature.bind(*args, **kwargs)
        for name in entry.cells:
            if name in bound.arguments:
                bound.arguments[name] = str_to_int(bound.arguments[name])
        for name in entry.cell_lists:
            if name in bound.arguments:
                bound.arguments[name] = F.transform(
                    to_col(bound.arguments[name]), str_to_int
                )
        result = func(*bound.args, **bound.kwargs)
        if entry.returns == "cell":
            return int_to_str(result)
        if entry.returns == "cells":
            return F.transform(result, int_to_str)
        return result

    wrapper.__module__ = __name__
    return wrapper


__all__ = sorted(name for name, entry in REGISTRY.items() if entry.hex)
globals().update({name: _hexify(REGISTRY[name]) for name in __all__})
