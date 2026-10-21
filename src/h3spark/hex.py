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

import h3.api.numpy_int as h3

from . import REGISTRY, int_to_str, str_to_int
from . import _bits as B
from ._registry import H3Function, column_arg, named, to_col


def _to_hex(value):
    return (
        [_to_hex(v) for v in value] if isinstance(value, list) else h3.int_to_str(value)
    )


def _hex_table(entry: H3Function):
    """
    Hex version of a function of (at most) a resolution: a lookup in hex strings precomputed at import, which Spark
    constant-folds like the native version
    """
    params = list(entry.signature.parameters)
    if not params:
        values = _to_hex(entry.reference())
        return functools.wraps(entry.default)(
            lambda: named(B.literal_array(values), entry.name, [])
        )
    (param,) = params
    table = [_to_hex(entry.reference(r)) for r in B.RESOLUTIONS]

    @functools.wraps(entry.default)
    def wrapper(*args, **kwargs) -> Column:
        bound = entry.signature.bind(*args, **kwargs)
        res = bound.arguments[param]
        return named(B.lookup(table, column_arg(res)), entry.name, [res])

    return wrapper


def _hexify(entry: H3Function):
    func = entry.default
    if entry.hex == "table":
        return _hex_table(entry)
    if not (entry.cells or entry.cell_lists or entry.returns):
        return func
    params = [p for p in entry.signature.parameters if p not in entry.static]

    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> Column:
        bound = entry.signature.bind(*args, **kwargs)
        values = [bound.arguments.get(p) for p in params]
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
            result = int_to_str(result)
        elif entry.returns == "cells":
            result = F.transform(result, int_to_str)
        return named(result, entry.name, values)

    wrapper.__module__ = __name__
    return wrapper


__all__ = sorted(name for name, entry in REGISTRY.items() if entry.hex)
globals().update({name: _hexify(REGISTRY[name]) for name in __all__})
