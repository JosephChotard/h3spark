"""
Single source of truth for h3spark functions.

Each logical H3 function is defined once with ``@h3fn``. The decorator records its h3-py reference, which parameters
are cells / lists of cells / shapes / plan-time ("static") values and what it returns, and builds:

- a Python UDF backed by the h3-py reference (always, exposed as ``h3spark.udf.<name>``)
- a native implementation from the function body, unless the body is just ``...`` (exposed as ``h3spark.native``)

The decorated name is bound to the default implementation: native if there is one, else the UDF. ``h3spark.native``,
``h3spark.udf``, ``h3spark.hex`` and the tests are all generated from ``REGISTRY``.
"""

import functools
import inspect
import json
import operator
import re
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from typing import Any, Callable, Literal, ParamSpec, TypeAlias, TypeVar, cast

import h3.api.numpy_int as h3
from pyspark.sql import functions as F
from pyspark.sql import types as T
from pyspark.sql.column import Column

from . import _bits as B
from .utils import to_h3_cell, to_h3_shape

try:
    # Spark Connect columns only subclass Column from Spark 4
    from pyspark.sql.connect.column import Column as ConnectColumn

    COLUMN_TYPES: tuple[type, ...] = (Column, ConnectColumn)
except ImportError:  # Spark Connect dependencies aren't installed
    COLUMN_TYPES = (Column,)

ColumnOrName: TypeAlias = Column | str
IntOrColumn: TypeAlias = int | ColumnOrName
Returns = Literal["cell", "cells"] | None
E = TypeVar("E", bound=Enum)
P = ParamSpec("P")


@dataclass(frozen=True)
class H3Function:
    name: str
    signature: inspect.Signature
    udf: Callable[..., Column]
    native: Callable[..., Column] | None
    # The native implementation without the cell type checks and result name, for callers that already have long
    # cell columns (h3spark.hex). Shallower, see _bits.balanced.
    native_unchecked: Callable[..., Column] | None
    # Python version taking the same arguments as the Spark functions (cells as ints, values as Spark returns them)
    reference: Callable[..., Any]
    return_type: T.DataType
    cells: tuple[str, ...]
    cell_lists: tuple[str, ...]
    static: tuple[str, ...]
    returns: Returns
    # Whether h3spark.hex exposes a hex string version, "table" to precompute it per resolution
    hex: bool | Literal["table"]

    @property
    def default(self) -> Callable[..., Column]:
        return self.native or self.udf


REGISTRY: dict[str, H3Function] = {}


def to_col(col: ColumnOrName) -> Column:
    if isinstance(col, COLUMN_TYPES):
        return col
    if isinstance(col, str):
        return F.col(col)
    raise TypeError(f"Expected a Column or a column name, got {type(col).__name__}")


def column_arg(value: Any) -> Column:
    """Columns pass through, strings are column names, anything else is a literal"""
    if isinstance(value, (*COLUMN_TYPES, str)):
        return to_col(value)
    return F.lit(value)


def cell_check(col: Column, name: str, has_hex: bool, array: bool = False) -> Column:
    """
    Always false, but makes Spark fail when it analyses the plan (e.g. on df.select) unless col is an integral cell
    column (or array of them), instead of producing nulls or failing once rows are processed
    """
    expected = "array<bigint>" if array else "bigint"
    hint = f"h3spark.hex.{name} or " if has_hex else ""
    message = f"{name} expects {expected} cells (for hex string cells use {hint}h3spark.str_to_int)"
    # bitwise_not only accepts integral types. Naming a struct field after the message puts it in Spark's error.
    probe = F.struct(col.alias(message))[message]
    if array:
        probe = probe[0]
    return F.lit(False) & F.bitwise_not(probe).isNull()


def with_checks(result: Column, checks: list[Column]) -> Column:
    """
    Puts checks in a branch that's never taken, which the optimizer removes. Beside the result rather than around
    each argument, so they don't nest the expression deeper: Spark Connect rejects deeply nested expressions.
    """
    if not checks:
        return result
    return F.when(B.balanced(operator.or_, checks), F.lit(None)).otherwise(result)


def _arg_name(value: Any) -> str:
    """How Spark shows an argument in a column name, e.g. cell for F.col("cell")"""
    if not isinstance(value, COLUMN_TYPES):
        return str(value)
    text = re.sub(r"^Column<'(.*)'>$", r"\1", str(value), flags=re.S)
    alias = re.search(r" AS (?:`([^`]*)`|(\w+))$", text)
    return (alias.group(1) or alias.group(2)) if alias else text


def named(result: Column, name: str, args: list) -> Column:
    """Names a result like a UDF call, e.g. cell_to_parent(cell, 5), instead of after the whole expression"""
    return result.alias(f"{name}({', '.join(map(_arg_name, args))})")


def unit_value(unit_type: type[E]) -> Callable[[E | str], str]:
    """Static converter accepting an enum member or its value"""

    def convert(unit: E | str) -> str:
        if isinstance(unit, COLUMN_TYPES):
            raise TypeError(
                "unit must be a plain Python value (e.g. 'km^2' or AreaUnit.KM2), not a Column"
            )
        return unit_type(unit).value

    return convert


def _stub(): ...


def _doc_stub():
    """doc"""


_STUB_CODES = {_stub.__code__.co_code, _doc_stub.__code__.co_code}


def _is_stub(func: Callable) -> bool:
    """True if the body is only a docstring and/or `...`"""
    code = func.__code__
    return code.co_code in _STUB_CODES and all(
        c is None or c == func.__doc__ for c in code.co_consts
    )


def _to_spark_value(value: Any, array: bool) -> Any:
    if hasattr(value, "tolist"):  # numpy arrays and scalars
        return value.tolist()
    if isinstance(value, h3.H3Shape):
        return json.dumps(value.__geo_interface__)
    if array and isinstance(value, tuple):
        return list(value)
    return value


def _cell_list(cells: list) -> list[int]:
    return [to_h3_cell(c) for c in cells]


def h3fn(
    *,
    h3: Callable,
    return_type: T.DataType,
    cells: tuple[str, ...] = (),
    cell_lists: tuple[str, ...] = (),
    shapes: tuple[str, ...] = (),
    static: dict[str, Callable[[Any], Any]] | None = None,
    returns: Returns = None,
    hex: bool | Literal["table"] = True,
) -> Callable[[Callable[P, Column]], Callable[P, Column]]:
    """
    h3: per-row Python implementation, called positionally with the non-static arguments (cells as ints).
    return_type: Spark type of the result.
    cells / cell_lists / shapes: parameters holding a cell (long, or int|hex string in the UDF), an array of cells,
        or a shape (see to_h3_shape, UDF only).
    static: plan-time parameters (not columns) mapped to a converter. The converted value is passed to the native body
        and, if `h3` accepts a parameter of that name, as a keyword argument to `h3`. One UDF is built per value.
    returns: "cell" / "cells" if the result is a cell / array of cells (used by h3spark.hex).
    hex: whether h3spark.hex exposes a hex string version. "table" for functions whose result only depends on a
        resolution (or nothing): the hex version looks up hex strings precomputed from `h3` for every resolution, so
        Spark can constant-fold it instead of converting every row.

    Every other argument becomes a Column: column names and plain Python literals (e.g. res=9) are accepted.
    """
    static = static or {}

    def decorator(func: Callable[P, Column]) -> Callable[P, Column]:
        signature = inspect.signature(func)
        unknown = {*cells, *cell_lists, *shapes, *static} - set(signature.parameters)
        if unknown:
            raise ValueError(f"{func.__name__} has no parameters {sorted(unknown)}")
        column_params = [p for p in signature.parameters if p not in static]
        h3_kwargs = set(inspect.signature(h3).parameters) & set(static)

        def bind(args: tuple, kwargs: dict) -> inspect.BoundArguments:
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            for p, convert in static.items():
                bound.arguments[p] = convert(bound.arguments[p])
            return bound

        # UDF backed by h3
        row_converters = {
            **dict.fromkeys(cells, to_h3_cell),
            **dict.fromkeys(cell_lists, _cell_list),
            **dict.fromkeys(shapes, to_h3_shape),
        }
        converters = [row_converters.get(p) for p in column_params]
        array = isinstance(return_type, T.ArrayType)
        row_signature = inspect.Signature(
            [
                inspect.Parameter(p, inspect.Parameter.POSITIONAL_ONLY)
                for p in column_params
            ]
        )

        def call(values: tuple, h3_kw: dict) -> Any:
            args = [c(v) if c else v for c, v in zip(converters, values)]
            return _to_spark_value(h3(*args, **h3_kw), array)

        @lru_cache
        def build_udf(h3_items: tuple):
            h3_kw = dict(h3_items)

            def row(*values):
                return call(values, h3_kw)

            # Spark only enables Arrow for functions with named arguments
            row.__signature__ = row_signature
            # Spark names UDF results after the function
            row.__name__ = row.__qualname__ = func.__name__
            # Arrow is slower than pickling for arrays
            return F.udf(row, return_type, useArrow=not array)

        def split(bound: inspect.BoundArguments) -> tuple[list, tuple]:
            values = [bound.arguments[p] for p in column_params]
            return values, tuple((p, bound.arguments[p]) for p in sorted(h3_kwargs))

        @functools.wraps(func)
        def udf(*args, **kwargs) -> Column:
            values, h3_items = split(bind(args, kwargs))
            return build_udf(h3_items)(*(column_arg(v) for v in values))

        def reference(*args, **kwargs) -> Any:
            values, h3_items = split(bind(args, kwargs))
            return call(values, dict(h3_items))

        # Native implementation from the body
        @functools.wraps(func)
        def native(*args, **kwargs) -> Column:
            bound = bind(args, kwargs)
            values = [bound.arguments[p] for p in column_params]
            checks = []
            for p in column_params:
                value = bound.arguments[p]
                if p in cells or p in cell_lists:
                    array = p in cell_lists
                    value = to_col(value)
                    checks.append(cell_check(value, func.__name__, hex, array))
                    value = value.cast("array<bigint>" if array else "bigint")
                bound.arguments[p] = column_arg(value)
            result = with_checks(func(*bound.args, **bound.kwargs), checks)
            return named(result, func.__name__, values)

        def native_unchecked(*args, **kwargs) -> Column:
            bound = bind(args, kwargs)
            for p in column_params:
                bound.arguments[p] = column_arg(bound.arguments[p])
            return func(*bound.args, **bound.kwargs)

        if func.__name__ in REGISTRY:
            raise ValueError(f"{func.__name__} is already registered")
        entry = H3Function(
            name=func.__name__,
            signature=signature,
            udf=udf,
            native=None if _is_stub(func) else native,
            native_unchecked=None if _is_stub(func) else native_unchecked,
            reference=reference,
            return_type=return_type,
            cells=cells,
            cell_lists=cell_lists,
            static=tuple(static),
            returns=returns,
            hex=hex,
        )
        REGISTRY[entry.name] = entry
        # Typed as the decorated function, so type checkers see its real signature
        return cast(Callable[P, Column], entry.default)

    return decorator
