# Contributing

Every function is defined once in `src/h3spark/__init__.py` with the `@h3fn` decorator. The decorator declares the function's h3-py reference, its Spark return type, which parameters are cells (or arrays of cells, shapes, or plan-time values), and whether it returns cells:

```python
@h3fn(h3=h3.grid_disk, return_type=T.ArrayType(T.LongType()), cells=("cell",), returns="cells")
def grid_disk(cell: ColumnOrName, k: IntOrColumn) -> Column: ...
```

A body of `...` gives a Python UDF built from `h3=`. A real body is the native implementation and becomes the default; cell arguments arrive as long Columns and other arguments as Columns. `h3spark.native`, `h3spark.udf`, `h3spark.hex` and the tests in `test/test_registry.py` are all generated from these definitions. Adding a function, or reimplementing one natively, needs nothing else, unless it introduces a new parameter name, which needs an argument generator in the test.

`h3spark.native`, `h3spark.udf` and `h3spark.hex` are generated at runtime, so type checkers read their signatures from the committed stubs (`src/h3spark/*.pyi`). After adding or changing a function, regenerate them with `uv run python scripts/gen_stubs.py`; `test/test_stubs.py` fails if they're out of date. Top-level `h3spark` functions keep their real signatures through the decorator's typing. Use `hex="table"` for functions whose result only depends on a resolution, so the hex version is a precomputed lookup.

```sh
uv sync
uv run pytest
uv run black --check . && uv run flake8 .
uv run python bench/bench.py --rows 1000000  # UDF vs native vs hex timings
```

Releases are published to PyPI by CI when a pull request labelled `release:patch`, `release:minor` or `release:major` is merged.

