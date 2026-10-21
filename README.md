# h3spark

![Tile the world in hexes](images/big_geo.jpeg "Tile the world in hexes")

`h3spark` is a Python library that provides a set of native and user-defined functions (UDFs) for working with H3 geospatial indexing in PySpark. The functions in this library follow the same assumptions and rules as the native H3 functions, allowing for seamless integration and usage in PySpark data pipelines.

Where possible, functions are implemented natively with Spark SQL expressions. This avoids the serialization/deserialization overhead of a Python UDF, at the cost of less validation than h3-py.

## Installation

Requires PySpark 3.5+. UDFs returning scalars or structs run with Arrow optimization enabled (`useArrow=True`), so `pandas` and `pyarrow` are installed via `pyspark[sql]`.

You can install `h3spark` using either pip or conda.

### Using pip
```bash
pip install h3spark
```

### Using conda
```bash
conda install -c conda-forge h3spark
```

## Usage

```python
from pyspark.sql import functions as F
import h3spark

df.select(
    h3spark.latlng_to_cell("lat", "lng", 9).alias("cell"),     # column names or Columns, plain Python ints
    h3spark.cell_to_parent(F.col("cell"), 5).alias("parent"),  # native Spark expressions, no Python UDF
    h3spark.cell_area("cell", "m^2").alias("area"),            # unit is a plain Python value
)
```

Every function comes in several flavours:

- **`h3spark.<fn>`**: the default. Natively implemented functions (see below) are pure Spark SQL expressions, which avoid the serialization overhead of Python UDFs. Everything else is a Python UDF calling h3-py.
- **`h3spark.native.<fn>`**: only the natively implemented functions. The 0.2 names `minchild`, `maxchild` and `get_base_cell` still work.
- **`h3spark.udf.<fn>`**: always the Python UDF backed by h3-py, for every function. Use it when you want h3-py's validation, or a reference to compare against.
- **`h3spark.hex.<fn>`**: the default, but taking and returning cells as hex strings (see below).

Arguments:

- Column arguments accept a `Column` or a column name (`str`). Other arguments, such as `res` or `k`, also accept plain Python values (`res=9`, no `F.lit` needed).
- `unit` arguments (`AreaUnit`/`LengthUnit` or their string values) are plain Python values resolved when the plan is built, not columns.
- Cells are 64-bit integers (`LongType`). Native functions accept integral cell columns (`bigint`, `int`, ...) and raise an error for any other type, such as hex strings, even if every value is null. Convert hex strings with `str_to_int` or use `h3spark.hex`. Python UDFs still accept either long or hex string cells (`H3CellInput = Union[str, int]`).

Types: resolutions and base cell numbers are `int`, cells and counts are `long` (`bigint`), and areas, lengths and coordinates are `double`. `COORDINATE_TYPE` is `struct<lat: double, lon: double>` and `BOUNDARY_TYPE` is an array of it.

### Natively implemented functions

These follow the same rules as h3-py and return the same values for valid cells and in-range resolutions, but do far less validation. Invalid cells or resolutions (e.g. a parent resolution finer than the cell's) produce null or meaningless values instead of raising. Use `h3spark.udf.<fn>` if you need h3-py's errors.

- **`str_to_int(h3_str: string) -> long`**: Converts a hex string cell (any case) to its integer representation.
- **`int_to_str(h3_int: long) -> string`**: Converts a cell to its lowercase hex string representation.
- **`get_resolution(cell: long) -> int`**: Resolution of a cell.
- **`get_base_cell_number(cell: long) -> int`**: Base cell number of a cell.
- **`is_pentagon(cell: long) -> bool`**: Whether a cell is a pentagon.
- **`is_res_class_III(cell: long) -> bool`**: Whether a cell has a class III resolution.
- **`cell_to_parent(cell: long, res: int) -> long`**: Parent of a cell at a resolution.
- **`cell_to_center_child(cell: long, res: int) -> long`**: Center child of a cell at a resolution.
- **`cell_to_children_size(cell: long, res: int, validate_resolution: bool = False) -> long`**: Number of children of a cell at a resolution. Null if `res` is coarser than the cell's resolution, or an error if `validate_resolution` is True.
- **`min_child(cell: long, resolution: int) -> long`**: Child with the lowest index at a resolution (the center child).
- **`max_child(cell: long, resolution: int) -> long`**: Child with the highest index at a resolution.
- **`is_childof(child: long, parent: long) -> bool`**: Whether `child` is `parent` or one of its descendants.
- **`has_child_at_resolution(cell: long, child_resolution: int) -> bool`**: Whether a cell has children at a resolution.
- **`get_num_cells(res: int) -> long`**: Number of cells at a resolution.
- **`average_hexagon_area(res: int, unit: AreaUnit | str = AreaUnit.KM2) -> double`**: Average hexagon area at a resolution. h3 only supports `km^2` and `m^2`.
- **`average_hexagon_edge_length(res: int, unit: LengthUnit | str = LengthUnit.KM) -> double`**: Average hexagon edge length at a resolution. h3 only supports `km` and `m`.
- **`get_pentagons(res: int) -> array<long>`**: All pentagons at a resolution.
- **`get_res0_cells() -> array<long>`**: All resolution 0 cells.

### Python UDF functions

- **`latlng_to_cell(lat: double, lng: double, res: int) -> long`**: Cell containing a point.
- **`cell_to_latlng(cell: H3CellInput) -> COORDINATE_TYPE`**: Center of a cell.
- **`cell_to_boundary(cell: H3CellInput) -> BOUNDARY_TYPE`**: Boundary of a cell.
- **`grid_distance(cell1: H3CellInput, cell2: H3CellInput) -> int`**: Grid distance between two cells.
- **`grid_disk(cell: H3CellInput, k: int) -> array<long>`**: Cells within `k` steps of a cell.
- **`grid_ring(cell: H3CellInput, k: int) -> array<long>`**: Cells exactly `k` steps from a cell.
- **`grid_path_cells(start: H3CellInput, end: H3CellInput) -> array<long>`**: Grid path between two cells.
- **`are_neighbor_cells(cell1: H3CellInput, cell2: H3CellInput) -> bool`**: Whether two cells are neighbors.
- **`cell_to_children(cell: H3CellInput, res: int) -> array<long>`**: Children of a cell at a resolution.
- **`cell_to_child_pos(child: H3CellInput, res_parent: int) -> long`**: Position of a cell among the children of its parent at `res_parent`.
- **`child_pos_to_cell(parent: H3CellInput, res_child: int, child_pos: long) -> long`**: Child of a cell at a position.
- **`compact_cells(cells: array<H3CellInput>) -> array<long>`**: Compacts a set of cells.
- **`uncompact_cells(cells: array<H3CellInput>, res: int) -> array<long>`**: Uncompacts a set of cells to a resolution.
- **`h3shape_to_cells(shape: H3Shape, res: int) -> array<long>`**: Cells covering a shape. `H3Shape` is a GeoJSON string, an array of `(lat, lng)`, or an array of arrays of `(lat, lng)`.
- **`h3shape_to_cells_compacted(shape: H3Shape, res: int) -> array<long>`**: `h3shape_to_cells` followed by `compact_cells`, without an additional pass over the data.
- **`cells_to_h3shape(cells: array<H3CellInput>) -> string`**: GeoJSON shape of a set of cells.
- **`get_icosahedron_faces(cell: H3CellInput) -> array<int>`**: Icosahedron faces intersected by a cell.
- **`cell_to_local_ij(origin: H3CellInput, cell: H3CellInput) -> array<int>`**: Local IJ coordinates of a cell relative to an origin.
- **`local_ij_to_cell(origin: H3CellInput, i: int, j: int) -> long`**: Cell at local IJ coordinates relative to an origin.
- **`cell_area(cell: H3CellInput, unit: AreaUnit | str = AreaUnit.KM2) -> double`**: Area of a cell.

`min_child`, `max_child` and `h3shape_to_cells_compacted` can also be imported from `h3spark.convenience`.

### Working with hex string cells (`h3spark.hex`)

If your data stores cells as hex strings (e.g. `"8928308280fffff"`), `h3spark.hex` mirrors the API above but takes cells as strings and returns cells as lowercase strings. Arrays of cells become `array<string>`. Other arguments and non-cell results (resolutions, booleans, areas, coordinates, counts) are unchanged.

```python
import h3spark.hex as h3hex

df.select(
    h3hex.cell_to_parent("cell_str", 5),  # string
    h3hex.grid_disk("cell_str", 1),       # array<string>
    h3hex.get_resolution("cell_str"),     # int
)
```

Cells are converted with the native `str_to_int` / `int_to_str`, so natively implemented functions stay entirely in the JVM (no Python UDF). The conversions do no more validation than the native functions. If you chain several operations, it's cheaper to convert once with `str_to_int` and work with long cells.

## Breaking changes in 0.3.0

- Coordinates, areas and lengths are `DoubleType` instead of `FloatType`, which lost ~0.4m of precision. This affects `COORDINATE_TYPE`, `BOUNDARY_TYPE`, `cell_area`, `average_hexagon_area` and `average_hexagon_edge_length`.
- Resolutions and base cell numbers are always `IntegerType`. Previously `get_resolution` returned `short` from `h3spark` and `long` from `h3spark.native`, and `get_base_cell` returned `long`. `cell_to_children_size` and `cell_to_child_pos` now return `long` (child positions overflowed `int`).
- These functions are now native: `str_to_int`, `int_to_str`, `get_resolution`, `get_base_cell_number`, `is_pentagon`, `is_res_class_III`, `cell_to_parent`, `cell_to_center_child`, `cell_to_children_size`, `get_num_cells`, `average_hexagon_area`, `average_hexagon_edge_length`, `get_pentagons`, `get_res0_cells`, and `h3spark.convenience.min_child` / `max_child`. They require integral cell columns: a hex string column now raises an error naming the `h3spark.hex` equivalent. They also no longer raise on invalid cell values. **If you store cells as hex strings, use `h3spark.hex`.** For the previous h3-py backed behaviour, use `h3spark.udf`.
- `unit` arguments must be plain Python values, not Columns (e.g. `average_hexagon_area(7, "km^2")` instead of `F.lit("km^2")`).
- Public functions are plain Python functions returning Columns, not `UserDefinedFunction` objects, so they can't be passed to `spark.udf.register`.
- `h3spark.native` functions use the same parameter names as `h3spark` (e.g. `cell_to_parent(cell, res)` instead of `(col, parent_resolution)`). `convenience.min_child` / `max_child` take `cell` instead of `cell_id`.
- `cell_to_local_ij` takes `(origin, cell)` like h3-py (it previously always failed), and `get_icosahedron_faces` now works (it failed with h3 v4).
- Removed the deprecated `h3spark.native.cell_to_parent_fixed`, the unused `h3spark.models.Coordinates`, and the `h3spark.native` constants (`BASE_CELL_DATA`, ...).

## License

This library is released under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Contributing

Contributions are welcome! Please submit a pull request or open an issue to contribute to the project.

Every function is defined once in `src/h3spark/__init__.py` with the `@h3fn` decorator. The decorator declares the function's h3-py reference, its Spark return type, which parameters are cells (or arrays of cells, shapes, or plan-time values), and whether it returns cells:

```python
@h3fn(h3=h3.grid_disk, return_type=T.ArrayType(T.LongType()), cells=("cell",), returns="cells")
def grid_disk(cell: ColumnOrName, k: IntOrColumn) -> Column: ...
```

A body of `...` gives a Python UDF built from `h3=`. A real body is the native implementation and becomes the default; cell arguments arrive as long Columns and other arguments as Columns. `h3spark.native`, `h3spark.udf`, `h3spark.hex` and the tests in `test/test_registry.py` are all generated from these definitions. Adding a function, or reimplementing one natively, needs nothing else, unless it introduces a new parameter name, which needs an argument generator in the test.

```sh
uv sync
uv run pytest
uv run black --check . && uv run flake8 .
uv run python bench/bench.py --rows 1000000  # UDF vs native vs hex timings
```

Releases are published to PyPI by CI when a pull request labelled `release:patch`, `release:minor` or `release:major` is merged.

## Acknowledgments

This library is built on top of the H3 geospatial indexing library and PySpark. Special thanks to the developers of these libraries for their contributions to the open-source community.

For more information, check the [official H3 documentation](https://h3geo.org/docs/) and [PySpark documentation](https://spark.apache.org/docs/latest/api/python/index.html).
