# h3spark: H3 for PySpark

[![PyPI](https://img.shields.io/pypi/v/h3spark)](https://pypi.org/project/h3spark/)
[![Conda](https://img.shields.io/conda/vn/conda-forge/h3spark)](https://anaconda.org/conda-forge/h3spark)
[![Python](https://img.shields.io/pypi/pyversions/h3spark)](https://pypi.org/project/h3spark/)
[![Tests](https://github.com/JosephChotard/h3spark/actions/workflows/test.yml/badge.svg)](https://github.com/JosephChotard/h3spark/actions/workflows/test.yml)
[![License](https://img.shields.io/pypi/l/h3spark)](https://github.com/JosephChotard/h3spark/blob/main/LICENSE)

![Tile the world in hexes](https://raw.githubusercontent.com/JosephChotard/h3spark/main/images/big_geo.jpeg "Tile the world in hexes")

**h3spark** brings [Uber's H3](https://h3geo.org) hexagonal hierarchical geospatial index to PySpark and Spark SQL, with the current h3 v4 API. Index points to H3 cells, roll cells up to parents, find neighbours, fill polygons and join datasets on H3 cells, at Spark scale.

Most cell operations compile to native Spark SQL expressions that run in the JVM with no Python UDF, so they're several times faster than calling h3-py from a UDF. The rest wrap [h3-py](https://uber.github.io/h3-py/) 4.x.

## Why h3spark

- **h3 v4 API**: the same function names and behaviour as [h3-py 4.x](https://uber.github.io/h3-py/api_quick.html) (`latlng_to_cell`, `cell_to_parent`, `grid_disk`, `h3shape_to_cells`, ...).
- **Fast**: native Spark expressions for cell inspection and hierarchy functions (`cell_to_parent`, `get_resolution`, `is_valid_cell`, `cell_to_children_size`, ...), and Arrow-optimized UDFs for the rest. See [performance](#performance).
- **Long or string cells**: cells are `bigint` by default; `h3spark.hex` takes and returns hex string cells (`"89283082e73ffff"`) so you don't have to change your schemas.
- **Spark 3.5 and 4, Spark Connect and Databricks**: works with classic sessions and Spark Connect, including Databricks shared and serverless compute.
- **Typed**: ships type stubs, and passing the wrong cell type fails when the plan is built rather than halfway through a job.

## Installation

```bash
pip install h3spark
```

or

```bash
conda install -c conda-forge h3spark
```

Requires PySpark 3.5 or 4.x, classic or [Spark Connect](https://spark.apache.org/docs/latest/spark-connect-overview.html). UDFs returning scalars or structs run with Arrow optimization enabled (`useArrow=True`), so `pandas` and `pyarrow` are installed via `pyspark[sql]`.

## Quickstart

```python
from pyspark.sql import SparkSession, functions as F
import h3spark

spark = SparkSession.builder.getOrCreate()
df = spark.createDataFrame([(37.769377, -122.388903)], ["lat", "lng"])

df = (
    df.withColumn("cell", h3spark.latlng_to_cell("lat", "lng", 9))  # point to H3 cell at resolution 9
    .withColumn("parent", h3spark.cell_to_parent("cell", 5))        # native Spark expression, no UDF
    .withColumn("neighbors", F.size(h3spark.grid_disk("cell", 1)))  # the cell and its neighbours
    .withColumn("cell_str", h3spark.int_to_str("cell"))             # hex string representation
)
df.show()
```

```
+---------+-----------+------------------+------------------+---------+---------------+
|      lat|        lng|              cell|            parent|neighbors|       cell_str|
+---------+-----------+------------------+------------------+---------+---------------+
|37.769377|-122.388903|617700170065510399|599685771850416127|        7|89283082e73ffff|
+---------+-----------+------------------+------------------+---------+---------------+
```

With cells stored as hex strings, use `h3spark.hex`:

```python
import h3spark.hex as h3hex

df = spark.createDataFrame([("89283082e73ffff",)], ["cell"])
df.select(h3hex.cell_to_parent("cell", 5), h3hex.get_resolution("cell")).show()
```

```
+-----------------------+--------------------+
|cell_to_parent(cell, 5)|get_resolution(cell)|
+-----------------------+--------------------+
|        85283083fffffff|                   9|
+-----------------------+--------------------+
```

### Spatial joins on H3 cells

Index both sides at the same resolution and join on the cell. H3 joins are approximate: pairs in the same cell are candidates, to be confirmed with an exact geometry check if you need one.

```python
points = points.withColumn("cell", h3spark.latlng_to_cell("lat", "lng", 9))
areas = areas.withColumn("cell", F.explode(h3spark.h3shape_to_cells("geojson", 9)))
candidates = points.join(areas, "cell")
```

## Performance

1M rows on a laptop (`local[8]`), best of 3, in seconds. h3-pyspark stores cells as strings, so compare it with `h3spark.hex`.

| Function | h3-pyspark 1.2.6 (h3 v3) | h3spark.hex (string cells) | h3spark (long cells) |
|---|---|---|---|
| `latlng_to_cell` | 2.30 | 1.83 | 1.60 |
| `cell_to_latlng` | 2.45 | 1.85 | 1.93 |
| `cell_to_parent` | 1.69 | 0.52 | 0.12 |
| `get_resolution` | 1.30 | 0.29 | 0.17 |
| `is_valid_cell` | 1.36 | 0.55 | 0.27 |
| `grid_disk` (k=1) | 5.21 | 3.94 | 3.21 |

Run `uv run python bench/bench.py` to compare h3spark's native, UDF and hex versions on your machine.

## Usage

Every function comes in several flavours:

- **`h3spark.<fn>`**: the default. Natively implemented functions (see below) are pure Spark SQL expressions, which avoid the serialization overhead of Python UDFs. Everything else is a Python UDF calling h3-py.
- **`h3spark.native.<fn>`**: only the natively implemented functions. The 0.2 names `minchild`, `maxchild` and `get_base_cell` still work.
- **`h3spark.udf.<fn>`**: always the Python UDF backed by h3-py, for every function. Use it when you want h3-py's validation, or a reference to compare against.
- **`h3spark.hex.<fn>`**: the default, but taking and returning cells as hex strings (see below).

Arguments:

- Column arguments accept a `Column` or a column name (`str`). Other arguments, such as `res` or `k`, also accept plain Python values (`res=9`, no `F.lit` needed).
- `unit` arguments (`AreaUnit`/`LengthUnit` or their string values) are plain Python values resolved when the plan is built, not columns.
- Cells are 64-bit integers (`LongType`). Native functions accept integral cell columns (`bigint`, `int`, ...). Any other type, such as hex strings, fails when Spark analyses the plan (e.g. on `df.select`), before any data is processed, with an error pointing to `h3spark.hex`. Convert hex strings with `str_to_int` or use `h3spark.hex`. Python UDFs still accept either long or hex string cells (`H3CellInput = Union[str, int]`).
- Results are named like a function call, e.g. `cell_to_parent(cell, 5)`, whichever implementation runs.

Types: resolutions and base cell numbers are `int`, cells and counts are `long` (`bigint`), and areas, lengths and coordinates are `double`. `COORDINATE_TYPE` is `struct<lat: double, lon: double>` and `BOUNDARY_TYPE` is an array of it.

### Natively implemented functions

These follow the same rules as h3-py and return the same values for valid cells and in-range resolutions, but do far less validation. Invalid cells or resolutions (e.g. a parent resolution finer than the cell's) produce null or meaningless values instead of raising. Filter untrusted data with `is_valid_cell` first, or use `h3spark.udf.<fn>` if you need h3-py's errors.

- **`is_valid_cell(cell: long) -> bool`**: Whether a cell is valid, with the same checks as h3-py.
- **`str_to_int(h3_str: string) -> long`**: Converts a hex string cell (any case) to its integer representation. Strings that aren't hex give meaningless values or null.
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
- **`cell_to_child_pos(child: long, res_parent: int) -> long`**: Position of a cell among the children of its parent at `res_parent`.
- **`child_pos_to_cell(parent: long, res_child: int, child_pos: long) -> long`**: Child of a cell at a position. Null if the position or resolution is out of range.
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

## Coming from h3-pyspark or h3 v3

[h3-pyspark](https://github.com/kevinschaich/h3-pyspark) uses the h3 v3 API, with cells as hex strings. h3spark uses h3 v4 names; with string cells, use `h3spark.hex`, which takes and returns hex strings like h3-pyspark. Plain Python values work for non-column arguments, so `F.lit(9)` can be just `9`.

| h3-pyspark / h3 v3 | h3spark (h3 v4) |
|---|---|
| `geo_to_h3` | `latlng_to_cell` |
| `h3_to_geo` | `cell_to_latlng` (returns `struct<lat, lon>`) |
| `h3_to_geo_boundary` | `cell_to_boundary` |
| `h3_to_parent` | `cell_to_parent` |
| `h3_to_children` | `cell_to_children` |
| `h3_to_center_child` | `cell_to_center_child` |
| `h3_get_resolution` | `get_resolution` |
| `h3_get_base_cell` | `get_base_cell_number` |
| `h3_is_valid` | `is_valid_cell` |
| `h3_is_pentagon` | `is_pentagon` |
| `h3_is_res_class_III` | `is_res_class_III` |
| `h3_get_faces` | `get_icosahedron_faces` |
| `k_ring` | `grid_disk` |
| `hex_ring` | `grid_ring` |
| `h3_distance` | `grid_distance` |
| `h3_line` | `grid_path_cells` |
| `h3_indexes_are_neighbors` | `are_neighbor_cells` |
| `polyfill` | `h3shape_to_cells` |
| `h3_set_to_multi_polygon` | `cells_to_h3shape` |
| `compact` / `uncompact` | `compact_cells` / `uncompact_cells` |
| `cell_area` | `cell_area` |
| `hex_area` | `average_hexagon_area` |
| `edge_length` | `average_hexagon_edge_length` |
| `num_hexagons` | `get_num_cells` |
| `get_res0_indexes` | `get_res0_cells` |
| `get_pentagon_indexes` | `get_pentagons` |
| `experimental_h3_to_local_ij` | `cell_to_local_ij` |
| `experimental_local_ij_to_h3` | `local_ij_to_cell` |
| `string_to_h3` / `h3_to_string` | `str_to_int` / `int_to_str` |

h3spark doesn't support directed edges and vertices yet.

## Changelog

See [CHANGELOG.md](https://github.com/JosephChotard/h3spark/blob/main/CHANGELOG.md), including the breaking changes in 0.3.0.

## Contributing

Contributions are welcome: open an issue or a pull request. See [CONTRIBUTING.md](https://github.com/JosephChotard/h3spark/blob/main/CONTRIBUTING.md) for how functions are defined, tested and released.

## License

MIT. See [LICENSE](https://github.com/JosephChotard/h3spark/blob/main/LICENSE).

## Acknowledgments

Built on [H3](https://h3geo.org/docs/), [h3-py](https://uber.github.io/h3-py/) and [PySpark](https://spark.apache.org/docs/latest/api/python/index.html). Thanks to their developers.
