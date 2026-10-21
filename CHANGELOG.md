# Changelog

## 0.4.0

- Supports PySpark 4 and Spark Connect. With pyspark 3.5, 0.3.0 rejected Spark Connect `Column` objects.
- New native `is_valid_cell`. `cell_to_child_pos` and `child_pos_to_cell` are now native (about 2-5x faster); `h3spark.udf` has the h3-py versions.
- Passing a non-integral cell column to a native function fails when the plan is analysed rather than when the first row is processed, so it also fails for empty DataFrames.
- Results are named like a function call (e.g. `grid_disk(cell, 1)`) instead of `row(cell, 1)` for UDFs or the whole expression for native functions.
- Out of range resolutions give null in native functions also when ANSI mode is enabled (the default in Spark 4), instead of raising an array index error.

## 0.3.0

Breaking changes:

- Coordinates, areas and lengths are `DoubleType` instead of `FloatType`, which lost ~0.4m of precision. This affects `COORDINATE_TYPE`, `BOUNDARY_TYPE`, `cell_area`, `average_hexagon_area` and `average_hexagon_edge_length`.
- Resolutions and base cell numbers are always `IntegerType`. Previously `get_resolution` returned `short` from `h3spark` and `long` from `h3spark.native`, and `get_base_cell` returned `long`. `cell_to_children_size` and `cell_to_child_pos` now return `long` (child positions overflowed `int`).
- These functions are now native: `str_to_int`, `int_to_str`, `get_resolution`, `get_base_cell_number`, `is_pentagon`, `is_res_class_III`, `cell_to_parent`, `cell_to_center_child`, `cell_to_children_size`, `get_num_cells`, `average_hexagon_area`, `average_hexagon_edge_length`, `get_pentagons`, `get_res0_cells`, and `h3spark.convenience.min_child` / `max_child`. They require integral cell columns: a hex string column now raises an error naming the `h3spark.hex` equivalent. They also no longer raise on invalid cell values. **If you store cells as hex strings, use `h3spark.hex`.** For the previous h3-py backed behaviour, use `h3spark.udf`.
- `unit` arguments must be plain Python values, not Columns (e.g. `average_hexagon_area(7, "km^2")` instead of `F.lit("km^2")`).
- Public functions are plain Python functions returning Columns, not `UserDefinedFunction` objects, so they can't be passed to `spark.udf.register`.
- `h3spark.native` functions use the same parameter names as `h3spark` (e.g. `cell_to_parent(cell, res)` instead of `(col, parent_resolution)`). `convenience.min_child` / `max_child` take `cell` instead of `cell_id`.
- `cell_to_local_ij` takes `(origin, cell)` like h3-py (it previously always failed), and `get_icosahedron_faces` now works (it failed with h3 v4).
- Removed the deprecated `h3spark.native.cell_to_parent_fixed`, the unused `h3spark.models.Coordinates`, and the `h3spark.native` constants (`BASE_CELL_DATA`, ...).

For older releases, see the [GitHub releases](https://github.com/JosephChotard/h3spark/releases).
