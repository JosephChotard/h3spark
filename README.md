# h3spark

![Tile the world in hexes](images/big_geo.jpeg "Tile the world in hexes")

`h3spark` is a Python library that provides a set of native and user-defined functions (UDFs) for working with H3 geospatial indexing in PySpark. The functions in this library follow the same assumptions and rules as the native H3 functions, allowing for seamless integration and usage in PySpark data pipelines.

It also provides native implementations of some H3 functions that are more performant in PySpark than using UDFs. These functions are reimplemented in PySpark and avoid the serialization/deserialization overhead of a UDF.

## Installation

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

Below is a brief overview of the available functions in `h3spark`. These functions are designed to work with PySpark DataFrames and provide H3 functionality within a distributed data processing environment.

Some of the functions have been reimplemented in pyspark for performance reasons. These functions can be imported from `h3spark.native`. The rest of the functions are wrappers around the native H3 functions and can be imported from `h3spark`. Note that the native functions strive to be as close to the original H3 functions as possible, but there may be some differences in behavior around edge cases + less validation.

### Functions

`H3CellInput` is a type alias that represents an H3 cell, which can be either a hexadecimal string or a long integer (`H3CellInput = Union[str, int]`). h3spark will handle conversion from between types if required by h3. Prefer long integers if possible for more efficient processing.

- **`str_to_int(h3_str: string) -> long`**: Converts an H3 string to an integer. _Has a pyspark native equivalent_
- **`int_to_str(h3_int: Union[str, int]) -> string`**: Converts an H3 integer to a string. Allows strings due to Spark's limitation with unsigned 64-bit integers. _Has a pyspark native equivalent_
- **`get_num_cells(res: int) -> int`**: Returns the number of H3 cells at a given resolution.
- **`average_hexagon_area(res: int, unit: Union[AreaUnit, str] = AreaUnit.KM2) -> float`**: Calculates the average area of an H3 hexagon at a given resolution and unit.
- **`average_hexagon_edge_length(res: int, unit: Union[LengthUnit, str] = LengthUnit.KM) -> float`**: Computes the average edge length of an H3 hexagon at a specified resolution and unit.
- **`latlng_to_cell(lat: float, lng: float, res: int) -> long`**: Converts latitude and longitude to an H3 cell at a specified resolution.
- **`cell_to_latlng(cell: H3CellInput) -> COORDINATE_TYPE`**: Converts an H3 cell to its central latitude and longitude.
- **`get_resolution(cell: H3CellInput) -> short`**: Retrieves the resolution of a given H3 cell. _Has a pyspark native equivalent_
- **`cell_to_parent(cell: H3CellInput, res: int) -> long`**: Converts an H3 cell to its parent cell at a specified resolution.
- **`grid_distance(cell1: H3CellInput, cell2: H3CellInput) -> int`**: Calculates the distance in grid cells between two H3 cells. _Has a pyspark native equivalent if the cell's resolution and parent are literals_
- **`cell_to_boundary(cell: H3CellInput) -> BOUNDARY_TYPE`**: Returns the boundary of an H3 cell as a list of coordinates.
- **`grid_disk(cell: H3CellInput, k: int) -> List[long]`**: Returns all cells within k rings around the given H3 cell.
- **`grid_ring(cell: H3CellInput, k: int) -> List[long]`**: Returns cells in a ring of k distance from the given H3 cell.
- **`cell_to_children_size(cell: H3CellInput, res: int) -> int`**: Returns the number of children cells for a given cell at a specified resolution. _Has a pyspark native equivalent_
- **`cell_to_children(cell: H3CellInput, res: int) -> List[long]`**: Returns the children of an H3 cell at a specified resolution.
- **`cell_to_child_pos(child: H3CellInput, res_parent: int) -> int`**: Finds the position of a child cell relative to its parent cell at a specified resolution.
- **`child_pos_to_cell(parent: H3CellInput, res_child: int, child_pos: int) -> long`**: Converts a child position back to an H3 cell.
- **`compact_cells(cells: List[H3CellInput]) -> List[long]`**: Compacts a list of H3 cells.
- **`uncompact_cells(cells: List[H3CellInput], res: int) -> List[long]`**: Uncompacts a list of H3 cells to a specified resolution.
- **`h3shape_to_cells(shape: H3Shape, res: int) -> List[long]`**: Converts a shape to H3 cells at a specified resolution.
- **`cells_to_h3shape(cells: List[H3CellInput]) -> string`**: Converts a list of H3 cells to a GeoJSON shape.
- **`is_pentagon(cell: H3CellInput) -> bool`**: Checks if an H3 cell is a pentagon. _Has a pyspark native equivalent_
- **`get_base_cell_number(cell: H3CellInput) -> int`**: Retrieves the base cell number of an H3 cell. _Has a pyspark native equivalent_
- **`are_neighbor_cells(cell1: H3CellInput, cell2: H3CellInput) -> bool`**: Checks if two H3 cells are neighbors.
- **`grid_path_cells(start: H3CellInput, end: H3CellInput) -> List[long]`**: Finds the grid path between two H3 cells.
- **`is_res_class_III(cell: H3CellInput) -> bool`**: Checks if an H3 cell is of class III resolution.
- **`get_pentagons(res: int) -> List[long]`**: Returns all pentagon cells at a given resolution.
- **`get_res0_cells() -> List[long]`**: Returns all resolution 0 base cells.
- **`cell_to_center_child(cell: H3CellInput, res: int) -> long`**: Finds the center child cell of a given cell at a specified resolution.
- **`get_icosahedron_faces(cell: H3CellInput) -> List[int]`**: Retrieves icosahedron face indexes for a given H3 cell.
- **`cell_to_local_ij(cell: H3CellInput) -> List[int]`**: Converts an H3 cell to local IJ coordinates.
- **`local_ij_to_cell(origin: H3CellInput, i: int, j: int) -> long`**: Converts local IJ coordinates back to an H3 cell.
- **`cell_area(cell: H3CellInput, unit: Union[AreaUnit, str] = AreaUnit.KM2) -> float`**: Computes the area of an H3 cell in a specified unit.


### Spark native Functions

Some H3 functions can ~mostly be reimplemented purely within pyspark. Doing so avoids the serialization/deserialization overhead of a UDF. These functions should be mostly equivalent to their C native counterparts while being more performant in pyspark. You can import them from `h3spark.native`

- **`cell_to_children_size(cell: long, res: int, validate_resolution: Optional[bool] = False) -> int`**: Returns the number of children cells for a given cell at a specified resolution. If `validate_resolution` is set to True, it will throw an error if the resolution of the input cell is less than the requested child resolution.
- **`cell_to_parent_fixed(cell: long, current_resolution: int, parent_resolution: int) -> long`**: Given a column where every row has the same resolution (current_resolution), call `cell_to_parent` on every row to the same constant resolution (parent_resolution). Does not perform any validation on the input cells
- **`get_base_cell(cell: long) -> long`**: Retrieves the base cell number of an H3 cell.
- **`get_resolution(cell: long) -> long`**: Retrieves the resolution of a given H3 cell.
- **`int_to_str(cell: long) -> string`**: Converts an H3 integer to its string representation
- **`is_pentagon(cell: long) -> bool`**: Checks if an H3 cell is a pentagon.
- **`str_to_int(cell: string) -> long`**: Converts an H3 string to its integer representation

### Convenience functions

We provide some functions that wrap other h3 functions for streamlining commonly used operations. You can import them from `h3spark.convenience`

- **`min_child(cell: H3CellInput, resolution: int) -> long`**: Finds the child of minimum value of the input H3 cell at the specified resolution
- **`max_child(cell: H3CellInput, resolution: int) -> long`**: Finds the child of maximum value of the input H3 cell at the specified resolution

## License

This library is released under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Contributing

Contributions are welcome! Please submit a pull request or open an issue to contribute to the project.

## Acknowledgments

This library is built on top of the H3 geospatial indexing library and PySpark. Special thanks to the developers of these libraries for their contributions to the open-source community.

For more information, check the [official H3 documentation](https://h3geo.org/docs/) and [PySpark documentation](https://spark.apache.org/docs/latest/api/python/index.html).

## Building + Deploying

```sh
python -m build
python -m twine upload --verbose --repository pypi dist/*
```