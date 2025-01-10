# h3spark

![Tile the world in hexes](images/big_geo.jpeg "Tile the world in hexes")

`h3spark` is a Python library that provides a set of user-defined functions (UDFs) for working with H3 geospatial indexing in PySpark. The functions in this library follow the same assumptions and rules as the native H3 functions, allowing for seamless integration and usage in PySpark data pipelines.

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

### Functions

- `str_to_int(h3_str)`: Converts an H3 string to a decimal integer.
- `int_to_str(h3_int)`: Converts an H3 integer to a string.
- `get_num_cells(res)`: Returns the number of H3 cells for a given resolution.
- `average_hexagon_area(res, unit)`: Returns the average area of H3 hexagons at a given resolution.
- `average_hexagon_edge_length(res, unit)`: Returns the average edge length of H3 hexagons at a given resolution.
- `latlng_to_cell_decimal(lat, lng, res)`: Converts latitude and longitude to an H3 cell in decimal format.
- `latlng_to_cell(lat, lng, res)`: Converts latitude and longitude to an H3 cell in string format.
- `cell_to_latlng(cell)`: Converts an H3 cell to latitude and longitude coordinates.
- `get_resolution(cell)`: Returns the resolution of an H3 cell.
- `cell_to_parent_decimal(cell, res)`: Converts an H3 cell to its parent cell at a given resolution in decimal format.
- `cell_to_parent(cell, res)`: Converts an H3 cell to its parent cell at a given resolution in string format.
- `grid_distance(cell1, cell2)`: Returns the distance between two H3 cells.
- `cell_to_boundary(cell)`: Returns the boundary of an H3 cell.
- `grid_disk_decimal(cell, k)`: Returns the H3 cells within k rings from the origin cell in decimal format.
- `grid_disk(cell, k)`: Returns the H3 cells within k rings from the origin cell in string format.
- `grid_ring_decimal(cell, k)`: Returns the H3 cells in the k-th ring from the origin cell in decimal format.
- `grid_ring(cell, k)`: Returns the H3 cells in the k-th ring from the origin cell in string format.
- `cell_to_children_size(cell, res)`: Returns the number of child cells at a given resolution for an H3 cell.
- `cell_to_children_decimal(cell, res)`: Returns the child cells of an H3 cell at a given resolution in decimal format.
- `cell_to_children(cell, res)`: Returns the child cells of an H3 cell at a given resolution in string format.
- `cell_to_child_pos(child, res_parent)`: Returns the position of a child cell relative to its parent.
- `child_pos_to_cell_decimal(parent, res_child, child_pos)`: Converts a parent cell, resolution, and child position to an H3 cell in decimal format.
- `child_pos_to_cell(parent, res_child, child_pos)`: Converts a parent cell, resolution, and child position to an H3 cell in string format.
- `compact_cells_decimal(cells)`: Compacts a list of H3 cells in decimal format.
- `compact_cells(cells)`: Compacts a list of H3 cells in string format.
- `uncompact_cells_decimal(cells, res)`: Uncompacts a list of H3 cells to a given resolution in decimal format.
- `uncompact_cells(cells, res)`: Uncompacts a list of H3 cells to a given resolution in string format.
- `h3shape_to_cells_decimal(shape, res)`: Converts an H3 shape to cells at a given resolution in decimal format.
- `h3shape_to_cells(shape, res)`: Converts an H3 shape to cells at a given resolution in string format.
- `cells_to_h3shape(cells)`: Converts a list of H3 cells to an H3 shape.
- `is_pentagon(cell)`: Checks if an H3 cell is a pentagon.
- `get_base_cell_number(cell)`: Returns the base cell number of an H3 cell.
- `are_neighbor_cells(cell1, cell2)`: Checks if two H3 cells are neighbors.
- `grid_path_cells_decimal(start, end)`: Returns the H3 cells forming a path between two cells in decimal format.
- `grid_path_cells(start, end)`: Returns the H3 cells forming a path between two cells in string format.
- `is_res_class_III(cell)`: Checks if an H3 cell is of class III resolution.
- `get_pentagons_decimal(res)`: Returns the pentagons at a given resolution in decimal format.
- `get_pentagons(res)`: Returns the pentagons at a given resolution in string format.
- `get_res0_cells_decimal()`: Returns the resolution 0 cells in decimal format.
- `get_res0_cells()`: Returns the resolution 0 cells in string format.
- `cell_to_center_child_decimal(cell, res)`: Returns the center child cell of an H3 cell at a given resolution in decimal format.
- `cell_to_center_child(cell, res)`: Returns the center child cell of an H3 cell at a given resolution in string format.
- `get_icosahedron_faces(cell)`: Returns the icosahedron faces of an H3 cell.
- `cell_to_local_ij(cell)`: Converts an H3 cell to local IJ coordinates.
- `local_ij_to_cell_decimal(origin, i, j)`: Converts local IJ coordinates to an H3 cell in decimal format.
- `local_ij_to_cell(origin, i, j)`: Converts local IJ coordinates to an H3 cell in string format.
- `cell_area(cell, unit)`: Returns the area of an H3 cell in a specified unit.

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