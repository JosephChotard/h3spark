"""
Native implementations: pure Spark SQL expressions (bit operations and constant lookup tables), generated from the
functions in ``h3spark`` that have a native body.

These avoid Python UDF overhead but do far less validation than h3-py: for valid cells and in-range resolutions they
return the same values as h3-py, while invalid input produces null or meaningless values instead of raising. Cells
must be ``LongType`` (other types are cast to long, so hex strings become null: use ``str_to_int`` or ``h3spark.hex``).
"""

from . import REGISTRY

__all__ = sorted(name for name, entry in REGISTRY.items() if entry.native)
globals().update({name: REGISTRY[name].native for name in __all__})

# Names from h3spark < 0.3
minchild = REGISTRY["min_child"].native
maxchild = REGISTRY["max_child"].native
get_base_cell = REGISTRY["get_base_cell_number"].native
__all__ += ["get_base_cell", "maxchild", "minchild"]
