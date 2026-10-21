"""
Native implementations: pure Spark SQL expressions (bit operations and constant lookup tables), generated from the
functions in ``h3spark`` that have a native body.

These avoid Python UDF overhead but do far less validation than h3-py: for valid cells and in-range resolutions they
return the same values as h3-py, while invalid input produces null or meaningless values instead of raising. Cells
must be integral columns; other types (e.g. hex strings) raise an error: use ``str_to_int`` or ``h3spark.hex``.
"""

from . import REGISTRY

__all__ = sorted(name for name, entry in REGISTRY.items() if entry.native)
globals().update({name: REGISTRY[name].native for name in __all__})

# Names from h3spark < 0.3
ALIASES = {
    "get_base_cell": "get_base_cell_number",
    "maxchild": "max_child",
    "minchild": "min_child",
}
globals().update({alias: REGISTRY[name].native for alias, name in ALIASES.items()})
__all__ += list(ALIASES)
