"""M2Stitch package for tile registration and stage coordinate updates."""
try:
    from importlib.metadata import version
except ImportError:  # pragma: no cover
    from importlib_metadata import version  # type: ignore

from .tile_registration import register_and_update_coordinates
from .tile_registration import register_tiles

__all__ = ["register_and_update_coordinates", "register_tiles"]

__version__ = version(__name__)
