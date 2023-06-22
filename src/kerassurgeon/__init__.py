from importlib_metadata import version

from kerassurgeon.surgeon import Surgeon  # noqa: F401

try:
    __version__ = version(__name__)
except ImportError:
    pass
