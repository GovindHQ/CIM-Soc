"""Minimal timm stand-in: only the symbols TinyViT's source file imports.
Avoids pulling the full timm dependency chain. Model construction and weight
loading are done directly in run_tinyvit.py, so the builder is never called."""
__version__ = "0.9.0"
from . import models  # noqa: F401
