"""
Compatibility package for importing Backend/app as app from the repository root.
"""

from pathlib import Path

_backend_app = Path(__file__).resolve().parents[1] / "Backend" / "app"
__path__ = [str(_backend_app)]
