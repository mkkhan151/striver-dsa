"""
Connect Four UI entry point.

Run from the repo root:
    python LLD/connect_four/main_ui.py
"""

import os
import sys
import types

# Add repo root to sys.path so we can import LLD.connect_four submodules
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

# Register the package in sys.modules without executing __init__.py's game loop
for _pkg_name, _pkg_path in [
    ("LLD", os.path.join(_repo_root, "LLD")),
    ("LLD.connect_four", os.path.join(_repo_root, "LLD", "connect_four")),
]:
    if _pkg_name not in sys.modules:
        _pkg = types.ModuleType(_pkg_name)
        _pkg.__path__ = [_pkg_path]
        _pkg.__package__ = _pkg_name
        sys.modules[_pkg_name] = _pkg

import customtkinter  # noqa: E402

from LLD.connect_four.ui import UIManager  # noqa: E402

if __name__ == "__main__":
    customtkinter.set_appearance_mode("dark")
    customtkinter.set_default_color_theme("dark-blue")
    app = UIManager()
    app.run()
