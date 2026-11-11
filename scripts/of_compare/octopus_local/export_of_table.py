"""Compatibility entry. Canonical implementation: experiments/of_comparison/octopus_local/export_of_table.py."""
from pathlib import Path as _CompatPath
import runpy as _compat_runpy
import sys as _compat_sys
_compat_root = _CompatPath(__file__).resolve().parents[3]
_compat_target = _compat_root / 'experiments/of_comparison/octopus_local/export_of_table.py'
_compat_sys.path.insert(0, str(_compat_root))
_compat_sys.path.insert(0, str(_compat_target.parent))
globals().update(_compat_runpy.run_path(str(_compat_target), run_name=__name__))
