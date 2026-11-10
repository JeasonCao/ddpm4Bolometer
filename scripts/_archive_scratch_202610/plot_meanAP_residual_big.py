"""Compatibility entry. Canonical implementation: archive/research_202610/plot_meanAP_residual_big.py."""
from pathlib import Path as _CompatPath
import runpy as _compat_runpy
import sys as _compat_sys
_compat_root = _CompatPath(__file__).resolve().parents[2]
_compat_target = _compat_root / 'archive/research_202610/plot_meanAP_residual_big.py'
_compat_sys.path.insert(0, str(_compat_root))
_compat_sys.path.insert(0, str(_compat_target.parent))
globals().update(_compat_runpy.run_path(str(_compat_target), run_name=__name__))
