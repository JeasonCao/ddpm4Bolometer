
# Repository entry bootstrap; no data or model is loaded here.
from pathlib import Path as _RepoPath
import sys as _repo_sys
_REPO_ROOT = _RepoPath(__file__).resolve().parents[2]
_repo_sys.path.insert(0, str(_REPO_ROOT))
_repo_sys.path.insert(0, str(_REPO_ROOT / "scripts"))
import sys, io, contextlib, json, time
P = str(_REPO_ROOT)
sys.path.insert(0, P)
import numpy as np, h5py
import src.pulse.simulator as S
sys.path.insert(0, P + '/scripts'); import generate_eval_dataset as G
with contextlib.redirect_stdout(io.StringIO()):
    params, eq = G._find_stable_params([1.0], seed=1)
ref = h5py.File('/mnt/d/storage/research/DDPM4bolometer/eval_data_OFcompare/7p5mV/eval_efficiency_1keV.h5')['clean_template'][:]
_, v1 = S.simulate_pulse(1.0, params, eq, duration=10.0, t_onset=1.5, f_sample=1000.0)
print('reproduce 1 keV template: max rel diff', np.abs(v1 - ref).max() / ref.max())
print({k: getattr(params, k) for k in ('R0','T0','lambda0','q','g_ec','a_ec','C_p','T_base')})
print('eq', eq)
orig_alpha = S._compute_alpha
def alpha_first_order(dte, dv, p, e):
    Te, Ve, g = e.T_e, e.V_bol, p.gamma
    eta = g * (p.T0 / Te) ** g
    CA = S.E_CHARGE * p.lambda0 / (S.K_B * p.W) * Ve / Te ** (g + 1)
    u = dte / Te; w = dv / Ve
    return (eta - (g + 1) * CA) * u + CA * w
def alpha_q0(dte, dv, p, e):
    q = p.q; p.q = 0.0
    try: return orig_alpha(dte, dv, p, e)
    finally: p.q = q
E = [0.01, 1, 10, 50, 239, 511, 1000, 1461, 2615, 3034, 4000, 5407]
out = {}
for name, fa in (('full', orig_alpha), ('q=0', alpha_q0), ('alpha first order only', alpha_first_order)):
    S._compute_alpha = fa; res = {}
    for e in E:
        t0 = time.time()
        _, v = S.simulate_pulse(e, params, eq, duration=10.0, t_onset=1.5, f_sample=1000.0)
        res[e] = v
    S._compute_alpha = orig_alpha
    lin = res[0.01] / 0.01           # small-signal response per keV
    rows = {}
    for e in E:
        v = res[e]
        rows[e] = dict(peak=v.max() / e / lin.max(), area=v[1500:].sum() / e / lin[1500:].sum(),
                       proj=float(v @ lin / (lin @ lin)) / e, tpeak_ms=int(np.argmax(v)) - 1500)
    out[name] = rows
    print(f'\n== {name}  (all relative to the 0.01 keV response scaled per keV)')
    print(f"{'E keV':>7} {'peak/E':>9} {'area/E':>9} {'proj/E':>9} {'tpeak':>6}")
    for e in E:
        r = rows[e]; print(f"{e:7g} {r['peak']:9.5f} {r['area']:9.5f} {r['proj']:9.5f} {r['tpeak_ms']:6d}")
json.dump({k: {str(e): r for e, r in v.items()} for k, v in out.items()}, open(sys.argv[1], 'w'), indent=1)  # usage: python scripts/check_template_linearity.py out.json
