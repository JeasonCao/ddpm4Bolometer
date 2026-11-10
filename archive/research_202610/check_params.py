import sys, os, glob, io, contextlib
sys.path.insert(0, '.'); sys.path.insert(0, 'scripts')
import numpy as np, h5py
import generate_eval_dataset as g
from src.pulse.simulator import sample_params, find_equilibrium, is_valid_equilibrium
D = '/mnt/d/storage/research/DDPM4bolometer/eval_data_holdout/'
Es = [1,2,3,5,10,15,20,25,30,40,50,239,511,727,1461,2615,3034]
# which attempt is chosen for each energy (seed=1)
for E in Es:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        p, eq = g._find_stable_params([E], seed=1)
    att = buf.getvalue().split('after ')[1].split(' ')[0]
    tm = g._simulate_template(E, p, eq)
    diffs = []
    for sub in sorted(os.listdir(D)):
        f = os.path.join(D, sub, f'eval_efficiency_{E}keV.h5')
        if os.path.exists(f):
            with h5py.File(f) as h: ct = h['clean_template'][:]
            diffs.append(f"{sub}:{np.abs(ct-tm).max()/np.abs(ct).max():.1e}")
    print(E, 'attempt', att, 'R0=%.4g' % p.R0, ' '.join(diffs), flush=True)
