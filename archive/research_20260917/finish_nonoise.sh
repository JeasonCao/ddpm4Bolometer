#!/bin/bash
# Downstream chain for the full-statistics no_noise run.
# The source eval HDF5s are opened READ-ONLY throughout: instead of appending a
# field to a 6 GB file, a self-contained fit input is assembled in a staging
# directory from the datasets run_fitting.py actually reads.
set -e
SP=/tmp/claude-1000/-home-wsl-0vbb/4159fbbb-99dd-4ed5-b7cf-493a0436b6d5/scratchpad
cd "/mnt/d/坚果云存储/复旦事宜/科研/CUPID/paperwriting/Denoising/ddpm4Bolometer#"
PY=~/mambaforge3/envs/DDPM/bin/python3

echo "[$(date +%H:%M)] assembling fit inputs"
$PY -u - <<'PYEOF'
import os, h5py, numpy as np
EVAL='/mnt/d/storage/research/DDPM4bolometer/eval_data_holdout_denoised/efficiency_1p382mV'
NN  ='/mnt/d/storage/research/DDPM4bolometer/nonoise_out/efficiency_1p382mV'
STG ='/mnt/d/storage/research/DDPM4bolometer/nonoise_fitinput/efficiency_1p382mV'
NEED=['waveforms_noisy','snr_db','noise_rms','clean_template']
os.makedirs(STG, exist_ok=True)
for E in (15, 2615):
    fn=f'eval_efficiency_{E}keV.h5'
    dst=os.path.join(STG,fn)
    if os.path.exists(dst): print(f'  {E} keV staged already'); continue
    with h5py.File(os.path.join(NN,fn),'r') as n:
        den=n['waveforms_denoised_nonoise'][:]
    with h5py.File(os.path.join(EVAL,fn),'r') as s, h5py.File(dst+'.part','w') as d:
        assert abs(float(s.attrs['energy_kev'])-E)<1e-6
        assert den.shape[0]==s['waveforms_noisy'].shape[0], 'n_events mismatch'
        for k in NEED: d.create_dataset(k, data=s[k][:])
        g=d.create_group('noise_params')
        for k in s['noise_params']: g.create_dataset(k, data=s['noise_params'][k][:])
        d.create_dataset('waveforms_denoised_nonoise', data=den)
        for k in s.attrs: d.attrs[k]=s.attrs[k]
    os.replace(dst+'.part',dst)
    print(f'  {E} keV staged  {den.shape} {den.dtype}')
PYEOF

echo "[$(date +%H:%M)] fitting"
$PY -u scripts/run_fitting.py \
    --input_dir /mnt/d/storage/research/DDPM4bolometer/nonoise_fitinput/efficiency_1p382mV \
    --denoised_field waveforms_denoised_nonoise \
    --output_dir results/fit_results/1p382mV_nonoise \
    --workers 11 2>&1 | grep -E "Fitting 5000|Fitting done|Convergence"

echo "[$(date +%H:%M)] comparison"
$PY -u - <<'PYEOF'
import sys; sys.path.insert(0,'.')
import h5py, numpy as np
from scripts.run_gauss_fit import fit_gaussian
K=782.06
print(f'\n{"E[keV]":>8} {"method":>12} {"FWHM[keV]":>11} {"conv%":>7} {"vs noisy":>9}')
print('-'*54)
for E in (15,2615):
    rows=[]
    with h5py.File(f'results/fit_results/1p382mV_nonoise/fit_efficiency_{E}keV.h5') as f:
        n=f['amp_noisy'][:].astype(float); d=f['amp_denoised'][:].astype(float)
        cn=f['converged_noisy'][:].mean()*100; cd=f['converged_denoised'][:].mean()*100
    fn=2.355*fit_gaussian(n)['sigma']*K
    rows.append(('noisy',fn,cn))
    rows.append(('no_noise',2.355*fit_gaussian(d)['sigma']*K,cd))
    with h5py.File(f'results/fit_results/1p382mV_zeroinit/fit_efficiency_{E}keV.h5') as f:
        z=f['amp_denoised'][:].astype(float); cz=f['converged_denoised'][:].mean()*100
    rows.append(('zeroinit',2.355*fit_gaussian(z)['sigma']*K,cz))
    for name,v,c in rows:
        print(f'{E:8d} {name:>12} {v:11.4f} {c:7.1f} {v/fn:9.3f}')
PYEOF
echo "ALL_NONOISE_DONE"
