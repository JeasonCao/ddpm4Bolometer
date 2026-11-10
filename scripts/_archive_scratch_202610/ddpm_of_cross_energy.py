# DDPM -> OF with AP/ANPS built from one energy and applied to the other (pilot runs, halves A -> B).
import sys; sys.path.insert(0,'.')
import numpy as np, h5py
from scripts.of_compare.plot_of_vs_ddpm_15keV import of_amplitudes
D='/mnt/d/storage/research/DDPM4bolometer'
bsub=lambda x: x-x[...,:1400].mean(-1,keepdims=True)
sh=np.arange(-10,10.01,0.5)
for run in (999996,999997):
    den={}
    for E in (15,2615):
        with h5py.File(f'{D}/nonoise_out/pilot_run{run}/eval_efficiency_{E}keV.h5') as f: den[E]=bsub(f['waveforms_denoised_nonoise'][:].astype(float))
    for src in (15,2615):
        A=den[src][:2500]; AP=A.mean(0); Dn=np.abs(np.fft.rfft(A-AP)).mean(0)
        out=[]
        for E in (15,2615):
            a,_=of_amplitudes(np.fft.rfft(den[E][2500:]),np.fft.rfft(AP),Dn,sh); out.append(f'{E} keV {a.std()/a.mean()*E:.4f}')
        # mixed ANPS: residuals of both energies, each about its own mean
        print(f'run {run}, AP+ANPS from {src} keV -> '+', '.join(out),flush=True)
    APm=den[2615][:2500].mean(0)
    Dm=np.abs(np.fft.rfft(np.r_[den[15][:2500]-den[15][:2500].mean(0),den[2615][:2500]-den[2615][:2500].mean(0)])).mean(0)
    out=[]
    for E in (15,2615):
        a,_=of_amplitudes(np.fft.rfft(den[E][2500:]),np.fft.rfft(APm),Dm,sh); out.append(f'{E} keV {a.std()/a.mean()*E:.4f}')
    print(f'run {run}, AP 2615 + ANPS mixed 15&2615 -> '+', '.join(out),flush=True)
