# DDPM -> OF with AP = mean DDPM output of half A and ANPS from the fluctuations around it (no truth used).
import sys; sys.path.insert(0,'.')
import numpy as np, h5py
from scripts.of_compare.plot_of_vs_ddpm_15keV import of_amplitudes
D='/mnt/d/storage/research/DDPM4bolometer'
bsub=lambda x: x-x[...,:1400].mean(-1,keepdims=True)
sets=[(f'pilot{r}',f'{D}/nonoise_out/pilot_run{r}','waveforms_denoised_nonoise') for r in (999996,999997)]
sets+=[('old1p382mV',f'{D}/nonoise_out/efficiency_1p382mV','waveforms_denoised_nonoise'),('old7p5mV',f'{D}/zeroinit_out/efficiency_7p5mV','waveforms_denoised_zeroinit')]
for tag,dd,field in sets:
    for E in (15,2615):
        with h5py.File(f'{dd}/eval_efficiency_{E}keV.h5') as f: den=bsub(f[field][:].astype(float))
        A,B=slice(0,2500),slice(2500,5000); sh=np.arange(-10,10.01,0.5)
        AP=den[A].mean(0); S=np.fft.rfft(AP)
        R=np.fft.rfft(den[A]-AP); XB=np.fft.rfft(den[B])
        a,_=of_amplitudes(XB,S,np.abs(R).mean(0),sh)
        print(tag,E,f'ddpm+OF(meanAP,|N|) {a.std()/a.mean()*E:.4f}',flush=True)
