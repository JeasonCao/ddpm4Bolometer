# Better short-segment ANPS for the raw pilot noise: Hann window, power average, log interpolation;
# and an overlapping-segment variant. Compare with 10 s.
import sys; sys.path.insert(0,'.')
import numpy as np, h5py
from scripts.of_compare.plot_of_vs_ddpm_15keV import of_amplitudes
D='/mnt/d/storage/research/DDPM4bolometer'
bsub=lambda x: x-x[...,:1400].mean(-1,keepdims=True)
sh=np.arange(-10,10.01,0.5); F=np.fft.rfftfreq(10000)
def anps(segs, win):
    n=segs.shape[1]; w=np.hanning(n) if win else np.ones(n)
    segs=(segs-segs.mean(1,keepdims=True))*w
    P=(np.abs(np.fft.rfft(segs))**2).mean(0)/np.mean(w**2); f=np.fft.rfftfreq(n)
    return np.sqrt(np.exp(np.interp(F,f[1:],np.log(P[1:]))))
for run in (999996,999997):
    E=15
    with h5py.File(f'{D}/nonoise_out/pilot_run{run}/eval_efficiency_{E}keV.h5') as f:
        noisy=bsub(f['waveforms_noisy'][:].astype(float)); tm=bsub(f['clean_template'][:].astype(float))
    N=noisy-tm; X=np.fft.rfft(noisy[2500:]); S=np.fft.rfft(tm)
    r=lambda Dn: (lambda a: a.std()/a.mean()*E)(of_amplitudes(X,S,Dn,sh)[0])
    out=[('10s',r(np.abs(np.fft.rfft(N[:2500])).mean(0)))]
    for n in (5000,2000,1000):
        segs=N[:2500].reshape(-1,n)
        out+= [(f'{n/1000:g}s rect',r(anps(segs,False))),(f'{n/1000:g}s hann',r(anps(segs,True)))]
    print(run,' | '.join(f'{k} {v:.4f}' for k,v in out))
