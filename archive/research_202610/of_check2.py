# Compare OF weightings on the same windows: 1/J (standard), 1/mean|X| (Octopus old), plus subsets
import sys, numpy as np, csv, h5py
sys.path.insert(0,'scripts/of_compare'); import verify_of_stream as V
run=int(sys.argv[1]); nsig=int(sys.argv[2]) if len(sys.argv)>2 else 1000
R=f'/mnt/d/storage/research/DDPM4bolometer/OFcompare_truth/run{run}'
raw=f'/mnt/d/storage/research/DDPM4bolometer/OFcompare_octopus_staging/run{run}/raw'
seg=list(csv.DictReader(open(f'{R}/truth/segments.csv')))
data=[]; bounds=[0]
for s in seg:
    _,_,_,d=V.read_bin(f'{raw}/{s["file"]}'); data.append(d); bounds.append(bounds[-1]+len(d))
def win(t):
    p=np.searchsorted(bounds,t,side='right')-1; o=t-1500-bounds[p]
    w=V.volts(data[p][o:o+10000],10.0); return w-w[:1400].mean()
ev=list(csv.DictReader(open(f'{R}/truth/events.csv'))); nz=list(csv.DictReader(open(f'{R}/truth/noise_windows.csv')))
Xn=np.array([np.fft.rfft(win(int(r['trigger_sample']))) for r in nz])
half=len(Xn)//2
with h5py.File(f'{R}/templates/templates.h5') as f: T={int(k[:-3]):f[k][:10000] for k in f}
t26=T[2615]-T[2615][:1400].mean(); S=np.fft.rfft(t26)
Xs={E:np.array([np.fft.rfft(win(int(r['onset_sample']))) for r in ev if int(r['energy_keV'])==E][:nsig]) for E in T}
def est(D, X):           # amplitude in units of the 2615 template
    H=np.conj(S)/D; return np.real(X@H)/np.real((H*S).sum())
for name, D in [('1/J  (standard OF)', (np.abs(Xn[:half])**2).mean(0)),
                ('1/|X| (Octopus old)', np.abs(Xn[:half]).mean(0))]:
    an=est(D, Xn[half:])            # noise PSD from first half, evaluated on the other half
    out=f'{name}: noise std {an.std():.3e} ({an.std()*2615:.4f} keV)'
    for E,X in Xs.items():
        a=est(D,X); out+=f' | {E} keV mean {a.mean()*2615:.3f} std {a.std()*2615:.4f} keV'
    print(out)
