# Independent numpy OF on run 999996: true templates, PSD from quiet windows, fixed delay (onset known)
import sys, numpy as np, csv, h5py
sys.path.insert(0,'scripts/of_compare'); import verify_of_stream as V
run=int(sys.argv[1])
R=f'/mnt/d/storage/research/DDPM4bolometer/OFcompare_truth/run{run}'
raw=f'/mnt/d/storage/research/DDPM4bolometer/OFcompare_octopus_staging/run{run}/raw'
seg=list(csv.DictReader(open(f'{R}/truth/segments.csv')))
data=[]; bounds=[0]
for s in seg:
    _,_,_,d=V.read_bin(f'{raw}/{s["file"]}'); data.append(d); bounds.append(bounds[-1]+len(d))
def win(t):
    p=np.searchsorted(bounds,t,side='right')-1; o=t-1500-bounds[p]
    return V.volts(data[p][o:o+10000],10.0)
ev=list(csv.DictReader(open(f'{R}/truth/events.csv'))); nz=list(csv.DictReader(open(f'{R}/truth/noise_windows.csv')))
N=10000
X=np.array([np.fft.rfft(w-w[:1400].mean()) for w in (win(int(r['trigger_sample'])) for r in nz)])
J=(np.abs(X)**2).mean(0)
with h5py.File(f'{R}/templates/templates.h5') as f: T={int(k[:-3]):f[k][:N] for k in f}
S=np.fft.rfft(T[2615]-T[2615][:1400].mean())
H=np.conj(S)/J; norm=(np.abs(S)**2/J).sum()
amp=lambda w: np.real((H*np.fft.rfft(w-w[:1400].mean())).sum())/norm
print('theory sigma (amplitude units of 2615 template): %.3e -> %.4f keV'%(1/np.sqrt((np.abs(S)**2/J).sum())*np.sqrt(0.5)*0+0, 0))
res={}
for E in (15,2615):
    on=[int(r['onset_sample']) for r in ev if int(r['energy_keV'])==E][:300]
    a=np.array([amp(win(t)) for t in on]); res[E]=a
    print(f'{E} keV: mean {a.mean():.6g} std {a.std():.3e} -> sigma {a.std()/a.mean()*E:.4f} keV')
an=np.array([amp(win(int(r['trigger_sample']))) for r in nz])
print(f'noise windows: std {an.std():.3e} -> sigma {an.std()*2615/res[2615].mean():.4f} keV (in-sample PSD)')
Sn=np.abs(S)**2/J
print('theory sigma: %.3e  (only k>0: %.3e)'%(np.sqrt(1/(2*Sn.sum())), np.sqrt(1/(2*Sn[1:].sum()))))
print('fraction of 1/sigma^2 from DC bin: %.3f, from <1 Hz: %.3f, <5 Hz: %.3f'%(Sn[0]/Sn.sum(), Sn[:11].sum()/Sn.sum(), Sn[:51].sum()/Sn.sum()))
np.save('J_numpy_%d.npy'%run, J)
