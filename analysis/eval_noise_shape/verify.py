"""Independently verify exported per-event statistics against existing noisy fits."""
import argparse,csv,json
from pathlib import Path
import h5py,numpy as np
from run_analysis import KEYS,dump,sha

def main(a):
 out=Path(a.output);manifest=json.loads((out/'manifest.json').read_text(encoding='utf-8'))
 rows=list(csv.DictReader((out/'scatter_statistics.csv').open(encoding='utf-8-sig')))
 assert len(rows)==102
 checked=0;count=0
 with h5py.File(out/'per_event_scatter.h5') as saved:
  for source in manifest['sources']:
   path=Path(source['fit']);level=path.parent.name.removeprefix('efficiency_')
   with h5py.File(path,'r') as h:
    assert sha(path)==source['fit_sha256'];e=int(h.attrs['energy_kev']);g=saved[f'{level}/{e}keV']
    y=np.column_stack([h[k+'_noisy'][:].astype(float)*1000 for k in KEYS]);valid=h['success_noisy'][:]&np.isfinite(y).all(axis=1)
    np.testing.assert_array_equal(g['tau_ms'][:],y);np.testing.assert_array_equal(g['valid'][:],valid);np.testing.assert_array_equal(g['source_row'][:],np.arange(len(y)))
    count+=len(y)
    for j,key in enumerate(KEYS):
     row=next(r for r in rows if r['noise_level']==level and int(r['energy_kev'])==e and r['feature']==key)
     v=y[valid,j];q=np.quantile(v,[.16,.5,.84]);assert int(row['n'])==len(v)
     np.testing.assert_allclose([float(row['std_ms']),float(row['median_ms']),float(row['half68_ms'])],[np.std(v,ddof=1),q[1],(q[2]-q[0])/2],rtol=1e-12,atol=1e-12)
     checked+=1
 assert count==170000 and checked==102
 dump(out/'independent_verification.json',dict(passed=True,events_checked=count,summary_rows_recomputed=checked,fit_hashes_unchanged=True))
 print('Verified 170000 source rows and all 102 summaries.')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);main(p.parse_args())
