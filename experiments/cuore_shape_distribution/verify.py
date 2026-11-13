"""Verify identities, extraction semantics, tabulated quantiles, and control counts."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import sys,json,csv,argparse
from pathlib import Path
import numpy as np,h5py
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from experiments.cuore_shape_distribution.run_study import shape_features,crossing,sha,dump

def main(a):
    root=Path(a.input);data=Path(a.data);fits=Path(a.fits)
    manifest=json.loads((root/'manifest.json').read_text())
    for f,value in manifest['sources'].items():assert sha(data/f)==value
    for f,value in manifest['fit_h5'].items():assert sha(fits/f)==value
    assert sha(fits/'full_single.jsonl')==manifest['fit_json_sha256']
    t=np.arange(5000)/1000.;y=np.where(t<=1,t,np.exp(-(t-1)/.3))
    rise=crossing(y,1000,.9,True)-crossing(y,1000,.1,True)
    decay=crossing(y,1000,.3,False)-crossing(y,1000,.9,False)
    assert abs(rise-.8)<1e-12 and abs(decay-.3*np.log(3))<1e-6
    sampled=[]
    with h5py.File(root/'shape_distributions.h5') as h:
        g=h['events'];n=len(g['index']);assert n==5000
        assert len(set(zip(g['source_file'][:],g['source_row'][:])))==5000
        assert sum(g['legacy_4980'][:])==4980
        for f,count in [('cuoreTraining.h5',4500),('cuoreTest.h5',500)]:
            ind=np.flatnonzero(g['source_file'][:]==f.encode())
            with h5py.File(data/f) as src,h5py.File(fits/(Path(f).stem+'_readout_fit.h5')) as fit:
                np.testing.assert_array_equal(g['source_row'][ind],np.arange(count))
                np.testing.assert_array_equal(g['event_id'][ind],src['eventId'][:count].ravel())
                np.testing.assert_array_equal(g['fit_p'][ind],fit['p'][:])
                for j,k in enumerate(['tau_r_ms','tau_fast_ms','tau_slow_ms']):
                    np.testing.assert_allclose(g[k][ind],fit['tau'][:,j]*1000,rtol=1e-14)
                for ix in ind[np.linspace(0,len(ind)-1,8,dtype=int)]:
                    row=int(g['source_row'][ix]);yy=(src['Waveform'][row]-g['offset_adc'][ix])/g['amplitude_adc'][ix]
                    check=shape_features(yy)
                    for key in ['rise_10_90_ms','decay_90_30_ms','t50_s']:
                        assert abs(check[key]-g[key][ix])<1e-10
                    sampled.append(int(ix))
        for j in range(4):
            b=h[f'bands/{j}'];q=b['raw_aligned_waveform_quantiles'][:]
            assert np.all(np.diff(q,axis=0)>=0)
            assert abs(q[2,3000]-.5)<.005
            assert len(b['event_indices'])==sum(g['band'][:]==j)
        with (root/'narrow_band_statistics.csv').open(encoding='utf-8-sig') as f:
            for r in csv.DictReader(f):
                sel=g['band'][:]==int(r['band'])
                if r['selection']=='legacy_4980':sel&=g['legacy_4980'][:]
                values=g[r['feature']][:][sel]
                assert int(r['n'])==len(values)
                np.testing.assert_allclose([float(r[k]) for k in ['q16','median','q84']],np.quantile(values,[.16,.5,.84]),rtol=1e-13)
    controls=[json.loads(s) for s in (root/'noise_controls.jsonl').read_text().splitlines()]
    assert len(controls)==len({(r['band'],r['kind'],r['replicate']) for r in controls})==8*manifest['noise_repeats']
    sens=[json.loads(s) for s in (root/'response_sensitivity.jsonl').read_text().splitlines()]
    assert len(sens)==len({(r['index'],r['fc'],r['rc']) for r in sens})==128
    result=dict(passed=True,source_hashes_unchanged=True,fit_hashes_unchanged=True,events=5000,
        sampled_recomputed_indices=sampled,controls=len(controls),control_success=sum(r['success'] for r in controls),
        sensitivity=len(sens),sensitivity_success=sum(r['success'] for r in sens),
        note='Numerical/source checks do not validate the stationary-noise approximation or convert amplitude to energy.')
    dump(root/'verification.json',result);print(json.dumps(result,indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ['input','data','fits']:p.add_argument('--'+k,required=True)
    main(p.parse_args())
