"""Read-only verification of final fit files, source identities and sampled reconstructions."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import sys,json,argparse
from pathlib import Path
import numpy as np,h5py
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from experiments.cuore_readout_fit.run_study import sha,load_event,dump
from experiments.cuore_readout_fit.model import Forward

def main(a):
    root=Path(a.input);data=Path(a.data)
    manifest=json.loads((root/'manifest.json').read_text())
    full=[json.loads(s) for s in (root/'full_single.jsonl').read_text().splitlines()]
    lookup={(r['source_file'],r['row']):r for r in full}
    assert len(full)==len(lookup)==5000
    checks={}
    with h5py.File(data/'cuoreTraining.h5') as train,h5py.File(data/'cuoreTest.h5') as test:
        assert np.all(train['Labels'][:4500]==0)
        assert int(test.attrs['cleanPulseCount'])==500
        assert str(test.attrs['rowOrder'])=='clean rows followed by pileup rows'
        assert not set(train['eventId'][:4500].ravel()) & set(test['eventId'][:500].ravel())
    for name,count in [('cuoreTraining.h5',4500),('cuoreTest.h5',500)]:
        assert sha(data/name)==manifest['sources'][name]['sha256']
        with h5py.File(data/name) as src,h5py.File(root/(Path(name).stem+'_readout_fit.h5')) as fit:
            assert 'Waveform' not in fit
            assert fit.attrs['source_sha256']==manifest['sources'][name]['sha256']
            np.testing.assert_array_equal(fit['row'][:],np.arange(count))
            np.testing.assert_array_equal(fit['event_id'][:],src['eventId'][:count].reshape(-1))
            for f in ['normalizationScale','normalizationMaximum','normalizationOffset']:
                np.testing.assert_array_equal(fit[f][:],src[f][:count])
            assert np.all(np.isfinite(fit['p'][:])) and np.all(np.isfinite(fit['tau'][:]))
            assert np.all(np.diff(fit['tau'][:],axis=1)>=0)
            indices=list(np.unique(np.r_[0,count-1,np.linspace(0,count-1,12,dtype=int)]))
            errors=[];discretization=[]
            for row in indices:
                r=lookup[name,int(row)]
                np.testing.assert_array_equal(fit['p'][row],r['p'])
                y,_=load_event(data,name,int(row))
                fw=Forward(len(y),'rc',fit.attrs['fc_hz'],fit.attrs['rc_s'],2)
                pred=fw.evaluate(r['p'])[0]
                errors.append(abs(np.sqrt(np.mean((y-pred)**2))-fit['nrmse'][row]))
                fine=Forward(len(y),'rc',fit.attrs['fc_hz'],fit.attrs['rc_s'],4).evaluate(r['p'])[0]
                discretization.append(float(np.max(abs(pred-fine))))
            assert max(errors)<1e-12
            assert max(discretization)<1e-4
            checks[name]=dict(n=count,source_sha256=manifest['sources'][name]['sha256'],
                waveform_stored=False,reconstructed_rms_max_error=float(max(errors)),
                actual_fit_2vs4_oversample_max_difference=max(discretization),
                fit_h5_sha256=sha(root/(Path(name).stem+'_readout_fit.h5')))
    dump(root/'verification.json',dict(passed=True,checks=checks))
    print(json.dumps(checks,indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--data',required=True)
    main(p.parse_args())
