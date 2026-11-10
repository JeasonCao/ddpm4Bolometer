"""Standalone, resumable clean fitting. No DDPM, no writes to input files."""
from pathlib import Path
import os
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'deps'))
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'

import argparse
import ast
import hashlib
import importlib.util
import json
import multiprocessing as mp
import time
import traceback
import warnings
from datetime import datetime
import h5py
import numpy as np
import scipy

SOURCE = Path('D:/storage/research/DDPM4bolometer/simu_data/clean')
REPO = Path('D:/坚果云存储/复旦事宜/科研/CUPID/paperwriting/Denoising/ddpm4Bolometer#')
NOTES = Path('D:/Obsidian/note/课题/DDPM4Bolometer')
OUTPUT = NOTES / 'results/fit_results/train_clean_triexp'
SNAP = ROOT / 'source_snapshot'
SCHEMA = 'train_clean_triexp_v1'
CONFIG = dict(schema=SCHEMA, workers=6, timeout_seconds=120, pilot_per_shard=50,
              seed=20261107, tau_constraint=0.20, max_projected_hours=8,
              max_run_hours=12, min_single_success=0.95,
              max_bad_single_residual_fraction=0.10, bad_relative_rms=0.05,
              max_single_tau_boundary_fraction=0.20,
              max_single_degenerate_fraction=0.20,
              boundary_range_fraction=0.001, degeneracy_relative_gap=0.05)
PULSE_FIELDS = ['amp', 'amp_raw', 'onset', 't_rise', 't_fall1', 't_fall2', 'alpha',
                'peak_time', 'onset_error']
SCALAR_FIELDS = ['baseline', 'amp_maxbase', 'residual_rms', 'relative_rms',
                 'fit_pearsonr', 'chi2', 'noise_sigma', 'fit_time',
                 'seg1_end_t', 'seg2_start_t', 'initial_sensitivity']
BOOL_FIELDS = ['success', 'timeout', 'tau_boundary', 'decay_degenerate', 'done']


def now():
    return datetime.now().astimezone().isoformat(timespec='seconds')


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    os.replace(tmp, path)


def note(message):
    path = NOTES / 'diary/2026-11-07.md'
    with path.open('a', encoding='utf-8') as f:
        f.write(f'\n- [{now()}] clean 三指数后台任务：{message}\n')


def status(state, **details):
    info = dict(time=now(), state=state, pid=os.getpid(), **details)
    atomic_json(ROOT / 'status.json', info)
    print(json.dumps(info, ensure_ascii=False), flush=True)


def load_fitter():
    spec = importlib.util.spec_from_file_location('frozen_clean_fit', SNAP / 'fit.py')
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def prepare():
    SNAP.mkdir(parents=True, exist_ok=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    paths = ['src/basics/fit.py', 'src/basics/amplitude.py', 'src/pulse/simulator.py',
             'src/pulse/generate.py', 'scripts/generate_clean_shards.py', 'scripts/run_fitting.py']
    hashes = {}
    for rel in paths:
        dst = SNAP / Path(rel).name
        if not dst.exists():
            with dst.open('xb') as f:
                f.write((REPO / rel).read_bytes())
        hashes[rel] = sha(dst)
    # Parse constants only: these are code-derived settings, not HDF5 historical truth.
    tree = ast.parse((SNAP / 'simulator.py').read_text(encoding='utf-8-sig'))
    defaults = {}
    random_names = {'R0','T0','lambda0','q','g_ec','a_ec','C_p','T_base'}
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == 'DetectorParams':
            for item in node.body:
                if isinstance(item, ast.AnnAssign) and item.target.id not in random_names:
                    defaults[item.target.id] = ast.literal_eval(item.value)
    info = dict(schema=SCHEMA, config=CONFIG, code_hashes=hashes,
                runner_sha256=sha(__file__), python=sys.version,
                versions=dict(numpy=np.__version__, scipy=scipy.__version__, h5py=h5py.__version__),
                fixed_settings_from_current_code=defaults,
                generation_settings_from_current_code=dict(method='Radau',rtol=1e-8,atol=1e-12,
                    internal_sample_hz=10000,max_r_ratio=0.02,max_loop_gain=0.8),
                provenance_warning='Fixed settings come from archived current code, not source HDF5; historical generation version unverified.',
                source_files=[])
    for path in sorted(SOURCE.glob('clean_*.h5')):
        with h5py.File(path, 'r') as f:
            n, ns = f['waveforms'].shape
            assert ns == int(f.attrs['n_samples'])
            assert int(f.attrs['n_pulses']) == n
            assert float(f.attrs['f_sample']) > 0
            for k in ['params','equilibrium','energies_1','energies_2','onsets_1','onsets_2','is_pileup']:
                assert k in f, k
            info['source_files'].append(dict(path=str(path), sha256=sha(path), count=n,
                samples=ns, fs=float(f.attrs['f_sample']), size_bytes=path.stat().st_size))
    assert len(info['source_files']) == 10
    assert sum(x['count'] for x in info['source_files']) == 90500
    fingerprint = hashlib.sha256(json.dumps(info, sort_keys=True).encode()).hexdigest()
    info['fingerprint'] = fingerprint
    old = ROOT / 'manifest.json'
    if old.exists():
        assert json.loads(old.read_text(encoding='utf-8'))['fingerprint'] == fingerprint, 'Run configuration/source changed; use new run directory.'
    atomic_json(old, info)
    return info


def blank():
    r = {k: np.nan for k in SCALAR_FIELDS}
    r.update({k: np.full(2, np.nan) for k in PULSE_FIELDS})
    r.update({k: False for k in BOOL_FIELDS})
    r.update(raw_params=np.full(13,np.nan), bounds_lower=np.full(13,np.nan),
             bounds_upper=np.full(13,np.nan), boundary_mask=np.zeros(13,bool),
             detected_peak_times=np.full(2,np.nan), message='', warning='', n_params=0)
    return r


def fit_one(mod, wave, n_pulses, fs, true_onsets, sensitivity=False):
    r = blank()
    started = time.monotonic()
    r['amp_maxbase'] = float(wave.max() - np.median(wave[:max(1,len(wave)//20)]))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        fr = mod.fit_pulse(wave, n_pulses=n_pulses, fs=fs, tau_constraint=CONFIG['tau_constraint'])
    r['warning'] = '; '.join(sorted(set(str(x.message) for x in caught)))[:4000]
    r['message'] = fr.message
    if not fr.success:
        r.update(fit_time=time.monotonic()-started, done=True)
        return r
    p = fr.params
    n = len(p)
    r['n_params'] = n
    r['raw_params'][:n] = p
    r['baseline'] = fr.baseline
    t = np.arange(len(wave))/fs
    model = fr.model_fn(t, *p)
    assert np.all(np.isfinite(model)), 'Nonfinite reconstruction'
    order = np.argsort(fr.onset_times)
    r['onset'][:n_pulses] = np.asarray(fr.onset_times)[order]
    for key, values in [('amp',fr.peak_amps),('t_rise',fr.tau_r),('t_fall1',fr.tau_d),
                        ('t_fall2',fr.tau_d2),('alpha',fr.alpha),('peak_time',fr.peak_times)]:
        r[key][:n_pulses] = np.asarray(values)[order]
    r['amp_raw'][:n_pulses] = np.asarray([p[1]] if n_pulses==1 else [p[1],p[7]])[order]
    r['onset_error'][:n_pulses] = r['onset'][:n_pulses]-np.sort(true_onsets[:n_pulses])
    # Independently reconstruct from the exact scalar fields that will be stored.
    rebuilt = np.full(len(wave), r['baseline'])
    for j in range(n_pulses):
        rebuilt += mod.biexp_single(t,0,r['amp_raw'][j],r['onset'][j],r['t_rise'][j],
                                    r['alpha'][j],r['t_fall1'][j],r['t_fall2'][j])
    np.testing.assert_allclose(rebuilt, model, rtol=1e-10, atol=1e-12)
    residual = wave-model
    r['residual_rms'] = float(np.sqrt(np.mean(residual**2)))
    r['relative_rms'] = r['residual_rms']/max(abs(r['amp_maxbase']),1e-12)
    r['fit_pearsonr'] = float(np.corrcoef(wave,model)[0,1]) if np.std(model)>0 and np.std(wave)>0 else np.nan
    r['chi2'], r['noise_sigma'] = fr.chi2_per_ndf, fr.noise_sigma
    if n_pulses == 1:
        _,lo,hi = mod._setup_single(wave,fs)
        r['bounds_lower'][:7],r['bounds_upper'][:7] = lo,hi
    else:
        split = int(round(fr.seg2_start_t*fs))
        _,lo,hi = mod._setup_single(wave[:split],fs)
        hi[2] = (split-1)/fs
        # Combined B=B1+B2. First-stage B1 bounds remain in snapshot, not this combined vector.
        r['bounds_lower'][1:7],r['bounds_upper'][1:7] = lo[1:],hi[1:]
        tr = p[3]
        r['bounds_lower'][7:] = [0,split/fs,max(mod._TAU_R_MIN,tr*0.8),mod._ALPHA_MIN,
                                 max(mod._TAU_D_MIN,tr*mod._TAU_ORDER_FACTOR),mod._TAU_D2_MIN]
        r['bounds_upper'][7:] = [mod._A_MAX,t[-1],min(mod._TAU_R_MAX,tr*1.2),mod._ALPHA_MAX,
                                 mod._TAU_D_MAX,mod._TAU_D2_MAX]
        r['seg1_end_t'],r['seg2_start_t'] = fr.seg1_end_t,fr.seg2_start_t
        r['detected_peak_times'][:] = fr.detected_peak_times
    lo,hi = r['bounds_lower'],r['bounds_upper']
    margin = CONFIG['boundary_range_fraction']*(hi-lo)
    r['boundary_mask'] = ((r['raw_params']-lo)<=margin)|((hi-r['raw_params'])<=margin)
    r['tau_boundary'] = bool(np.any(r['boundary_mask'][[3,5,6] if n_pulses==1 else [3,5,6,9,11,12]]))
    r['decay_degenerate'] = bool(np.any((r['t_fall2'][:n_pulses]-r['t_fall1'][:n_pulses]) /
                                      r['t_fall2'][:n_pulses] < CONFIG['degeneracy_relative_gap']))
    # Diagnostic only: alternative initial guess, same objective and bounds; never replace main result.
    if sensitivity and n_pulses == 1:
        p0,lo7,hi7 = mod._setup_single(wave,fs)
        p0[4],p0[5],p0[6] = 0.25,0.6,1.5
        alt,msg = mod._run_curve_fit(mod.biexp_single,t,wave,p0,lo7,hi7,None)
        if alt is not None:
            other = mod._build_result(alt,mod.biexp_single,wave,fs)
            orig = np.array([fr.tau_r[0],fr.tau_d[0],fr.tau_d2[0]])
            vals = np.array([other.tau_r[0],other.tau_d[0],other.tau_d2[0]])
            r['initial_sensitivity'] = float(np.max(np.abs(vals-orig)/np.maximum(orig,1e-12)))
    r.update(success=True,done=True,fit_time=time.monotonic()-started)
    return r


def worker(conn):
    mod = load_fitter()
    while True:
        task = conn.recv()
        if task is None:
            return
        jobid,wave,n,fs,onsets,sensitivity = task
        try:
            result = fit_one(mod,wave,n,fs,onsets,sensitivity)
        except (AssertionError, FloatingPointError):
            conn.send((jobid,dict(fatal=traceback.format_exc())))
            continue
        except Exception:
            result=blank()
            result.update(done=True,message=traceback.format_exc()[-4000:])
        conn.send((jobid,result))


class Workers:
    def __init__(self, n):
        self.ctx=mp.get_context('spawn')
        self.slots=[self.make() for _ in range(n)]

    def make(self):
        parent,child=self.ctx.Pipe()
        proc=self.ctx.Process(target=worker,args=(child,),daemon=True)
        proc.start()
        child.close()
        return dict(proc=proc,conn=parent,job=None,start=0)

    def close(self):
        for s in self.slots:
            if s['proc'].is_alive():
                s['proc'].terminate()
            s['proc'].join(timeout=5)
            s['conn'].close()

    def run(self, jobs, consume, heartbeat):
        jobs=iter(jobs)
        exhausted=False
        last=time.monotonic()
        while True:
            for i,s in enumerate(self.slots):
                if s['job'] is not None:
                    failed=None
                    if s['conn'].poll():
                        try:
                            job,result=s['conn'].recv()
                            assert job==s['job']
                            if 'fatal' in result:
                                raise RuntimeError(result['fatal'])
                            consume(job,result)
                            s['job']=None
                        except EOFError:
                            failed='worker exited unexpectedly'
                    elif time.monotonic()-s['start']>CONFIG['timeout_seconds']:
                        failed='fit timeout'
                    elif not s['proc'].is_alive():
                        failed='worker exited unexpectedly'
                    if failed:
                        r=blank()
                        r.update(done=True,timeout=failed=='fit timeout',message=failed,
                                 fit_time=time.monotonic()-s['start'])
                        consume(s['job'],r)
                        s['proc'].terminate()
                        s['proc'].join(timeout=5)
                        s['conn'].close()
                        self.slots[i]=s=self.make()
                if s['job'] is None and not exhausted:
                    try:
                        task=next(jobs)
                    except StopIteration:
                        exhausted=True
                    else:
                        s['conn'].send(task)
                        s['job']=task[0]
                        s['start']=time.monotonic()
            if exhausted and all(s['job'] is None for s in self.slots):
                break
            if time.monotonic()-last>20:
                heartbeat()
                last=time.monotonic()
            time.sleep(0.03)


def choose_pilot(f, rng):
    selected=[]
    pile=f['is_pileup'][:].astype(bool)
    for mask,values,count in [(~pile,f['energies_1'][:],30),
                               (pile,f['onsets_2'][:]-f['onsets_1'][:],20)]:
        indices=np.flatnonzero(mask)
        bins=np.array_split(indices[np.argsort(values[indices])],5)
        for group in bins:
            selected.extend(rng.choice(group,count//5,replace=False).tolist())
    return np.array(sorted(selected),dtype=np.int64)


def copy_metadata(src,dst,indices):
    for k,v in src.attrs.items():
        dst.attrs[k]=v
    def copy_group(a,b):
        for name,obj in a.items():
            if name=='waveforms' and a.name=='/':
                continue
            if isinstance(obj,h5py.Group):
                out=b.create_group(name)
                copy_group(obj,out)
            else:
                if obj.ndim and obj.shape[0]==len(src['waveforms']):
                    data=obj[indices]
                else:
                    data=obj[()]
                out=b.create_dataset(name,data=data,dtype=obj.dtype)
            for key,value in obj.attrs.items():
                out.attrs[key]=value
    copy_group(src,dst)


def init_output(path,source,indices,manifest,item):
    with h5py.File(source,'r') as src,h5py.File(path,'x') as out:
        copy_metadata(src,out,indices)
        out.attrs.update(schema=SCHEMA,complete=False,source_path=str(source),
                         source_shard=source.name,source_sha256=item['sha256'],
                         fingerprint=manifest['fingerprint'],created_at=now(),
                         output_row_count=len(indices),pulse_order='ascending fitted onset',
                         units='V, seconds, keV; params/equilibrium copied in original SI units',
                         chi2_note='Noise sigma floored at 1e-12 V. Do not gate clean fits on chi2.',
                         n_pulses_note='Inherited source attr n_pulses means SOURCE WINDOW COUNT, not per-row multiplicity.')
        out.create_dataset('window_idx',data=indices)
        out.create_dataset('n_pulses_fit',data=1+src['is_pileup'][indices].astype(np.uint8))
        out.create_dataset('n_params',shape=(len(indices),),dtype='u1')
        for key in PULSE_FIELDS+SCALAR_FIELDS:
            shape=(len(indices),2) if key in PULSE_FIELDS else (len(indices),)
            out.create_dataset(key,shape=shape,dtype='f8',fillvalue=np.nan)
        for key in BOOL_FIELDS:
            out.create_dataset(key,shape=(len(indices),),dtype='?',fillvalue=False)
        for key in ['raw_params','bounds_lower','bounds_upper']:
            out.create_dataset(key,shape=(len(indices),13),dtype='f8',fillvalue=np.nan)
        out.create_dataset('boundary_mask',shape=(len(indices),13),dtype='?')
        out.create_dataset('detected_peak_times',shape=(len(indices),2),dtype='f8',fillvalue=np.nan)
        for key in ['message','warning']:
            out.create_dataset(key,shape=(len(indices),),dtype=h5py.string_dtype('utf-8'))
        out.create_dataset('provenance_json',data=json.dumps(manifest,ensure_ascii=False),dtype=h5py.string_dtype('utf-8'))


def verify(path,source,manifest,item,complete=True):
    with h5py.File(path,'r') as out,h5py.File(source,'r') as src:
        assert out.attrs['fingerprint']==manifest['fingerprint']
        assert out.attrs['source_sha256']==item['sha256']
        assert 'waveforms' not in out
        idx=out['window_idx'][:]
        assert len(np.unique(idx))==len(idx)
        if complete:
            assert np.all(out['done'][:])
        def compare(a,b):
            for k,obj in a.items():
                if k=='waveforms' and a.name=='/':
                    continue
                if isinstance(obj,h5py.Group):
                    compare(obj,b[k])
                else:
                    expected=obj[idx] if obj.ndim and obj.shape[0]==len(src['waveforms']) else obj[()]
                    assert obj.dtype==b[k].dtype
                    np.testing.assert_array_equal(expected,b[k][()])
        compare(src,out)
        singles=out['n_pulses_fit'][:]==1
        for key in PULSE_FIELDS:
            assert np.all(np.isnan(out[key][:][singles,1]))
        # Verify persisted parameter columns reproduce raw-vector model, for EVERY successful row.
        mod=load_fitter()
        t=np.arange(0,item['samples'],79)/item['fs']
        for j in np.flatnonzero(out['success'][:]):
            n=int(out['n_params'][j])
            p=out['raw_params'][j,:n]
            fn=mod.biexp_single if n==7 else mod.biexp_double_free
            curve=fn(t,*p)
            restored=np.full(len(t),out['baseline'][j])
            for k in range(int(out['n_pulses_fit'][j])):
                restored+=mod.biexp_single(t,0,*[out[name][j,k] for name in
                    ['amp_raw','onset','t_rise','alpha','t_fall1','t_fall2']])
            np.testing.assert_allclose(restored,curve,rtol=1e-10,atol=1e-12)


def fit_file(item,indices,manifest,pool,pilot=False,deadline=None):
    source=Path(item['path'])
    directory=OUTPUT/'pilot' if pilot else OUTPUT
    directory.mkdir(parents=True,exist_ok=True)
    final=directory/('fit_'+source.name)
    partial=final.with_suffix('.partial.h5')
    if final.exists():
        verify(final,source,manifest,item)
        return final
    if not partial.exists():
        init_output(partial,source,indices,manifest,item)
    verify(partial,source,manifest,item,complete=False)
    with h5py.File(source,'r') as src,h5py.File(partial,'r+') as out:
        np.testing.assert_array_equal(out['window_idx'][:],indices)
        if not pilot:
            # Reuse audited pilot results; no need to fit the same window twice.
            pf=OUTPUT/'pilot'/('fit_'+source.name)
            with h5py.File(pf,'r') as p:
                for j,idx in enumerate(p['window_idx'][:]):
                    if not out['done'][idx]:
                        for key in blank():
                            out[key][idx]=p[key][j]
        pending=np.flatnonzero(~out['done'][:])
        written=0
        def heartbeat():
            out.flush()
            status('pilot_running' if pilot else 'full_running',shard=source.name,
                   done=int(np.sum(out['done'][:])),total=len(indices))
            if deadline and time.monotonic()>deadline:
                raise RuntimeError('Configured 12-hour run limit reached; partial output preserved.')
        def jobs():
            for row in pending:
                idx=int(indices[row])
                wave=src['waveforms'][idx]
                n=1+int(src['is_pileup'][idx])
                yield (int(row),wave,n,item['fs'],
                       [float(src['onsets_1'][idx]),float(src['onsets_2'][idx])],
                       bool(pilot and n==1 and row%5==0))
        def consume(row,result):
            nonlocal written
            for key,value in result.items():
                if key!='done':
                    out[key][row]=value
            out['done'][row]=True
            written+=1
            if written%25==0:
                heartbeat()
        heartbeat()
        pool.run(jobs(),consume,heartbeat)
        out.flush()
    verify(partial,source,manifest,item)
    with h5py.File(partial,'r+') as f:
        f.attrs['complete']=True
        f.attrs['completed_at']=now()
    assert not final.exists(), 'Refuse overwrite'
    partial.rename(final)
    if not pilot:
        note(f'完成并验证 {final.name}（{len(indices)} 窗），源行号/元信息/参数重建核对通过。')
    return final


def summarize(files):
    keys=['success','n_pulses_fit','tau_boundary','decay_degenerate','relative_rms','fit_time','initial_sensitivity','timeout']
    values={k:[] for k in keys}
    for path in files:
        with h5py.File(path,'r') as f:
            for k in keys:
                values[k].append(f[k][:])
    a={k:np.concatenate(v) for k,v in values.items()}
    summary={}
    for name,n in [('single',1),('pileup',2)]:
        mask=a['n_pulses_fit']==n
        good=mask&a['success']
        rel=a['relative_rms'][good]
        times=a['fit_time'][mask]
        times=times[np.isfinite(times)]
        summary[name]=dict(count=int(mask.sum()),success_count=int(good.sum()),
            success_rate=float(good.sum()/max(mask.sum(),1)),timeout_count=int(a['timeout'][mask].sum()),
            tau_boundary_fraction=float(a['tau_boundary'][good].mean()) if good.any() else 1.,
            degenerate_fraction=float(a['decay_degenerate'][good].mean()) if good.any() else 1.,
            bad_residual_fraction=float(np.mean(rel>CONFIG['bad_relative_rms'])) if len(rel) else 1.,
            relative_rms_median=float(np.median(rel)) if len(rel) else None,
            relative_rms_p95=float(np.quantile(rel,.95)) if len(rel) else None,
            mean_seconds=float(times.mean()) if len(times) else CONFIG['timeout_seconds'],
            p95_seconds=float(np.quantile(times,.95)) if len(times) else CONFIG['timeout_seconds'])
    sensitivity=a['initial_sensitivity'][np.isfinite(a['initial_sensitivity'])]
    summary['sensitivity']=dict(count=len(sensitivity),fraction_over_20_percent=float(np.mean(sensitivity>.20)) if len(sensitivity) else None)
    return summary


def smoke(manifest):
    mod=load_fitter()
    fs=1000.
    t=np.arange(10000)/fs
    wave=mod.biexp_single(t,0,0.05,1.5,.04,.4,.3,3.)
    result=fit_one(mod,wave,1,fs,[1.5,np.nan])
    assert result['success'] and result['relative_rms']<1e-3
    assert abs(result['onset'][0]-1.5)<.01
    item=manifest['source_files'][0]
    with h5py.File(item['path'],'r') as f:
        idx=int(np.flatnonzero(~f['is_pileup'][:].astype(bool))[0])
        real=fit_one(mod,f['waveforms'][idx],1,item['fs'],[f['onsets_1'][idx],np.nan])
    path=ROOT/'smoke.h5'
    if path.exists():
        raise FileExistsError('Smoke output exists; inspect it before repeating')
    init_output(path,Path(item['path']),np.array([idx]),manifest,item)
    with h5py.File(path,'r+') as f:
        for k,v in real.items():
            f[k][0]=v
    verify(path,Path(item['path']),manifest,item)
    # Exercise subprocess protocol and reuse without relying on a long overnight job.
    pool=Workers(1)
    received=[]
    try:
        pool.run([(0,wave,1,fs,[1.5,np.nan],False)],lambda j,r:received.append(r),lambda:None)
    finally:
        pool.close()
    assert len(received)==1 and received[0]['success']
    print(json.dumps(dict(smoke='passed',real_success=real['success'],real_relative_rms=real['relative_rms'])),flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args()
    if args.smoke:
        smoke(prepare())
        return
    # Exclusive OS lock is released on crash, allowing safe restart of partial files.
    import msvcrt
    lock=open(ROOT/'run.lock','a+b')
    lock.seek(0)
    if not lock.read(1):
        lock.write(b'0');lock.flush()
    lock.seek(0)
    msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    pool=None
    started=time.monotonic()
    try:
        status('preflight')
        manifest=prepare()
        note('后台进程已启动；先做跨10分片的500窗小样本，预设工程/科学诊断门控通过后自动进入全量。')
        pool=Workers(CONFIG['workers'])
        rng=np.random.default_rng(CONFIG['seed'])
        pilot=[]
        for item in manifest['source_files']:
            with h5py.File(item['path'],'r') as f:
                indices=choose_pilot(f,rng)
            pilot.append(fit_file(item,indices,manifest,pool,True,started+CONFIG['max_run_hours']*3600))
        summary=summarize(pilot)
        projected=(summary['single']['mean_seconds']*63486+summary['pileup']['mean_seconds']*27014)/CONFIG['workers']/3600*1.5
        s=summary['single']
        reasons=[]
        checks=[(s['success_rate']>=CONFIG['min_single_success'],'single success below 95%'),
                (s['bad_residual_fraction']<=CONFIG['max_bad_single_residual_fraction'],'over 10% of successful singles have relative RMS >5%'),
                (s['tau_boundary_fraction']<=CONFIG['max_single_tau_boundary_fraction'],'over 20% of successful singles touch tau bounds'),
                (s['degenerate_fraction']<=CONFIG['max_single_degenerate_fraction'],'over 20% of successful singles have nearly equal decays'),
                (projected<=CONFIG['max_projected_hours'],'projected runtime exceeds 8 hours')]
        if summary['sensitivity']['count']>=10:
            checks.append((summary['sensitivity']['fraction_over_20_percent']<=.20,
                           'over 20% of sensitivity probes change tau by >20% under alternate initialization'))
        reasons=[message for ok,message in checks if not ok]
        summary.update(projected_full_hours_with_50pct_margin=projected,gate_passed=not reasons,gate_reasons=reasons)
        atomic_json(ROOT/'pilot_report.json',summary)
        note('小样本完成：'+json.dumps(summary,ensure_ascii=False))
        if reasons:
            status('paused_for_scientific_review',report=str(ROOT/'pilot_report.json'),reasons=reasons)
            note('门控未通过，未启动全量；保留全部小样本与失败记录，需讨论科研方法后再继续。')
            return
        status('pilot_passed',projected_full_hours=projected)
        note('小样本门控通过，按用户授权自动进入全量；不更改旧拟合算法。')
        full=[]
        for item in manifest['source_files']:
            full.append(fit_file(item,np.arange(item['count'],dtype=np.int64),manifest,pool,
                                 deadline=started+CONFIG['max_run_hours']*3600))
        report=summarize(full)
        report.update(completed_at=now(),elapsed_hours=(time.monotonic()-started)/3600,
                      files=[str(p) for p in full],output_bytes=sum(p.stat().st_size for p in full))
        atomic_json(ROOT/'final_report.json',report)
        status('complete',report=str(ROOT/'final_report.json'))
        note('全量90500窗已完成并逐文件验证，报告：'+str(ROOT/'final_report.json'))
    except Exception:
        error=traceback.format_exc()
        status('stopped_error',error=error)
        note('任务异常停止，断点保留：'+error[-2000:])
        raise
    finally:
        if pool:
            pool.close()
        lock.close()


if __name__=='__main__':
    mp.freeze_support()
    main()
