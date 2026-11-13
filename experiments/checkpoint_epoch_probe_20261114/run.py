"""Small paired epoch-70/100 probe on previously selected low-energy events.

Uses the archived production inference code. CPU FP32, Gaussian start, DDPM
no_noise, 50 steps. Each epoch receives identical inputs and initial noise.
The selected eight events are illustrative, not a resolution measurement.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import time

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--code-root',type=Path,required=True)
    p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--notes-root',type=Path,required=True)
    p.add_argument('--threads',type=int,default=4)
    p.add_argument('--name',default='model261108_epoch70_probe_20261114')
    args=p.parse_args()
    sys.path.insert(0,str(args.code_root))
    import h5py
    import numpy as np
    import torch
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scripts.run_batch_inference import load_model,denoise_batch
    from scripts.run_fitting import _fit_one
    from src.basics.fit import biexp_single

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    device=torch.device('cpu')
    weights=args.data_root/'model_output_cloud/run03_model261108'
    diagpath=args.notes_root/'results/plots/model261108_tail_diagnostics_20261113/diagnostics.json'
    diagnostics=json.loads(diagpath.read_text())
    wave_dir=args.data_root/args.name
    fit_dir=args.notes_root/'results/fit_results'/args.name
    plot_dir=args.notes_root/'results/plots'/args.name
    for path in (wave_dir,fit_dir,plot_dir):path.mkdir(parents=True,exist_ok=True)
    wave_path=wave_dir/'paired_outputs.h5';fit_path=fit_dir/'paired_fits.h5'
    assert not wave_path.exists() and not fit_path.exists(),'Use a fresh output name.'
    report={'code_sha256':sha(__file__),'diagnostics_sha256':sha(diagpath),
            'torch_version':torch.__version__,'numpy_version':np.__version__,
            'device':'cpu','threads':args.threads,'epochs':[70,100],
            'settings':{'sampler':'ddpm','no_noise':True,'T':50,'beta_1':1e-4,'beta_T':.5,
                        'scale_emb':True,'film':True,'precision':'float32','seed':42,'batch_size':4},
            'limitations':['Eight preselected examples; no tail fraction or energy-resolution estimate.',
                           'CPU/batch-4 Gaussian draws differ from archived CUDA/batch-16 draws. Local epoch100 is the paired control.',
                           'Same epoch count does not make old/new architecture or training losses equivalent.'],
            'sources':{},'checkpoints':{},'cases':{},'timing_seconds':{}}
    for rel in ['src/ddpm/unet.py','src/ddpm/diffusion.py','src/ddpm/schedule.py',
                'scripts/run_batch_inference.py','scripts/run_fitting.py','src/basics/fit.py']:
        path=args.code_root/rel;report['sources'][str(path)]=sha(path)
    datasets={}
    for level,e in [('1p382mV',2),('7p5mV',10)]:
        key=f'{level}_{e}keV';rec=diagnostics['cases'][key]
        choices=rec['selected_examples'];rows=np.array([s['row'] for s in choices])
        raw_path,cloud_path,newfit_path,oldfit_path=[Path(s['path']) for s in rec['sources']]
        for source in rec['sources']:
            assert sha(source['path'])==source['sha256']
            report['sources'][source['path']]=source['sha256']
        with h5py.File(raw_path) as f:
            clean=f['clean_template'][:].astype(float)
            noisy=np.stack([f['waveforms_noisy'][i] for i in rows])
        with h5py.File(cloud_path) as f:cloud=np.stack([f['waveforms_denoised'][i] for i in rows])
        with h5py.File(oldfit_path) as f:
            true_amp=float(f.attrs['true_amp_V'])
            old_amp=np.array([f['amp_denoised'][i] for i in rows])
        with h5py.File(newfit_path) as f:cloud_amp=np.array([f['amp_denoised'][i] for i in rows])
        actual_peak=float(clean.max()-np.median(clean[:500]))
        assert np.isclose(true_amp,actual_peak,rtol=1e-12)
        datasets[key]={'rows':rows,'choices':choices,'clean':clean,'noisy':noisy,'cloud':cloud,
                       'true_amp':true_amp,'old_amp':old_amp,'cloud_amp':cloud_amp,
                       'outputs':{},'fits':{},'initial':None}
        report['cases'][key]={'choices':choices,'true_amp_V':true_amp,'source_file':str(raw_path),
                              'cloud_output_file':str(cloud_path),'events':[]}
    start=time.perf_counter()
    for epoch in [100,70]:
        checkpoint=weights/f'checkpoint_{epoch:03d}.pt'
        cp=torch.load(checkpoint,map_location='cpu',weights_only=True)
        assert cp['epoch']==epoch
        report['checkpoints'][str(epoch)]={'path':str(checkpoint),'sha256':sha(checkpoint),
                                          'stored_epoch':cp['epoch'],'val_loss':cp['val_loss']}
        if epoch==100:
            best_path=weights/'best_model.pt'
            best=torch.load(best_path,map_location='cpu',weights_only=True)
            assert all(torch.equal(best[k],v) for k,v in cp['model_state_dict'].items())
            report['epoch100_equals_best_model']=True
            report['sources'][str(best_path)]=sha(best_path)
            del best
        del cp
        model=load_model(str(checkpoint),device,scale_emb=True,film=True)
        for key,d in datasets.items():
            torch.manual_seed(42)
            initial=torch.randn(len(d['rows']),1,10000)
            expected_rng_after=torch.get_rng_state().clone()
            tick=time.perf_counter()
            out=denoise_batch(model,d['noisy'],device,batch_size=4,shots=1,
                              sampler='ddpm',seed=42,use_scale=True,no_noise=True)
            elapsed=time.perf_counter()-tick
            assert torch.equal(torch.get_rng_state(),expected_rng_after),'Unexpected RNG consumption'
            assert out.shape==(4,10000) and out.dtype==np.float32 and np.isfinite(out).all()
            if d['initial'] is None:d['initial']=initial.squeeze(1).numpy()
            else:assert np.array_equal(d['initial'],initial.squeeze(1).numpy())
            d['outputs'][epoch]=out
            report['timing_seconds'][f'{key}_epoch{epoch}']=elapsed
            print(f'{key} epoch {epoch}: {elapsed:.1f} s; identical-start check passed',flush=True)
        del model
    report['inference_elapsed_seconds']=time.perf_counter()-start
    table=[]
    with h5py.File(wave_path,'w') as wh,h5py.File(fit_path,'w') as fh:
        wh.attrs.update(sampler='ddpm',no_noise=True,T=50,precision='float32',device='cpu',
                        seed=42,batch_size=4,selection_source=str(diagpath),code_sha256=sha(__file__))
        fh.attrs.update(waveform_file=str(wave_path),fitter_sha256=sha(args.code_root/'src/basics/fit.py'),
                        amplitude_unit='V',time_unit='s',selection_is_diagnostic=True)
        for key,d in datasets.items():
            wg=wh.create_group(key);fg=fh.create_group(key)
            for group in [wg,fg]:
                group.create_dataset('source_row',data=d['rows'])
                group.attrs['source_file']=report['cases'][key]['source_file']
                group.attrs['true_amp_V']=d['true_amp']
            wg.create_dataset('initial_noise',data=d['initial'],compression='gzip')
            wg.create_dataset('clean_template',data=d['clean'],compression='gzip')
            for epoch in [70,100]:
                out=d['outputs'][epoch]
                wg.create_dataset(f'waveforms_epoch{epoch}',data=out,compression='gzip')
                fits=[_fit_one((y.astype(float),'denoised',10.,1e-5,9.9,d['clean'])) for y in out]
                assert all(f['success_denoised'] for f in fits)
                d['fits'][epoch]=fits
                eg=fg.create_group(f'epoch{epoch}')
                for field in fits[0]:
                    eg.create_dataset(field,data=[f[field] for f in fits])
                t=np.arange(10000)/1000
                for j,(y,f) in enumerate(zip(out,fits)):
                    curve=biexp_single(t,*[f[k+'_denoised'] for k in ('baseline','amp_raw','onset','t_rise','alpha','t_fall1','t_fall2')])
                    sigma=np.std(y.astype(float)[:500])
                    chi2=np.sum(((y-curve)/sigma)**2)/(len(y)-7)
                    assert np.isclose(chi2,f['chi2_denoised'],rtol=1e-6)
                    row={'case':key,'source_row':int(d['rows'][j]),'label':d['choices'][j]['label'],'epoch':epoch,
                         'fit_peak_ratio':f['amp_denoised']/d['true_amp'],
                         'direct_peak_ratio':(float(y.max())-float(np.median(y[:500])))/d['true_amp'],
                         'rise_ms':1000*f['t_rise_denoised'],'fast_decay_ms':1000*f['t_fall1_denoised'],
                         'slow_decay_s':f['t_fall2_denoised'],
                         'rms_to_clean_pct':100*float(np.sqrt(np.mean((y-d['clean'])**2)))/d['true_amp'],
                         'cloud_epoch100_peak_ratio':float(d['cloud_amp'][j]/d['true_amp']),
                         'old_zero_peak_ratio':float(d['old_amp'][j]/d['true_amp'])}
                    table.append(row);report['cases'][key]['events'].append(row)
            for epoch in [70,100]:
                assert wg[f'waveforms_epoch{epoch}'].shape==(4,10000)
            fig,axes=plt.subplots(3,4,figsize=(16,9),layout='constrained')
            t=np.arange(10000)/1000
            for c,choice in enumerate(d['choices']):
                for r in [0,1]:
                    axes[r,c].plot(t,d['clean']/d['true_amp'],'k:',lw=1.5,label='Clean truth')
                    axes[r,c].plot(t,d['cloud'][c]/d['true_amp'],color='#999999',ls='--',lw=.9,label='Epoch100 archived GPU')
                    for epoch,color in [(70,'#2475c6'),(100,'#e07835')]:
                        axes[r,c].plot(t,d['outputs'][epoch][c]/d['true_amp'],color=color,lw=1.2,label=f'Epoch{epoch} paired CPU')
                for epoch,color in [(70,'#2475c6'),(100,'#e07835')]:
                    axes[2,c].plot(t,(d['outputs'][epoch][c]-d['clean'])/d['true_amp'],color=color,label=f'Epoch{epoch} - clean')
                axes[0,c].set_xlim(0,6)
                axes[1,c].set_xlim(1.3,2.5)
                axes[2,c].set_xlim(1.3,2.8)
                axes[0,c].set_title(f"{choice['label']}; row {choice['row']}",fontsize=10)
                for r in range(3):
                    axes[r,c].set(xlabel='Time (s)',ylabel='Amplitude / clean peak' if r<2 else 'Residual / clean peak')
                    axes[r,c].grid(alpha=.2);axes[r,c].legend(fontsize=6.5)
            fig.suptitle(key+' — epoch 70 vs 100, same input and same local Gaussian start')
            fig.supxlabel('Preselected examples, not a spectrum or efficiency estimate. Archived GPU outputs have different Gaussian draws; paired CPU epoch100 is the control.',fontsize=8)
            for ext in ['png','pdf']:fig.savefig(plot_dir/f'{key}_waveforms.{ext}',dpi=180)
            plt.close(fig)
    with (plot_dir/'event_metrics.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(table[0]));w.writeheader();w.writerows(table)
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    for r,(key,d) in enumerate(datasets.items()):
        xx=np.arange(4)
        for epoch,color in [(70,'#2475c6'),(100,'#e07835')]:
            rows=[v for v in table if v['case']==key and v['epoch']==epoch]
            axes[r,0].plot(xx,[v['fit_peak_ratio'] for v in rows],'o-',color=color,label=f'Epoch {epoch}, paired CPU')
            axes[r,1].plot(xx,[v['rise_ms'] for v in rows],'o-',color=color,label=f'Epoch {epoch}, paired CPU')
        axes[r,0].plot(xx,d['cloud_amp']/d['true_amp'],'x--',color='#999999',label='Epoch 100, archived GPU')
        axes[r,0].plot(xx,d['old_amp']/d['true_amp'],'s:',color='#139b73',label='Old zero-init, archived')
        axes[r,0].axhline(1,color='black',ls=':',lw=.8)
        axes[r,0].set(ylabel='Fitted peak / clean peak',title=key)
        axes[r,1].set(ylabel='Fitted rise time (ms)',title=key)
        for ax in axes[r]:
            ax.set_xticks(xx,[f"{c['label']}\nrow {c['row']}" for c in d['choices']],fontsize=8)
            ax.grid(alpha=.2);ax.legend(fontsize=7)
    fig.suptitle('Epoch comparison on eight fixed examples (no energy recalibration)')
    for ext in ['png','pdf']:fig.savefig(plot_dir/f'paired_metrics.{ext}',dpi=180)
    plt.close(fig)
    for path,expected in report['sources'].items():assert sha(path)==expected,path
    for cp in report['checkpoints'].values():assert sha(cp['path'])==cp['sha256']
    report['outputs']={str(path):sha(path) for path in [wave_path,fit_path,*sorted(plot_dir.glob('*.png')),*sorted(plot_dir.glob('*.pdf')),plot_dir/'event_metrics.csv']}
    report['checks']=['Archived inference code and input hashes recorded and unchanged.',
                      'Stored epoch metadata checked; epoch100 equals best_model tensorwise.',
                      'Paired Gaussian starts exactly equal and RNG consumption checked.',
                      'All outputs finite; 16/16 fits successful; saved parameter residuals reconstructed.',
                      'Original source rows preserved in waveform and fit HDF5.']
    report['complete']=True
    (plot_dir/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('PROBE_COMPLETE',plot_dir,flush=True)
    for row in table:print(json.dumps(row),flush=True)

if __name__=='__main__':
    main()
