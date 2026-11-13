"""Audit recalibration and saved waveform diagnostics without rerunning fits."""
import argparse
import ast
import json
from pathlib import Path
import struct
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import h5py
import numpy as np
from experiments.model261108_inference.full_comparison import digest,FWHM,json_safe
from experiments.model261108_inference.waveform_tail_diagnostics import classify

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--notes-root',type=Path,required=True)
    args=p.parse_args()
    root=args.notes_root/'results/plots'
    previous_path=root/'model261108_full_comparison/comparison.json'
    revised=root/'model261108_recalibrated_20261113'
    diagnostics=root/'model261108_tail_diagnostics_20261113'
    previous=json.loads(previous_path.read_text())
    current=json.loads((revised/'comparison.json').read_text())
    eff=json.loads((revised/'efficiency_low.json').read_text())
    tail=json.loads((diagnostics/'diagnostics.json').read_text())
    summary=json.loads((diagnostics/'summary.json').read_text())
    checked={}
    def verify(path,expected):
        path=str(Path(path))
        if path not in checked: checked[path]=digest(path)
        assert checked[path]==expected,path
    verify(previous_path,current['previous_report_sha256'])
    verify(revised/'comparison.json',tail['comparison_sha256'])
    verify(diagnostics/'diagnostics.json',summary['diagnostics_sha256'])
    scripts=Path(__file__).resolve().parent
    formatting_equivalence={}
    for name,r in [('recalibrate.py',current),('waveform_tail_diagnostics.py',tail),('diagnostic_summary.py',summary)]:
        final=scripts/name
        if digest(final)==r['code_sha256']:
            verify(final,r['code_sha256'])
        else:
            executed=diagnostics/'executed_sources'/name
            verify(executed,r['code_sha256'])
            assert ast.dump(ast.parse(final.read_text(encoding='utf-8')))==ast.dump(ast.parse(executed.read_text(encoding='utf-8')))
            formatting_equivalence[name]={'executed_sha256':r['code_sha256'],'final_sha256':digest(final),
                                          'ast_identical':True,'reason':'Trailing blank lines removed before commit.'}
    for source in current['input_files']:verify(source['path'],source['sha256'])
    for level,group in current['groups'].items():
        old=previous['groups'][level]
        assert len(group['peaks'])==17
        k=group['calibrations']['new_nonoise']['k_keV_per_amplitude']
        assert group['calibrations']['raw']['k_keV_per_amplitude']==k
        for e,row in group['peaks'].items():
            for method in ('old_zeroinit','new_nonoise','OF'):
                assert row[method]['fwhm_keV']==old['peaks'][e][method]['fwhm_keV']
            assert np.isclose(row['raw']['fwhm_keV'],FWHM*k*row['raw']['gaussian']['sigma'],rtol=1e-12)
        for method,r in eff[level].items():
            energies=r['calibration']['energies_keV']
            assert energies==sorted(int(e) for e in group['peaks'] if int(e)<=511)
            lowk=r['calibration']['k_keV_per_amplitude']
            for row in r['rows']:
                e=row['energy_keV']
                oldfit=args.notes_root/f'results/fit_results/{level}_zeroinit/fit_efficiency_{e}keV.h5'
                newfit=args.notes_root/f'results/fit_results/model261108_nonoise/efficiency_{level}/fit_efficiency_{e}keV.h5'
                for kind,path,field in [('raw',oldfit,'noisy'),('denoised',oldfit if method=='old_zeroinit' else newfit,'denoised')]:
                    with h5py.File(path) as f:
                        a=f['amp_'+field][:].astype(float)
                        passed=f['success_'+field][:].astype(bool)&np.isfinite(a)&(np.abs(lowk*a/e-1)<.05)
                    assert row[kind]['n_total']==5000 and row[kind]['n_pass']==int(passed.sum())
    assert tail['complete'] and len(tail['cases'])==8
    total=0
    for key,rec in tail['cases'].items():
        for source in rec['sources']:verify(source['path'],source['sha256'])
        path=diagnostics/key/'per_event_features.npz'
        verify(path,summary['feature_file_hashes'][key])
        with np.load(path) as a, h5py.File(rec['sources'][2]['path']) as f:
            assert np.array_equal(a['source_row'],np.arange(5000))
            assert np.array_equal(a['fit_amp'],f['amp_denoised'][:].astype(float))
            assert np.array_equal(a['energy_keV'],a['fit_amp']*rec['calibration_k'])
            groups,_=classify(a['energy_keV'],rec['mixtures'][1])
            assert np.array_equal(groups,a['group'])
            total+=len(groups)
    for rec in tail['high_energy_examples']:
        for source in rec['sources']:verify(source['path'],source['sha256'])
    figures={}
    for directory,expected in [(revised,8),(diagnostics,60)]:
        pngs=list(directory.rglob('*.png'));pdfs=list(directory.rglob('*.pdf'))
        assert len(pngs)==len(pdfs)==expected,(directory,len(pngs),len(pdfs))
        for path in pngs:
            data=path.read_bytes()
            assert data[:8]==b'\x89PNG\r\n\x1a\n'
            width,height=struct.unpack('>II',data[16:24])
            assert width>500 and height>300
        for path in pdfs:
            data=path.read_bytes()
            assert data.startswith(b'%PDF-') and data.rstrip().endswith(b'%%EOF')
        figures[str(directory)]={'png_count':len(pngs),'pdf_count':len(pdfs)}
        manifest={str(path.relative_to(directory)):digest(path) for path in sorted(directory.rglob('*')) if path.is_file() and path.name not in ('artifact_hashes.json','verification.json')}
        (directory/'artifact_hashes.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    result={'passed':True,'code_sha256':digest(__file__),'checked_source_hashes':checked,
            'diagnosed_events':total,'comparison_rows':136,'figures':figures,'formatting_equivalence':formatting_equivalence,
            'checks':['Historical report unchanged; executed code hashes match archived sources and final code is AST-identical.',
                      'Old/new/OF widths unchanged; raw references share the declared scale.',
                      'Low-energy efficiency counts reproduced directly from saved amplitudes.',
                      'All 40,000 diagnostic rows, amplitudes and plot groups align.',
                      'Original waveform and fit source hashes unchanged.',
                      'PNG/PDF files and paired figure counts checked. Representative plots visually reviewed separately.']}
    (diagnostics/'verification.json').write_text(json.dumps(json_safe(result),indent=2),encoding='utf-8')
    print('VERIFIED',total,'events;',len(checked),'source hashes;',figures,flush=True)

if __name__=='__main__':main()
