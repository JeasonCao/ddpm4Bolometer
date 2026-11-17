"""Audit saved draft arithmetic/provenance without generating any waveforms."""
import argparse,csv,json,hashlib
from pathlib import Path
import numpy as np

def read(p):
    with p.open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))
def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output
    s=json.loads((out/'design_summary.json').read_text(encoding='utf-8'));rows=read(out/'targets_17_energy_points.csv');support=read(out/'training_support_checks.csv');scan=read(out/'slope_attenuation_scan.csv')
    keys=['tau_r','tau_fast','tau_slow'];energies=[int(r['energy_kev']) for r in rows]
    assert energies==[1,2,3,5,10,15,20,25,30,40,50,239,511,727,1461,2615,3034]
    anchor=np.array([float(s['anchor_source'][k+'_ms']) for k in keys]);slopes=np.array(s['A_slopes_ms_per_MeV'])*s['slope_factor_A']
    for r,e in zip(rows,energies):
        expected=anchor+slopes*(max(e,511)-2615)/1000
        np.testing.assert_allclose([float(r[k+'_center_ms']) for k in keys],expected,rtol=0,atol=1e-9)
        assert all(r[k+'_clean_sd_ms']=='' for k in keys)
        for k in keys:
            for group in ['A','B']:
                for width in ['sd','h68']:assert float(r[f'{k}_{group}_observed_{width}_ms'])>0
    ref=rows[energies.index(511)]
    for r,e in zip(rows,energies):
        if e<511:
            for k in keys:
                assert r[k+'_center_ms']==ref[k+'_center_ms']
                assert r[k+'_A_observed_sd_ms']==ref[k+'_A_observed_sd_ms']
    for k in keys:
        for g in ['A','B']:
            for w in ['sd','h68']:assert rows[-1][f'{k}_{g}_observed_{w}_ms']==rows[-2][f'{k}_{g}_observed_{w}_ms']
    assert len(support)==15
    for r in support:
        assert r['A_center_inside_16_84']=='True' and r['A_2sd_inside_2p5_97p5']=='True'
        assert float(r['center_3tau_knn_sparsity_percentile'])<=95
    assert scan[0]['marginal_pass']=='True' and scan[0]['center_joint_pass']=='False'
    selected=next(r for r in scan if abs(float(r['slope_factor'])-s['slope_factor_A'])<1e-8)
    assert selected['all_pass']=='True'
    for r in s['inputs']:assert hashlib.sha256(Path(r['path']).read_bytes()).hexdigest()==r['sha256']
    result=dict(passed=True,rows=17,all_clean_widths_unset=True,low_energy_freeze_verified=True,high_energy_width_freeze_verified=True,linear_centers_verified=True,all_high_center_density_and_marginal_checks_pass=True,full_CUORE_slope_rejected_by_joint_guard=True,inputs_unchanged=True,limits='Checks establish draft arithmetic/provenance only, not waveform feasibility or intrinsic width identification')
    (out/'independent_verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result))
if __name__=='__main__':main()
