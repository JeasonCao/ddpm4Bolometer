"""Compare coarse-seeded/full-sample fitting with direct full-sample multistart."""
import argparse,json,concurrent.futures
from pathlib import Path
import numpy as np
from run_study import worker
def main():
 p=argparse.ArgumentParser()
 for k in ('data','pilot'):p.add_argument('--'+k,required=True)
 p.add_argument('--workers',type=int,default=8);a=p.parse_args();out=Path(a.pilot)
 rr=[json.loads(s) for s in (out/'fits.jsonl').read_text().splitlines()];selected=[]
 for cls in (0,1):
  ss=sorted([r for r in rr if r['config']=='moderate' and r['label']==cls and 'Training' in r['file']],key=lambda r:r['amplitude_adc'])
  selected += [ss[i] for i in np.linspace(0,len(ss)-1,12,dtype=int)]
 jobs=[(str(Path(a.data)/r['file']),r['row'],r['label'],'moderate','full') for r in selected]
 with concurrent.futures.ProcessPoolExecutor(max_workers=a.workers) as pool:new=list(pool.map(worker,jobs))
 report=[dict(file=x['file'],row=x['row'],label=x['label'],direct=x['nrmse'],coarse_then_full=y['nrmse'],relative_rms_change=y['nrmse']/x['nrmse']-1) for x,y in zip(selected,new)]
 (out/'coarse_initialization_check.json').write_text(json.dumps(report,indent=2))
 print('Maximum relative RMS increase:',max(r['relative_rms_change'] for r in report))
if __name__=='__main__':main()
