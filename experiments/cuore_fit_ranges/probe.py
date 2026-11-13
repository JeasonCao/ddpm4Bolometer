"""Original-coordinate range/initialization probe; separate from stable refit."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np,h5py
from fitter import RANGES,single
def main():
 p=argparse.ArgumentParser()
 p.add_argument('--data',required=True);p.add_argument('--output',required=True)
 p.add_argument('--samples',type=int,default=8);p.add_argument('--configs',nargs='+',default=list(RANGES),choices=list(RANGES))
 a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True);rows=[]
 with h5py.File(Path(a.data)/'cuoreTraining.h5','r') as f:
  amps=f['normalizationScale'][:4500,0];ids=np.argsort(amps)[np.linspace(0,4499,a.samples,dtype=int)]
  for i in ids:
   y=(f['Waveform'][i]-f['normalizationOffset'][i,0])/amps[i]
   for cfg in a.configs:
    rr=single(y,cfg,stride=5,max_nfev=1000)
    rows.append(dict(row=int(i),event_id=int(f['eventId'][i]),amplitude_adc=float(amps[i]),config=cfg,results=rr))
    print(i,cfg,rr[0]['nrmse'],rr[0]['success'],flush=True)
 result=dict(ranges=RANGES,stride=5,rows=rows,source_sha256=hashlib.sha256((Path(a.data)/'cuoreTraining.h5').read_bytes()).hexdigest(),
             code_sha256=hashlib.sha256((Path(__file__).parent/'fitter.py').read_bytes()).hexdigest())
 (out/'original_coordinate_probe.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
if __name__=='__main__':main()
