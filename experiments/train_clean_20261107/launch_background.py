"""Supervisor for an explicitly started fit run; never runs on import."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from experiments.train_clean_20261107.paths import ROOT
import argparse,ctypes,json,os,subprocess
from datetime import datetime

def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    ROOT.mkdir(parents=True,exist_ok=True)
    kernel=ctypes.windll.kernel32 if os.name=='nt' else None
    awake=bool(kernel.SetThreadExecutionState(0x80000001)) if kernel else False
    record=dict(supervisor_pid=os.getpid(),started_at=datetime.now().astimezone().isoformat(),idle_sleep_inhibited=awake)
    try:
        with (ROOT/'run.log').open('a',encoding='utf-8') as log:
            child=subprocess.Popen([sys.executable,'-X','utf8','-u',str(Path(__file__).with_name('fit_train_clean.py'))],
                                   cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            record['runner_pid']=child.pid
            (ROOT/'process.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
            record['exit_code']=child.wait()
    finally:
        if kernel:kernel.SetThreadExecutionState(0x80000000)
        record['finished_at']=datetime.now().astimezone().isoformat()
        (ROOT/'process.json').write_text(json.dumps(record,indent=2),encoding='utf-8')

if __name__=='__main__':main()
