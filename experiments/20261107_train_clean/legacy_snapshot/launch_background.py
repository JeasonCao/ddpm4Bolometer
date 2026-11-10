"""Keep the machine awake only while the independent local fitting job runs."""
import ctypes
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime

root = Path(__file__).resolve().parent
kernel = ctypes.windll.kernel32
awake = bool(kernel.SetThreadExecutionState(0x80000001))
record = dict(supervisor_pid=os.getpid(), started_at=datetime.now().astimezone().isoformat(),
              idle_sleep_inhibited=awake)
try:
    with (root / 'run.log').open('a', encoding='utf-8') as log:
        child = subprocess.Popen([sys.executable, '-X', 'utf8', '-u',
                                  str(root / 'fit_train_clean.py')],
                                 cwd=root, stdout=log, stderr=subprocess.STDOUT)
        record['runner_pid'] = child.pid
        (root / 'process.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
        record['exit_code'] = child.wait()
finally:
    kernel.SetThreadExecutionState(0x80000000)
    record['finished_at'] = datetime.now().astimezone().isoformat()
    (root / 'process.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
