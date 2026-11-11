"""Configure machine paths before starting Python; dependencies use the active environment."""
from pathlib import Path
import os
from datetime import datetime

REPO=Path(__file__).resolve().parents[2]
DATA=Path(os.environ.get('DDPM_DATA_ROOT',str(REPO/'data'))).expanduser().resolve()
ROOT=Path(os.environ.get('DDPM_CLEAN_RUN_DIR',str(DATA/'runs/train_clean_triexp'))).expanduser().resolve()
SOURCE=Path(os.environ.get('DDPM_CLEAN_SOURCE_DIR',str(DATA/'simu_data/clean'))).expanduser().resolve()
OUTPUT=Path(os.environ.get('DDPM_CLEAN_FIT_DIR',str(ROOT/'fit_results'))).expanduser().resolve()
PLOTS=Path(os.environ.get('DDPM_CLEAN_PLOT_DIR',str(ROOT/'plots'))).expanduser().resolve()
NOTES=Path(os.environ.get('DDPM_NOTES_ROOT',str(ROOT/'notes'))).expanduser().resolve()
SNAP=ROOT/'source_snapshot'

def diary_path():
    date=os.environ.get('DDPM_DIARY_DATE',datetime.now().astimezone().date().isoformat())
    return NOTES/'diary'/f'{date}.md'

def acquire_lock(path):
    """Exclusive nonblocking lock, released on handle close or process exit."""
    handle=open(path,'a+b')
    try:
        if os.name=='nt':
            import msvcrt
            handle.seek(0)
            if not handle.read(1):handle.write(b'0');handle.flush()
            handle.seek(0)
            msvcrt.locking(handle.fileno(),msvcrt.LK_NBLCK,1)
        else:
            import fcntl
            fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        return handle
    except Exception:
        handle.close()
        raise
