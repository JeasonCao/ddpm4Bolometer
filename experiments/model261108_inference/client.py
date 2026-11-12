"""Independent Windows downloader, priority analysis coordinator and shutdown gate."""
import ctypes
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
import traceback
import msvcrt

from common import digest, read_json, save_json, verify_record

HERE = Path(__file__).resolve().parent
LOCAL = Path(r'D:\storage\research\DDPM4bolometer\model261108_inference')
TRAIN_LOCAL = Path(r'D:\storage\research\DDPM4bolometer\model_output_cloud\run03_model261108')
REMOTE = '/root/autodl-fs/output/run03_model261108_inference'
TRAIN_REMOTE = '/root/autodl-fs/output/run03_model261108'
SERVER = '/root/autodl-tmp/model261108_inference/server.py'
HOST = 'root@connect.bjb3.seetacloud.com'
KEY = r'C:\Users\ovbb\.ssh\id_ed25519_autodl_bjb3_54384'
# Resolve-DnsName verified this endpoint on 2026-11-12. Preserve the original
# known-host identity while avoiding intermittent DNS failures during this run.
ENDPOINT_OPTIONS = ['-o', 'HostName=123.127.15.157', '-o',
                    'HostKeyAlias=[connect.bjb3.seetacloud.com]:54384',
                    '-o', 'StrictHostKeyChecking=yes']
SSH = ['ssh', '-i', KEY, '-p', '54384', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=20',
       '-o', 'ServerAliveInterval=30', '-o', 'ServerAliveCountMax=3'] + ENDPOINT_OPTIONS + [HOST]
SCP = ['scp', '-q', '-i', KEY, '-P', '54384', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=20',
       '-o', 'ServerAliveInterval=30', '-o', 'ServerAliveCountMax=3'] + ENDPOINT_OPTIONS


def command(args, timeout=3600):
    result = subprocess.run(args, text=True, capture_output=True, timeout=timeout)
    if result.returncode:
        raise RuntimeError(result.stderr[-2000:])
    return result.stdout


def remote_json(path):
    result = command(SSH + ['if test -f ' + shlex.quote(path) + '; then cat ' + shlex.quote(path) + '; else echo null; fi'], timeout=90)
    return json.loads(result)


def status(stage, **details):
    value = {'stage': stage, 'time': time.strftime('%Y-%m-%dT%H:%M:%S'), **details}
    save_json(LOCAL / 'client_status.json', value)
    print(value, flush=True)


def download(root, remote_root, record):
    target = root / record['path']
    if verify_record(root, record):
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + '.partial')
    command(SCP + [HOST + ':' + remote_root + '/' + record['path'], str(temporary)])
    assert temporary.stat().st_size == record['bytes'] and digest(temporary) == record['sha256']
    temporary.replace(target)


def main():
    LOCAL.mkdir(parents=True, exist_ok=True)
    lock = (LOCAL / 'client.lock').open('a+b')
    lock.seek(0)
    lock.write(b'0')
    lock.flush()
    lock.seek(0)
    msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
    # Keep the Windows host awake while transfers and analysis are active.
    assert ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    analysis = None
    analysis_log = None
    attempts = {2615: 0, 15: 0}
    while True:
        try:
            manifest = remote_json(REMOTE + '/manifest.json')
            training = remote_json(REMOTE + '/training_manifest.json')
            if training:
                for record in training['files']:
                    download(TRAIN_LOCAL, TRAIN_REMOTE, record)
                save_json(LOCAL / 'training_download_manifest.json', training)
            if manifest:
                for record in manifest['files']:
                    download(LOCAL, REMOTE, record)
                    save_json(LOCAL / 'inference_download_manifest.json', manifest)
                    # Analysis runs separately so its CPU work does not hold up downloads.
                    if analysis is None or analysis.poll() is not None:
                        if analysis_log is not None:
                            analysis_log.close()
                            analysis_log = None
                        for energy in (2615, 15):
                            ready = (LOCAL / f'efficiency_1p382mV/eval_efficiency_{energy}keV.h5').exists()
                            done = (LOCAL / f'analysis_{energy}_complete.json').exists()
                            if ready and not done and attempts[energy] < 3:
                                attempts[energy] += 1
                                analysis_log = (LOCAL / f'analysis_{energy}.log').open('a', encoding='utf-8')
                                analysis = subprocess.Popen([sys.executable, '-u', str(HERE / 'analyze.py'), '--energy', str(energy)],
                                                            stdout=analysis_log, stderr=subprocess.STDOUT,
                                                            creationflags=subprocess.CREATE_NO_WINDOW)
                                break
                status('downloading_and_analyzing', downloaded=len(manifest['files']), total=34,
                       inference_complete=manifest['complete'])
                done = all((LOCAL / f'analysis_{e}_complete.json').exists() for e in (2615, 15))
                if manifest['complete'] and training and done:
                    assert len(manifest['files']) == 34
                    assert all(verify_record(LOCAL, r) for r in manifest['files'])
                    assert all(verify_record(TRAIN_LOCAL, r) for r in training['files'])
                    for record in manifest.get('auxiliary_files', []):
                        download(LOCAL, REMOTE, record)
                    receipt = {'all_verified': True, 'analysis_complete': True,
                               'inference_files': manifest['files'], 'training_files': training['files'],
                               'auxiliary_files': manifest.get('auxiliary_files', [])}
                    save_json(LOCAL / 'local_receipt.json', receipt)
                    command(SCP + [str(LOCAL / 'local_receipt.json'), HOST + ':' + REMOTE + '/local_receipt.json.partial'])
                    command(SSH + [f'mv {REMOTE}/local_receipt.json.partial {REMOTE}/local_receipt.json'])
                    command(SSH + [f'/root/miniconda3/bin/python {SERVER} --shutdown'], timeout=900)
                    status('shutdown_scheduled_after_all_checks')
                    # Scheduling +1 keeps the verification receipt readable before shutdown.
                    shutdown_receipt = remote_json(REMOTE + '/shutdown_verified.json')
                    save_json(LOCAL / 'shutdown_verified.json', shutdown_receipt)
                    return
            else:
                status('waiting_for_verified_training_and_inference')
            server_status = remote_json('/root/autodl-tmp/output/run03_model261108_inference/status.json')
            if server_status and server_status['stage'] == 'failed':
                status('server_needs_review', server=server_status)
            exhausted = [e for e in (2615, 15) if attempts[e] >= 3
                         and not (LOCAL / f'analysis_{e}_complete.json').exists()]
            if exhausted and (analysis is None or analysis.poll() is not None):
                status('analysis_needs_review', energies=exhausted)
        except Exception:
            status('retry', error=traceback.format_exc())
        time.sleep(60)


if __name__ == '__main__':
    main()
