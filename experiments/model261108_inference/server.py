"""Wait for verified training and inputs, then publish independent inference files."""
import argparse
import fcntl
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

from common import digest, read_json, record_file, save_json, verify_record

HERE = Path(__file__).resolve().parent
CODE = HERE / 'code'
sys.path.insert(0, str(CODE))
TRAIN = Path('/root/autodl-tmp/output/run03_model261108')
BACKUP = Path('/root/autodl-fs/output/run03_model261108')
INPUT = Path('/root/autodl-fs/simu_data/eval_data_holdout')
OUTPUT = Path('/root/autodl-tmp/output/run03_model261108_inference')
NAS = Path('/root/autodl-fs/output/run03_model261108_inference')


def status(stage, **details):
    value = {'stage': stage, 'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), **details}
    save_json(OUTPUT / 'status.json', value)
    print(value, flush=True)


def training_records():
    assert (BACKUP / 'BACKUP_VERIFIED').exists()
    assert 'backup_verified' in (TRAIN / 'status.tsv').read_text()
    history = read_json(BACKUP / 'history.json')
    for key in ('train_loss', 'val_loss'):
        assert len(history[key]) == 100 and all(map(math.isfinite, history[key]))
    config = read_json(BACKUP / 'config.json')
    for key, expected in {'T': 50, 'beta_1': 1e-4, 'beta_T': 0.5, 'cond_mode': 'step', 'scale_emb': True, 'film': True}.items():
        assert config[key] == expected, (key, config.get(key))
    records = []
    for line in (BACKUP / 'SHA256SUMS').read_text().splitlines():
        expected, name = line.split(maxsplit=1)
        name = name.lstrip('*')
        assert '/' not in name and '\\' not in name
        path = BACKUP / name
        assert digest(path) == expected, name
        records.append(record_file(path, BACKUP))
    assert digest(TRAIN / 'best_model.pt') == digest(BACKUP / 'best_model.pt')
    records += [record_file(BACKUP / name, BACKUP) for name in ('SHA256SUMS', 'BACKUP_VERIFIED')]
    return records


def input_records():
    report = read_json(INPUT / 'upload_verification_20261112.json')
    assert report['all_sha256_match'] is True
    records = report['files']
    assert len(records) == 34
    assert len({(r['group'], r['name']) for r in records}) == 34
    for r in records:
        assert r['shape'] == [5000, 10000]
        path = INPUT / r['group'] / r['name']
        assert path.stat().st_size == r['bytes'] and digest(path) == r['sha256'], str(path)
    priority = {('efficiency_1p382mV', 2615): 0, ('efficiency_1p382mV', 15): 1}
    return sorted(records, key=lambda r: (priority.get((r['group'], int(r['energy_kev'])), 2), -r['energy_kev'], r['group']))


def validate_output(path, source_sha, model_sha, count=5000):
    import h5py
    import numpy as np
    with h5py.File(path, 'r') as data:
        assert data.attrs['source_sha256'] == source_sha
        assert data.attrs['model_sha256'] == model_sha
        wave = data['waveforms_denoised']
        assert wave.shape == (count, 10000) and wave.dtype == np.float32
        assert np.array_equal(data['source_row'][:], np.arange(count))
        for start in range(0, count, 128):
            assert np.isfinite(wave[start:start + 128]).all()


def infer(diffusion, source, target, model_sha, count=None):
    import h5py
    import numpy as np
    import torch
    from scripts.run_batch_inference import denoise_batch
    source_path = INPUT / source['group'] / source['name']
    with h5py.File(source_path, 'r') as data:
        noisy = data['waveforms_noisy'][:count]
    start = time.monotonic()
    wave = denoise_batch(diffusion, noisy, torch.device('cuda'), batch_size=16,
                         shots=1, sampler='ddpm', seed=42, use_scale=True, no_noise=True)
    assert np.isfinite(wave).all()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix('.h5.partial')
    with h5py.File(temporary, 'w') as data:
        data.create_dataset('waveforms_denoised', data=wave, compression='gzip')
        data.create_dataset('source_row', data=np.arange(len(wave)))
        data.attrs.update(source_sha256=source['sha256'], source_file=str(source_path),
                          model_sha256=model_sha, model_path=str(BACKUP / 'best_model.pt'),
                          energy_kev=source['energy_kev'], f_sample=1000., n_events=len(wave),
                          sampler='ddpm', no_noise=True, shots=1, T=50, beta_1=1e-4,
                          beta_T=0.5, cond_mode='step', scale_emb=True, film=True,
                          precision='float32', seed=42, batch_size=16,
                          elapsed_seconds=time.monotonic() - start)
    validate_output(temporary, source['sha256'], model_sha, len(wave))
    temporary.replace(target)


def publish(path):
    relative = path.relative_to(OUTPUT)
    destination = NAS / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    expected = record_file(path, OUTPUT)
    if not verify_record(NAS, expected):
        temporary = destination.with_suffix(destination.suffix + '.partial')
        shutil.copyfile(path, temporary)
        assert digest(temporary) == expected['sha256']
        temporary.replace(destination)
    return expected


def run():
    import torch
    from scripts.run_batch_inference import load_model
    OUTPUT.mkdir(parents=True, exist_ok=True)
    lock = (OUTPUT / 'worker.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    while not ((BACKUP / 'BACKUP_VERIFIED').exists() and (INPUT / 'upload_verification_20261112.json').exists()
               and (TRAIN / 'status.tsv').exists() and 'backup_verified' in (TRAIN / 'status.tsv').read_text()):
        status('waiting_for_training_and_upload')
        time.sleep(60)
    status('verifying_training_and_inputs')
    train_files = training_records()
    sources = input_records()
    NAS.mkdir(parents=True, exist_ok=True)
    save_json(NAS / 'training_manifest.json', {'files': train_files})
    assert torch.cuda.is_available()
    torch.set_num_threads(4)
    model_sha = digest(BACKUP / 'best_model.pt')
    diffusion = load_model(str(BACKUP / 'best_model.pt'), torch.device('cuda'), scale_emb=True, film=True)
    # Both noise groups pass the actual-model smoke check before production.
    smoke_sources = [next(r for r in sources if r['group'] == group and r['energy_kev'] == 2615)
                     for group in ('efficiency_1p382mV', 'efficiency_7p5mV')]
    for source in smoke_sources:
        smoke_path = OUTPUT / 'smoke' / source['group'] / source['name']
        infer(diffusion, source, smoke_path, model_sha, count=2)
        import h5py
        from scripts.run_fitting import _fit_one
        with h5py.File(INPUT / source['group'] / source['name']) as raw, h5py.File(smoke_path) as smoke:
            template = raw['clean_template'][:]
            for waveform in smoke['waveforms_denoised']:
                fit = _fit_one((waveform, 'denoised', 10., 1e-5, 9.9, template))
                assert fit['success_denoised'] and math.isfinite(fit['amp_denoised']), 'Actual-model smoke fit failed'
    completed = []
    for source in sources:
        relative = source['group'] + '/' + source['name']
        path = OUTPUT / relative
        status('inference', file=relative, completed=len(completed), total=34)
        if path.exists():
            validate_output(path, source['sha256'], model_sha)
        else:
            infer(diffusion, source, path, model_sha)
        record = publish(path)
        record.update(source_sha256=source['sha256'], model_sha256=model_sha, energy_kev=source['energy_kev'])
        completed.append(record)
        save_json(NAS / 'manifest.json', {'complete': False, 'files': completed})
    assert all(verify_record(NAS, r) for r in completed)
    status('complete', completed=34, total=34)
    auxiliary = [publish(OUTPUT / 'status.json')]
    for name in ('server.py', 'common.py', 'client.py', 'analyze.py', 'code.zip'):
        shutil.copyfile(HERE / name, OUTPUT / name)
        auxiliary.append(publish(OUTPUT / name))
    shutil.copyfile(BACKUP / 'config.json', OUTPUT / 'training_config.json')
    auxiliary.append(publish(OUTPUT / 'training_config.json'))
    shutil.copyfile(HERE / 'server.log', OUTPUT / 'inference.log')
    auxiliary.append(publish(OUTPUT / 'inference.log'))
    save_json(NAS / 'manifest.json', {'complete': True, 'files': completed, 'auxiliary_files': auxiliary})


def shutdown():
    manifest = read_json(NAS / 'manifest.json')
    receipt = read_json(NAS / 'local_receipt.json')
    assert manifest['complete'] and len(manifest['files']) == 34
    assert receipt['all_verified'] and receipt['analysis_complete']
    assert receipt['inference_files'] == manifest['files']
    assert all(verify_record(NAS, r) for r in manifest['files'])
    assert receipt['auxiliary_files'] == manifest['auxiliary_files']
    assert all(verify_record(NAS, r) for r in manifest['auxiliary_files'])
    assert receipt['training_files'] == training_records()
    save_json(NAS / 'shutdown_verified.json', {'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'receipt_sha256': digest(NAS / 'local_receipt.json')})
    # AutoDL's shutdown is a shell file without a shebang and ignores CLI flags.
    # Delay in a detached shell so the client can fetch the verification receipt.
    subprocess.Popen(['/bin/bash', '-c', 'sleep 60; exec /bin/bash /usr/bin/shutdown'],
                     stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                     stderr=subprocess.DEVNULL, start_new_session=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--shutdown', action='store_true')
    args = parser.parse_args()
    if args.shutdown:
        shutdown()
    else:
        for attempt in range(1, 4):
            try:
                run()
                break
            except Exception:
                status('failed', attempt=attempt, error=traceback.format_exc())
                if attempt == 3:
                    raise
                time.sleep(60)
