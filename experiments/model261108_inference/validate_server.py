"""Test shutdown guards with isolated fixtures; never invoke a system shutdown."""
import json
from pathlib import Path
import tempfile
import numpy as np
import h5py
import server
from common import digest, record_file, save_json


with tempfile.TemporaryDirectory(prefix='inference_guard_test_') as directory:
    root = Path(directory)
    server.TRAIN, server.BACKUP, server.NAS = root / 'train', root / 'backup', root / 'nas'
    for path in (server.TRAIN, server.BACKUP, server.NAS):
        path.mkdir()
    calls = []
    server.subprocess.Popen = lambda *args, **kwargs: calls.append(args)
    (server.TRAIN / 'status.tsv').write_text('backup_verified')
    (server.BACKUP / 'BACKUP_VERIFIED').touch()
    save_json(server.BACKUP / 'history.json', {'train_loss': [1.] * 100, 'val_loss': [1.] * 100})
    save_json(server.BACKUP / 'config.json', {'T': 50, 'beta_1': 1e-4, 'beta_T': .5, 'cond_mode': 'step', 'scale_emb': True, 'film': True})
    for path in (server.TRAIN, server.BACKUP):
        (path / 'best_model.pt').write_bytes(b'fixture only')
    (server.BACKUP / 'SHA256SUMS').write_text(''.join(digest(server.BACKUP / name) + '  ' + name + '\n'
                                                for name in ('history.json', 'config.json', 'best_model.pt')))
    training = server.training_records()
    records = []
    for i in range(34):
        path = server.NAS / f'fixture_{i}.h5'
        path.write_bytes(b'fixture')
        records.append(record_file(path, server.NAS))
    manifest = {'complete': True, 'files': records, 'auxiliary_files': []}
    save_json(server.NAS / 'manifest.json', manifest)
    receipt = {'all_verified': False, 'analysis_complete': True, 'inference_files': records,
               'training_files': training, 'auxiliary_files': []}
    save_json(server.NAS / 'local_receipt.json', receipt)
    try:
        server.shutdown()
        raise RuntimeError('Incomplete receipt incorrectly permitted shutdown')
    except AssertionError:
        pass
    assert not calls
    receipt['all_verified'] = True
    save_json(server.NAS / 'local_receipt.json', receipt)
    (server.NAS / records[0]['path']).write_bytes(b'corrupt')
    try:
        server.shutdown()
        raise RuntimeError('Corrupt backup incorrectly permitted shutdown')
    except AssertionError:
        pass
    assert not calls
    (server.NAS / records[0]['path']).write_bytes(b'fixture')
    server.shutdown()
    assert calls == [(['/bin/bash', '-c', 'sleep 60; exec /bin/bash /usr/bin/shutdown'],)]
    path = root / 'output.h5'
    with h5py.File(path, 'w') as data:
        data.attrs.update(source_sha256='source', model_sha256='model')
        data['source_row'] = np.arange(2)
        data['waveforms_denoised'] = np.ones((2, 10000), dtype=np.float32)
    server.validate_output(path, 'source', 'model', 2)
    with h5py.File(path, 'a') as data:
        data['waveforms_denoised'][0, 0] = np.nan
    try:
        server.validate_output(path, 'source', 'model', 2)
        raise RuntimeError('Nonfinite output incorrectly accepted')
    except AssertionError:
        pass
print(json.dumps({'incomplete_receipt_blocked': True, 'corrupt_backup_blocked': True,
                  'valid_receipt_permitted_mock_only': True, 'nonfinite_output_blocked': True}))
