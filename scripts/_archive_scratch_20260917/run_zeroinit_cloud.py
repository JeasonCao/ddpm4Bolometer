"""
2026-09-21: production zero-init inference on AutoDL.

zero-init = deterministic DDIM (eta=0) started from x_T = 0 (the prior mean)
instead of torch.randn. Validated 2026-09-21 on 17 energies x 2 noise levels:
beats every existing method at every point, and unlike the others its
denoised/no-denoising FWHM ratio never exceeds 1.

Reads the inputs READ-ONLY and writes a separate, minimal output file per
energy containing ONLY `waveforms_denoised_zeroinit` plus identifying attrs.
Two reasons: the existing cloud results stay untouched, and the download
shrinks to ~166 MB/file instead of shipping the noisy waveforms back.
The local side merges the field into the local eval files afterwards.
Skips an output that already exists, so the job is restartable after a crash.
"""
import sys, os, time, csv
sys.path.insert(0, '/root/autodl-fs/ddpm4Bolometer')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np, torch, h5py
from datetime import datetime
from src.ddpm.unet import UNet1D
from src.ddpm.schedule import DiffusionSchedule
from src.ddpm.diffusion import GaussianDiffusion

MODEL_PATH = '/root/autodl-fs/best_model.pt'
# (input dir, output tag). Inputs are opened read-only and never modified.
# (input dir, output tag, sentinel). The 7.5 mV data is uploaded in parallel
# with the 1.382 mV inference, so that group waits for an explicit marker
# rather than risking a half-transferred file.
GROUPS = [
    ('/root/autodl-fs/eval_data_holdout_denoised/efficiency_1p382mV', 'efficiency_1p382mV', None),
    ('/root/autodl-fs/eval_upload/efficiency_7p5mV', 'efficiency_7p5mV',
     '/root/autodl-fs/eval_upload/UPLOAD_COMPLETE'),
]
WAIT_SENTINEL_MAX_S = 3 * 3600
OUT_ROOT = '/root/autodl-fs/zeroinit_out'
FIELD = 'waveforms_denoised_zeroinit'
BATCH = 256
TIMING = '/root/autodl-fs/timing_zeroinit.csv'


def load_model(device):
    schedule = DiffusionSchedule(T=50, beta_1=1e-4, beta_T=0.5).to(device)
    model = UNet1D(cond_mode='step').to(device)
    state = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    if isinstance(state, dict) and 'model_state_dict' in state:
        state = state['model_state_dict']
    model.load_state_dict(state)
    model.eval()
    return GaussianDiffusion(model, schedule, cond_mode='step'), schedule


@torch.no_grad()
def zero_init(diffusion, schedule, x_tilde):
    """Deterministic DDIM from x_T = 0, full T steps. x_tilde: (B,1,L) normalized."""
    T = schedule.T
    B, C, L = x_tilde.shape
    tau = torch.arange(T - 1, -1, -1, device=x_tilde.device)
    x_t = torch.zeros(B, 1, L, device=x_tilde.device)
    for i in range(len(tau)):
        tc = tau[i]
        cond = torch.full((B,), tc.item() + 1, device=x_tilde.device, dtype=torch.long)
        eps = diffusion._predict_noise(x_t, x_tilde, cond, None)
        x0 = (x_t - schedule.sqrt_one_minus_alpha_bar[tc] * eps) / schedule.sqrt_alpha_bar[tc]
        if i < len(tau) - 1:
            tn = tau[i + 1]
            x_t = schedule.sqrt_alpha_bar[tn] * x0 + \
                torch.sqrt(torch.clamp(1 - schedule.alpha_bar[tn], min=0.0)) * eps
        else:
            x_t = x0
    return x_t


def denoise_file(diffusion, schedule, noisy, device):
    N = noisy.shape[0]
    out = np.empty(noisy.shape, dtype=np.float32)
    for s in range(0, N, BATCH):
        e = min(s + BATCH, N)
        ch = noisy[s:e]
        sc = np.abs(ch).max(axis=1, keepdims=True)
        sc = np.where(sc == 0, 1.0, sc)
        x = torch.from_numpy((ch / sc).astype(np.float32)).unsqueeze(1).to(device)
        o = zero_init(diffusion, schedule, x).squeeze(1).cpu().numpy()
        out[s:e] = (o * sc).astype(np.float32)
        torch.cuda.empty_cache()
    return out


def log_timing(row):
    new = not os.path.exists(TIMING)
    with open(TIMING, 'a', newline='') as f:
        w = csv.writer(f)
        if new:
            w.writerow(['timestamp', 'gpu', 'group', 'energy_kev', 'n_events',
                        'batch', 'elapsed_s', 'events_per_s', 'ms_per_event'])
        w.writerow(row)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gpu = torch.cuda.get_device_name(0) if device.type == 'cuda' else 'cpu'
    print(f'Device: {device}  {gpu}', flush=True)
    diffusion, schedule = load_model(device)

    t_all = time.time()
    for group, tag, sentinel in GROUPS:
        if sentinel is not None:
            waited = 0
            while not os.path.exists(sentinel):
                if waited == 0:
                    print(f'\nwaiting for upload sentinel {sentinel} ...', flush=True)
                if waited > WAIT_SENTINEL_MAX_S:
                    print(f'!! sentinel never appeared after {waited/3600:.1f} h, skipping {tag}',
                          flush=True)
                    break
                time.sleep(60); waited += 60
            if not os.path.exists(sentinel):
                continue
            if waited:
                print(f'sentinel found after {waited/60:.0f} min, proceeding with {tag}', flush=True)
        if not os.path.isdir(group):
            print(f'!! missing input dir, skipping: {group}', flush=True); continue
        out_dir = os.path.join(OUT_ROOT, tag)
        os.makedirs(out_dir, exist_ok=True)
        files = sorted(f for f in os.listdir(group) if f.endswith('.h5'))
        print(f'\n===== {tag} : {len(files)} files =====', flush=True)
        for fn in files:
            out_path = os.path.join(out_dir, fn)
            if os.path.exists(out_path):
                print(f'  {fn:34s} output exists, skip', flush=True)
                continue
            with h5py.File(os.path.join(group, fn), 'r') as f:      # READ-ONLY
                noisy = f['waveforms_noisy'][:]
                E = float(f.attrs['energy_kev'])
            t0 = time.time()
            out = denoise_file(diffusion, schedule, noisy.astype(np.float64), device)
            dt = time.time() - t0
            tmp = out_path + '.part'
            with h5py.File(tmp, 'w') as g:                           # minimal output
                g.attrs['energy_kev'] = E
                g.attrs['n_events'] = len(noisy)
                g.attrs['zeroinit_xT_scale'] = 0.0
                g.attrs['zeroinit_timestamp'] = datetime.now().strftime('%Y%m%d_%H%M%S')
                g.attrs['source_group'] = tag
                g.create_dataset(FIELD, data=out, compression='gzip')
            os.replace(tmp, out_path)                                # atomic: no half files
            n = len(noisy)
            log_timing([datetime.now().isoformat(timespec='seconds'), gpu, tag, E, n, BATCH,
                        f'{dt:.1f}', f'{n/dt:.2f}', f'{dt/n*1000:.1f}'])
            print(f'  {fn:34s} E={E:7.0f} keV  N={n}  {dt:7.1f}s  '
                  f'{n/dt:6.2f} ev/s  {dt/n*1000:6.1f} ms/ev', flush=True)

    print(f'\nTotal: {(time.time()-t_all)/3600:.2f} h', flush=True)
    print('ZEROINIT_ALL_DONE', flush=True)


if __name__ == '__main__':
    main()
