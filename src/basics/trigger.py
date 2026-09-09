"""Derivative-threshold (DT) trigger for bolometer waveforms.

Fully BLIND decision rule — mirrors Octopus's "derivative" trigger module
    reference/Octopus_src/src/trigger/triggermodules/Derivative.cpp
    reference/Octopus_src/src/octo/Buffer.cpp  (Buffer::ReadDerivative,
                                                 Buffer::IsAnEvent)
in the sense that matters: the decision never uses any ground-truth
information (e.g. when a pulse "should" occur). It scans the entire trace
with a causal rolling baseline, exactly like a continuous DAQ would, and can
fire zero, one, or many times.

What's used:
  - dv[i] = x[i] - x[i-1] — a RAW single-sample backward difference, no
    smoothing, not divided by sampling time (real anti-aliasing happens in
    the analog front-end, not here).
  - threshold(i) = mu_roll(i) + nsigma*rms_roll(i), where mu_roll/rms_roll
    are the mean/std of dv over the `window_n` samples immediately
    PRECEDING i (causal — never looks at future samples). Computed for the
    whole trace at once via cumulative sums (vectorized), which is
    numerically identical to Octopus's incremental streaming update —
    we just don't need an explicit per-sample state machine for this offline
    batch use case.
  - Trigger condition: n_above consecutive samples of dv all exceed
    threshold (Buffer::IsAnEvent's nAbove==fSzTrig check).
  - Dead time: after a trigger fires, suppress any new trigger for
    `dead_samples` (default = window_n, matching Octopus's "use buffer
    length if DeadTimeList <= 0").

What ground truth is NOT used for: nothing in this module. Any comparison to
a known pulse onset (truth-matching, to score whether a given trigger
"caught" a specific injected pulse) happens entirely downstream, in the
calling script — exactly like a real "pulser injection" trigger-efficiency
study: the trigger runs blind over the continuous stream; only the
bookkeeping that scores efficiency afterward gets to use the known
injection time.
"""

import numpy as np


def rolling_baseline_threshold(dv: np.ndarray, window_n: int,
                               nsigma: float = 4.0,
                               use_fixed_threshold: bool = False,
                               fixed_threshold: float = 10.0):
    """Causal rolling mean/std of `dv` over a trailing window of `window_n`
    samples, vectorized via cumulative sums.

    threshold[k] (k = 0 .. len(dv)-window_n) applies to dv[window_n + k],
    computed from the PRECEDING window dv[k : k+window_n] — i.e. it never
    looks at dv[window_n+k] itself or anything after it.

    Returns (mu, rms, threshold), each of length len(dv) - window_n.
    """
    dv = np.asarray(dv, dtype=float)
    N = len(dv)
    cumsum  = np.concatenate([[0.0], np.cumsum(dv)])
    cumsum2 = np.concatenate([[0.0], np.cumsum(dv * dv)])

    # win_sum[k] = sum(dv[k : k+window_n]), for k = 0 .. N-window_n-1
    # (so it covers the window immediately preceding dv[window_n+k]).
    win_sum  = cumsum[window_n:N] - cumsum[0:N - window_n]
    win_sum2 = cumsum2[window_n:N] - cumsum2[0:N - window_n]
    mu  = win_sum / window_n
    var = (win_sum2 - window_n * mu * mu) / max(window_n - 1, 1)
    var = np.clip(var, 0.0, None)
    rms = np.sqrt(var)

    if use_fixed_threshold:
        threshold = mu + fixed_threshold
    else:
        threshold = mu + nsigma * rms
    return mu, rms, threshold


def dt_trigger_blind(waveform: np.ndarray, fs: float, window_n: int,
                     n_above: int = 5, nsigma: float = 4.0,
                     use_fixed_threshold: bool = False, fixed_threshold: float = 10.0,
                     dead_samples: int = None):
    """Fully blind derivative trigger — scans the ENTIRE waveform, no
    ground-truth information used anywhere in the decision.

    Parameters
    ----------
    waveform     (N,) array, waveform samples [V].
    fs           Sample rate [Hz].
    window_n     Causal rolling-baseline window length [samples].
    n_above      Consecutive dv samples that must all exceed threshold.
    nsigma, use_fixed_threshold, fixed_threshold
                 Threshold definition (see rolling_baseline_threshold).
    dead_samples Minimum sample gap between two accepted triggers (debounce).
                 Defaults to `window_n` (Octopus's behavior when
                 DeadTimeList <= 0: dead time = buffer length).

    Returns
    -------
    trigger_idx  sorted int array — waveform sample index where each
                 accepted trigger's qualifying run BEGINS (Octopus reports
                 the run start, not the confirmation point: see
                 ProcessSignalEvent(c, s-NAbove)). Empty if none fired.
    mu, rms, threshold
                 Rolling-baseline arrays (aligned to dv[window_n:]), for
                 diagnostics/plotting.
    """
    if dead_samples is None:
        dead_samples = window_n

    x = np.asarray(waveform, dtype=float)
    dv = np.diff(x)   # dv[k] = x[k+1] - x[k]; dv[k] <-> original index k+1

    mu, rms, threshold = rolling_baseline_threshold(
        dv, window_n, nsigma, use_fixed_threshold, fixed_threshold)

    above = dv[window_n:] > threshold   # aligned: above[k] <-> dv[window_n+k]
    if not above.any():
        return np.array([], dtype=int), mu, rms, threshold

    padded = np.concatenate([[False], above, [False]])
    diffs  = np.diff(padded.astype(np.int8))
    starts = np.where(diffs == 1)[0]
    ends   = np.where(diffs == -1)[0]
    run_len = ends - starts
    qualifying = starts[run_len >= n_above]   # indices into `above`
    if len(qualifying) == 0:
        return np.array([], dtype=int), mu, rms, threshold

    # Convert to original waveform sample indices (run START):
    # `above[k]` <-> dv[window_n+k] <-> original index window_n+k+1.
    orig_idx = qualifying + window_n + 1

    # Dead-time debounce: keep the first candidate, skip subsequent ones
    # within `dead_samples` of the last kept trigger.
    kept = []
    last_kept = -dead_samples - 1
    for idx in orig_idx:
        if idx - last_kept >= dead_samples:
            kept.append(idx)
            last_kept = idx

    return np.array(kept, dtype=int), mu, rms, threshold


def dt_trigger_blind_batch(waveforms: np.ndarray, fs: float, window_n: int, **kw):
    """Loop wrapper of `dt_trigger_blind` over (N_events, N_samples).

    Returns a list of length N_events; each element is the `trigger_idx`
    array for that event (possibly empty, possibly with several entries).
    """
    return [dt_trigger_blind(waveforms[i], fs, window_n, **kw)[0]
           for i in range(waveforms.shape[0])]
