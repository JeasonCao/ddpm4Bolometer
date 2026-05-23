"""
easytrigger.py — Direct Python port of easyTrigger.jl.

Fixed-baseline trigger: compute mean and RMS from the first N samples,
set threshold at mean ± multiplier * RMS, scan for crossings.

Public API:
    calculate_baseline_metrics, find_trigger_points,
    extract_signal_segment, process_triggered_signals
"""

import numpy as np
from .convert import process_adc_file


def calculate_baseline_metrics(voltages, baseline_period=1000):
    """Calculate baseline mean and RMS from the first `baseline_period` samples."""
    actual_period = min(baseline_period, len(voltages))
    baseline_data = voltages[:actual_period]
    mean_val = np.mean(baseline_data)
    rms = np.sqrt(np.mean((baseline_data - mean_val) ** 2))
    return float(mean_val), float(rms)


def find_trigger_points(voltages, threshold_upper, threshold_lower, fs,
                        pre_trigger_sec, post_trigger_sec, num_triggers=10):
    """Find trigger points that exceed the threshold, ensuring non-overlapping events."""
    trigger_indices = []
    marked_times = np.zeros(len(voltages), dtype=bool)

    pre_trigger_samples = round(pre_trigger_sec * fs)
    post_trigger_samples = round(post_trigger_sec * fs)

    for i in range(len(voltages)):
        if marked_times[i]:
            continue
        if voltages[i] > threshold_lower and voltages[i] < threshold_upper:
            continue

        trigger_indices.append(i)

        start_index = max(0, i - pre_trigger_samples)
        end_index = min(len(voltages), i + post_trigger_samples)
        marked_times[start_index:end_index] = True

        if len(trigger_indices) >= num_triggers:
            break

    return trigger_indices


def extract_signal_segment(times, voltages, trigger_index,
                           pre_trigger_sec, post_trigger_sec, fs):
    """Extract signal segment around a trigger index."""
    pre_trigger_samples = round(pre_trigger_sec * fs)
    post_trigger_samples = round(post_trigger_sec * fs)

    start_index = max(0, trigger_index - pre_trigger_samples)
    end_index = min(len(voltages), trigger_index + post_trigger_samples)

    return times[start_index:end_index], voltages[start_index:end_index]


def process_triggered_signals(filename, max_samples=None,
                              threshold_multiplier=9,
                              pre_trigger_sec=0.03, post_trigger_sec=0.07,
                              num_triggers=50):
    """Process a file: compute baseline, find triggers, return indices and data."""
    times, voltages, header = process_adc_file(filename, max_samples=max_samples)

    mean_val, rms = calculate_baseline_metrics(voltages)
    threshold_upper = mean_val + threshold_multiplier * rms
    threshold_lower = mean_val - threshold_multiplier * rms

    fs = header["fs"]

    trigger_indices = find_trigger_points(
        voltages, threshold_upper, threshold_lower, fs,
        pre_trigger_sec, post_trigger_sec, num_triggers,
    )

    if not trigger_indices:
        return None

    return trigger_indices, times, voltages
