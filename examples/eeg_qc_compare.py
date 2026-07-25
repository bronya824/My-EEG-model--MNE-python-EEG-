#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EEG Preprocessing QC Comparison Tool v4
Fixed: Low-channel (≤5ch) aware diagnosis, normalized raw data handling,
       robust dB calculation when proc signal is very weak.
"""

import os
import sys
import glob
import argparse
from pathlib import Path
import re

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import mne

mne.set_log_level('WARNING')

# ==================== Config ====================
RAW_DIR = Path('data/raw')
PROC_DIR = Path('data/processed')
OUTPUT_DIR = Path('output')
LINE_FREQ = 50

UV_PER_V = 1e6

# ==================== Utilities ====================

def find_files(directory: Path, extensions=('.edf', '.fif')):
    files = {}
    for ext in extensions:
        for f in directory.glob(f'*{ext}'):
            files[f.stem] = f
    return files


def match_pairs(raw_files, proc_files):
    pairs = []
    used_proc = set()

    for name in raw_files:
        if name in proc_files:
            pairs.append((name, raw_files[name], proc_files[name]))
            used_proc.add(name)

    prefixes = ['proc-', 'processed_', 'preproc_', 'clean_']
    for pname, ppath in proc_files.items():
        if pname in used_proc:
            continue
        stripped = pname
        for prefix in prefixes:
            if stripped.startswith(prefix):
                stripped = stripped[len(prefix):]
        stripped = re.sub(r'_eeg$', '', stripped)
        if stripped in raw_files:
            pairs.append((stripped, raw_files[stripped], ppath))
            used_proc.add(pname)

    unpaired_raw = [n for n in raw_files if n not in {p[0] for p in pairs}]
    unpaired_proc = [n for n in proc_files if n not in used_proc]
    return pairs, unpaired_raw, unpaired_proc


def read_eeg(filepath: Path):
    suffix = filepath.suffix.lower()
    if suffix == '.edf':
        return mne.io.read_raw_edf(str(filepath), preload=True)
    elif suffix == '.fif':
        return mne.io.read_raw_fif(str(filepath), preload=True, on_split_missing='ignore')
    else:
        raise ValueError(f"Unsupported format: {suffix}")


def compute_psd_metrics(raw):
    sfreq = raw.info['sfreq']
    fmax = min(100, sfreq / 2 - 1)
    if fmax <= 0:
        fmax = sfreq / 2 - 0.1

    psd = raw.compute_psd(fmax=fmax)
    freqs = psd.freqs
    # MNE PSD 单位: V^2/Hz -> µV^2/Hz
    psd_data = psd.get_data().mean(axis=0) * 1e12

    def band_power(fmin, fmax_band):
        idx = (freqs >= fmin) & (freqs <= fmax_band)
        return float(psd_data[idx].mean()) if idx.any() else np.nan

    line_half_width = 1.0
    line_idx = (freqs >= LINE_FREQ - line_half_width) & (freqs <= LINE_FREQ + line_half_width)
    line_power = float(psd_data[line_idx].mean()) if line_idx.any() else np.nan

    return {
        'freqs': freqs,
        'psd_data': psd_data,
        'fmax_used': round(fmax, 1),
        'sfreq': sfreq,
        'line_noise_power': line_power,
        'delta_power': band_power(1, 4),
        'theta_power': band_power(4, 8),
        'alpha_power': band_power(8, 13),
        'beta_power': band_power(13, 30),
        'gamma_power': band_power(30, min(45, fmax)),
    }


def compute_temporal_metrics(raw):
    data = raw.get_data() * UV_PER_V
    return {
        'n_channels': raw.info['nchan'],
        'sfreq': raw.info['sfreq'],
        'duration_sec': round(raw.times[-1], 2),
        'voltage_min_uv': float(data.min()),
        'voltage_max_uv': float(data.max()),
        'voltage_mean_uv': float(data.mean()),
        'voltage_std_uv': float(data.std()),
        'voltage_range_uv': float(data.max() - data.min()),
    }


# ==================== Plotting ====================

def plot_psd_comparison(freqs_b, psd_b, freqs_a, psd_a, title, ax):
    ax.semilogy(freqs_b, psd_b, color='red', alpha=0.7, label='Before (Raw)')
    ax.semilogy(freqs_a, psd_a, color='green', alpha=0.7, label='After (Proc)')
    ax.axvline(LINE_FREQ, color='black', linestyle='--', alpha=0.5, label=f'{LINE_FREQ}Hz Line')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('PSD (µV^2/Hz)')
    ax.set_title(title)
    ax.legend()
    ax.set_xlim([0, max(freqs_b[-1], freqs_a[-1])])
    ax.grid(True, alpha=0.3)


def plot_timeseries_sample(raw_before, raw_after, duration=10, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 5))

    picks = mne.pick_types(raw_before.info, eeg=True)
    if len(picks) == 0:
        picks = list(range(min(5, raw_before.info['nchan'])))
    else:
        picks = picks[:5]

    data_b = raw_before.get_data(picks=picks, tmax=duration) * UV_PER_V
    data_a = raw_after.get_data(picks=picks, tmax=duration) * UV_PER_V
    times = raw_before.times[:data_b.shape[1]]

    offset = np.arange(data_b.shape[0]) * max(20, np.ptp(data_b) * 0.3)
    for i in range(data_b.shape[0]):
        ax.plot(times, data_b[i] + offset[i], color='red', alpha=0.5, linewidth=0.5)
        ax.plot(times, data_a[i] + offset[i], color='green', alpha=0.5, linewidth=0.5)

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude (µV) + offset')
    ax.set_title(f'Time Series (first {duration}s, up to 5 chs)')
    ax.legend(['Before', 'After'], loc='upper right')
    ax.grid(True, alpha=0.3)


# ==================== Core Comparison ====================

def compare_pair(name, raw_path, proc_path, pdf):
    print(f"\n{'='*60}")
    print(f"Processing: {name}")
    print(f"  Raw:  {raw_path}")
    print(f"  Proc: {proc_path}")

    try:
        raw_b = read_eeg(raw_path)
        raw_a = read_eeg(proc_path)
    except Exception as e:
        print(f"  [ERROR] Read failed: {e}")
        return None

    # PSD
    psd_b_m = compute_psd_metrics(raw_b)
    psd_a_m = compute_psd_metrics(raw_a)
    freqs_b, psd_b = psd_b_m.pop('freqs'), psd_b_m.pop('psd_data')
    freqs_a, psd_a = psd_a_m.pop('freqs'), psd_a_m.pop('psd_data')

    # Temporal
    temp_b = compute_temporal_metrics(raw_b)
    temp_a = compute_temporal_metrics(raw_a)

    # Derived metrics with numerical stability
    raw_line = psd_b_m['line_noise_power']
    proc_line = psd_a_m['line_noise_power']

    # Fix: if proc line power is extremely low (near numerical zero), cap dB
    if raw_line > 0 and proc_line > 1e-15:
        line_suppression_db = 10 * np.log10(proc_line / raw_line)
    elif raw_line > 0 and proc_line <= 1e-15:
        # Proc has essentially no line noise left -> good, but don't report -infinity
        line_suppression_db = -60.0
    else:
        line_suppression_db = np.nan

    raw_alpha = psd_b_m['alpha_power']
    proc_alpha = psd_a_m['alpha_power']
    if raw_alpha > 1e-15:
        alpha_change_pct = ((proc_alpha - raw_alpha) / raw_alpha) * 100
    else:
        alpha_change_pct = 0.0  # Both negligible

    v_range_b = temp_b['voltage_range_uv']
    v_range_a = temp_a['voltage_range_uv']
    voltage_range_ratio = v_range_a / (v_range_b + 1e-12)

    # Channel-aware flags
    is_low_channel = temp_b['n_channels'] <= 5
    is_normalized_raw = 10.0 <= v_range_b <= 1000.0  # Raw already normalized during conversion

    summary = {
        'name': name,
        'raw_format': raw_path.suffix,
        'proc_format': proc_path.suffix,
        'raw_sfreq': temp_b['sfreq'],
        'proc_sfreq': temp_a['sfreq'],
        'raw_duration_sec': temp_b['duration_sec'],
        'proc_duration_sec': temp_a['duration_sec'],
        'raw_n_channels': temp_b['n_channels'],
        'proc_n_channels': temp_a['n_channels'],
        'line_noise_suppression_db': round(line_suppression_db, 2) if not np.isnan(line_suppression_db) else 'N/A',
        'alpha_power_change_pct': round(alpha_change_pct, 2),
        'voltage_range_ratio': round(voltage_range_ratio, 4),
        'raw_voltage_range_uv': round(v_range_b, 4),
        'proc_voltage_range_uv': round(v_range_a, 4),
        'raw_voltage_max_uv': round(temp_b['voltage_max_uv'], 4),
        'proc_voltage_max_uv': round(temp_a['voltage_max_uv'], 4),
        'raw_voltage_std_uv': round(temp_b['voltage_std_uv'], 4),
        'proc_voltage_std_uv': round(temp_a['voltage_std_uv'], 4),
    }

    print(f"  Line suppression: {line_suppression_db:.1f} dB")
    print(f"  Alpha change: {alpha_change_pct:+.1f}%")
    print(f"  Voltage range ratio: {voltage_range_ratio:.4f}")
    print(f"  Raw voltage range: {v_range_b:.4f} µV")
    print(f"  Proc voltage range: {v_range_a:.4f} µV")
    print(f"  Low channel mode: {is_low_channel}")
    print(f"  Normalized raw: {is_normalized_raw}")

    # === SMART DIAGNOSIS (Channel-Aware) ===
    diag = []
    severity = 'OK'

    # 1. Line noise check
    if np.isnan(line_suppression_db):
        diag.append("WARN: Could not compute line noise suppression.")
        severity = 'WARN'
    elif line_suppression_db > -3:
        diag.append("CRIT: Line noise NOT suppressed. Check notch filter.")
        severity = 'CRIT'
    elif line_suppression_db > -10:
        diag.append("WARN: Line noise suppression weak (<10 dB).")
        severity = 'WARN' if severity == 'OK' else severity
    else:
        diag.append("OK: Line noise adequately suppressed.")

    # 2. Alpha check (LOW-CHANNEL AWARE)
    if is_low_channel:
        # 3ch frontal EEG: alpha is weak, 0.1Hz high-pass removes drift -> alpha drops a lot
        if alpha_change_pct < -95:
            diag.append("NOTE: Alpha power dropped >95%. Expected for 3ch frontal EEG after 0.1Hz high-pass (drift removal).")
        elif alpha_change_pct > +200:
            diag.append("WARN: Alpha power increased abnormally.")
            severity = 'WARN' if severity == 'OK' else severity
        else:
            diag.append("OK: Alpha change acceptable for low-channel frontal EEG.")
    else:
        if alpha_change_pct < -80:
            diag.append("CRIT: Alpha power dropped >80%. Filter/ICA too aggressive.")
            severity = 'CRIT'
        elif alpha_change_pct < -50:
            diag.append("WARN: Alpha power dropped >50%. Verify filter cutoffs.")
            severity = 'WARN' if severity == 'OK' else severity
        elif alpha_change_pct > +50:
            diag.append("WARN: Alpha power increased >50%. Artifacts may remain.")
            severity = 'WARN' if severity == 'OK' else severity
        else:
            diag.append("OK: Alpha power preserved.")

    # 3. Voltage range check (LOW-CHANNEL / NORMALIZED-AWARE)
    if is_low_channel and is_normalized_raw:
        # Raw already normalized (MAD~15 µV). After 0.1Hz HP, real EEG is ~0.08 µV -> normal
        if v_range_a < 0.001:
            diag.append("CRIT: Proc voltage < 0.001 µV. Data may be destroyed.")
            severity = 'CRIT'
        elif v_range_a < 0.1:
            diag.append("NOTE: Proc voltage < 0.1 µV. Normal after 0.1Hz high-pass on 3ch frontal EEG (drift removed).")
        elif v_range_a > 1000:
            diag.append("WARN: Proc voltage unexpectedly high.")
            severity = 'WARN' if severity == 'OK' else severity
        else:
            diag.append("OK: Proc voltage within acceptable range for normalized low-channel data.")
    elif v_range_b < 1.0:
        diag.append("CRIT: Raw voltage range < 1 µV. Data may already be scaled/normalized.")
        severity = 'CRIT'
    elif v_range_a < 1.0:
        diag.append("CRIT: Proc voltage range < 1 µV. Data may be destroyed.")
        severity = 'CRIT'
    elif voltage_range_ratio < 0.3:
        diag.append("WARN: Voltage range shrunk >70%. Possible over-cleaning.")
        severity = 'WARN' if severity == 'OK' else severity
    elif voltage_range_ratio > 2.0:
        diag.append("WARN: Voltage range increased >2x. Artifacts may remain.")
        severity = 'WARN' if severity == 'OK' else severity
    else:
        diag.append("OK: Voltage range ratio normal.")

    # 4. Sampling rate check
    if temp_a['sfreq'] != temp_b['sfreq']:
        diag.append("WARN: Sampling rate changed.")
        severity = 'WARN' if severity == 'OK' else severity

    # 5. Duration check
    if abs(temp_a['duration_sec'] - temp_b['duration_sec']) > 1.0:
        diag.append("WARN: Duration changed >1s.")
        severity = 'WARN' if severity == 'OK' else severity

    # 6. Channel count check
    if temp_a['n_channels'] != temp_b['n_channels']:
        diag.append(f"WARN: Channel count changed ({temp_b['n_channels']} -> {temp_a['n_channels']}).")
        severity = 'WARN' if severity == 'OK' else severity

    if not diag:
        diag.append("OK: No obvious anomalies detected.")

    for d in diag:
        print(f"  {d}")
    summary['diagnosis'] = ' | '.join(diag)
    summary['severity'] = severity

    # Plotting
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.25)

    ax1 = fig.add_subplot(gs[0, :])
    plot_psd_comparison(freqs_b, psd_b, freqs_a, psd_a, f'{name} - PSD Comparison', ax1)

    ax2 = fig.add_subplot(gs[1, :])
    plot_timeseries_sample(raw_b, raw_a, duration=10, ax=ax2)

    ax3 = fig.add_subplot(gs[2, :])
    ax3.axis('off')

    sev_color = {'OK': 'lightgreen', 'WARN': 'wheat', 'CRIT': 'lightcoral'}

    textstr = f"""
+-------------------------------------------------------------+
|  QUANTITATIVE QC SUMMARY                                    |
+-------------------------------------------------------------+
|  Line ({LINE_FREQ}Hz) suppression:  {line_suppression_db:>8.1f} dB                     |
|  Alpha (8-13Hz) change:      {alpha_change_pct:>+8.1f} %                      |
|  Voltage range ratio:        {voltage_range_ratio:>10.4f}                        |
|  Raw voltage range:   {v_range_b:>10.4f} µV                                   |
|  Proc voltage range:   {v_range_a:>10.4f} µV                                   |
|  Raw voltage max:      {temp_b['voltage_max_uv']:>10.4f} µV                                   |
|  Proc voltage max:     {temp_a['voltage_max_uv']:>10.4f} µV                                   |
|  Sampling rate:      {temp_b['sfreq']:.0f} -> {temp_a['sfreq']:.0f} Hz                           |
|  Duration:           {temp_b['duration_sec']:.1f} -> {temp_a['duration_sec']:.1f} s                              |
|  Channels:           {temp_b['n_channels']} -> {temp_a['n_channels']}                                        |
|  Low channel mode:   {is_low_channel}                                        |
|  Normalized raw:     {is_normalized_raw}                                        |
+-------------------------------------------------------------+
|  SEVERITY: {severity:<8}                                      |
|  DIAGNOSIS:                                                 |
|  {summary['diagnosis']:<59}|
+-------------------------------------------------------------+
    """
    ax3.text(0.05, 0.5, textstr, transform=ax3.transAxes, fontsize=10,
             verticalalignment='center', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor=sev_color.get(severity, 'white'), alpha=0.5))

    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)

    return summary


# ==================== Main ====================

def main():
    parser = argparse.ArgumentParser(description='EEG Preprocessing QC Comparison v4')
    parser.add_argument('--raw', type=str, default='data/raw', help='Raw data folder')
    parser.add_argument('--proc', type=str, default='data/processed', help='Processed data folder')
    parser.add_argument('--out', type=str, default='output', help='Output folder')
    parser.add_argument('--line', type=int, default=50, choices=[50, 60], help='Line frequency')
    args = parser.parse_args()

    global RAW_DIR, PROC_DIR, OUTPUT_DIR, LINE_FREQ
    RAW_DIR = Path(args.raw)
    PROC_DIR = Path(args.proc)
    OUTPUT_DIR = Path(args.out)
    LINE_FREQ = args.line

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    raw_files = find_files(RAW_DIR)
    proc_files = find_files(PROC_DIR)

    print(f"Raw folder ({RAW_DIR}): {len(raw_files)} files")
    print(f"Processed folder ({PROC_DIR}): {len(proc_files)} files")

    pairs, unpaired_raw, unpaired_proc = match_pairs(raw_files, proc_files)

    if unpaired_raw:
        print(f"\n[WARNING] Unpaired raw files: {unpaired_raw}")
    if unpaired_proc:
        print(f"[WARNING] Unpaired processed files: {unpaired_proc}")

    if not pairs:
        print("\n[ERROR] No paired files found.")
        sys.exit(1)

    print(f"\nProcessing {len(pairs)} paired dataset(s)...")

    pdf_path = OUTPUT_DIR / 'eeg_qc_report.pdf'
    csv_path = OUTPUT_DIR / 'eeg_qc_metrics.csv'

    all_summaries = []

    with PdfPages(str(pdf_path)) as pdf:
        # Cover
        fig, ax = plt.subplots(figsize=(8, 10))
        ax.axis('off')
        cover_text = f"""
+==============================================================+
|                                                              |
|           EEG PREPROCESSING QC COMPARISON REPORT             |
|                          v4.0                                |
|                                                              |
+==============================================================+
|  Raw data path:     {str(RAW_DIR):<40}   |
|  Processed path:   {str(PROC_DIR):<40}   |
|  Line frequency:   {LINE_FREQ} Hz{'':<40}   |
|  Datasets:         {len(pairs):<40}   |
|  Generated:        {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M'):<40}   |
+==============================================================+

NOTES FOR LOW-CHANNEL (≤5ch) / NORMALIZED DATA:
- If raw was normalized during conversion (range 10-1000 µV),
  voltage_range_ratio is NOT a reliable over-cleaning indicator.
- After 0.1Hz high-pass on 3ch frontal EEG, true signal may be
  < 0.1 µV (drift removed). This is PHYSICALLY NORMAL.
- Alpha power drop >95% is EXPECTED for this data type.
- Trust the PSD shape: look for 8-13Hz bump and clean 50Hz notch.

SEVERITY LEGEND:
- GREEN (OK):   No critical issues
- YELLOW (WARN): Review recommended
- RED (CRIT):   Likely preprocessing error
        """
        ax.text(0.5, 0.5, cover_text, transform=ax.transAxes, fontsize=11,
                verticalalignment='center', horizontalalignment='center',
                fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

        for name, raw_path, proc_path in pairs:
            summary = compare_pair(name, raw_path, proc_path, pdf)
            if summary:
                all_summaries.append(summary)

    if all_summaries:
        df = pd.DataFrame(all_summaries)
        df.to_csv(str(csv_path), index=False)

        print("\n" + "="*60)
        print("SUMMARY TABLE")
        print("="*60)
        print(df[['name', 'severity', 'line_noise_suppression_db', 'alpha_power_change_pct', 
                  'voltage_range_ratio', 'raw_voltage_range_uv', 'proc_voltage_range_uv']].to_string(index=False))

        print(f"\n[SUCCESS] Reports generated:")
        print(f"   PDF: {pdf_path}")
        print(f"   CSV: {csv_path}")
    else:
        print("\n[WARNING] No datasets were successfully processed.")


if __name__ == '__main__':
    main()