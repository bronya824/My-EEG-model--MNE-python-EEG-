# eeg_proc/processing/artifact_detection.py

import mne
import numpy as np
import logging

logger = logging.getLogger('eeg_pipeline')


def compute_bad_channels(raw):
    """检测坏通道（幅度自适应 + 通道数感知）"""
    try:
        if hasattr(mne.preprocessing, 'find_bad_channels'):
            bads, _ = mne.preprocessing.find_bad_channels(raw)
            return bads

        data = raw.get_data()
        n_channels = data.shape[0]
        ptp = np.ptp(data, axis=1)

        # ========== 新增：幅度自适应归一化（防御性） ==========
        median_ptp = np.median(ptp)
        if median_ptp > 1e5:
            scale = median_ptp / 100.0
            ptp_norm = ptp / scale
            median = np.median(ptp_norm)
            mad = np.median(np.abs(ptp_norm - median))
            z_scores = (ptp_norm - median) / (mad * 1.4826 + 1e-12)
        else:
            median = np.median(ptp)
            mad = np.median(np.abs(ptp - median))
            if mad == 0:
                return []
            z_scores = (ptp - median) / (mad * 1.4826)
        # ========== 新增：低通道模式提高阈值 ==========
        threshold = 5.0 if n_channels <= 5 else 3.0
        bad_idx = np.where(np.abs(z_scores) > threshold)[0]
        return [raw.ch_names[i] for i in bad_idx]

    except Exception as e:
        logging.error(f"坏道检测失败: {str(e)}")
        return []


def classify_ica_components(ica, raw):
    """
    不依赖 ECG/EOG 通道名的 ICA 成分自动分类。
    基于频谱特征和拓扑分布判断眼电、肌电、心电成分。
    返回: dict {'eog': [], 'muscle': [], 'cardiac': [], 'brain': []}
    """
    from scipy.stats import kurtosis
    from mne.time_frequency import psd_array_welch

    labels = {'eog': [], 'muscle': [], 'cardiac': [], 'brain': [], 'line': []}

    if ica is None or not hasattr(ica, 'n_components_') or ica.n_components_ == 0:
        return labels

    sfreq = raw.info['sfreq']
    n_times = raw.n_times
    n_fft = min(256, n_times)
    if n_fft < 8:
        labels['brain'] = list(range(ica.n_components_))
        return labels

    fmax = min(100, sfreq / 2 - 1)
    if fmax <= 1:
        labels['brain'] = list(range(ica.n_components_))
        return labels

    for idx in range(ica.n_components_):
        comp_data = ica.get_components()[:, idx]

        try:
            psd, freqs = psd_array_welch(
                comp_data[np.newaxis, :], sfreq,
                fmin=1, fmax=fmax,
                n_fft=n_fft, n_per_seg=n_fft, verbose=False
            )
            psd = psd[0]
        except Exception:
            labels['brain'].append(idx)
            continue

        if freqs is None or len(freqs) == 0:
            labels['brain'].append(idx)
            continue

        total_power = np.sum(psd) + 1e-12

        line_mask = (freqs >= 48) & (freqs <= 52)
        high_mask = freqs >= 30
        low_mask = (freqs >= 1) & (freqs < 4)

        line_power = np.sum(psd[line_mask]) / total_power if np.any(line_mask) else 0
        high_power = np.sum(psd[high_mask]) / total_power if np.any(high_mask) else 0
        low_power = np.sum(psd[low_mask]) / total_power if np.any(low_mask) else 0

        try:
            comp_ts = ica.get_sources(raw).get_data()[idx]
            comp_kurt = kurtosis(comp_ts)
        except Exception:
            comp_kurt = 0

        # ========== 新增：低通道模式使用更严格的分类阈值 ==========
        n_channels = len(raw.ch_names)
        if n_channels <= 5:
            line_thr = 0.5
            high_thr = 0.6
            kurt_thr = 8.0
            low_thr = 0.4
        else:
            line_thr = 0.3
            high_thr = 0.4
            kurt_thr = 5.0
            low_thr = 0.3
        # ========== 修改结束 ==========

        if line_power > line_thr:
            labels['line'].append(idx)
        elif high_power > high_thr:
            labels['muscle'].append(idx)
        elif comp_kurt > kurt_thr and low_power > low_thr:
            labels['eog'].append(idx)
        else:
            labels['brain'].append(idx)

    return labels