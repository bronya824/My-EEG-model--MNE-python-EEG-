# eeg_proc/processing/quality_metrics_v2.py

import numpy as np
import logging
from scipy import signal

logger = logging.getLogger('eeg_pipeline')


def calculate_snr(raw, signal_band=(1, 40), noise_band=(45, 55)):
    """计算信噪比（SNR）"""
    try:
        data = raw.get_data()
        fs = raw.info['sfreq']
        nyquist = fs / 2.0

        sig_low = float(signal_band[0]) / nyquist
        sig_high = float(signal_band[1]) / nyquist
        sig_low = float(np.clip(sig_low, 0.001, 0.999))
        sig_high = float(np.clip(sig_high, 0.001, 0.999))
        b_sig, a_sig = signal.butter(4, [sig_low, sig_high], 'bandpass')
        sig_filtered = signal.filtfilt(b_sig, a_sig, data, axis=1)
        signal_power = np.mean(sig_filtered ** 2)

        noise_low = float(noise_band[0]) / nyquist
        noise_high = float(noise_band[1]) / nyquist
        noise_low = float(np.clip(noise_low, 0.001, 0.999))
        noise_high = float(np.clip(noise_high, 0.001, 0.999))
        b_noise, a_noise = signal.butter(4, [noise_low, noise_high], 'bandpass')
        noise_filtered = signal.filtfilt(b_noise, a_noise, data, axis=1)
        noise_power = np.mean(noise_filtered ** 2)

        noise_power = max(noise_power, 1e-15)
        snr_db = 10 * np.log10(signal_power / noise_power)
        return float(snr_db)

    except Exception as e:
        logger.warning(f"SNR calculation error: {e}")
        return 0.0


def compute_band_power(raw, band, picks='eeg'):
    """计算指定频段的平均功率 (uV^2)"""
    from mne.time_frequency import psd_array_welch
    try:
        data = raw.get_data(picks=picks)
        sfreq = float(raw.info['sfreq'])
        n_times = int(data.shape[1])

        if n_times < 8:
            return np.nan

        n_fft = min(256, n_times)
        fmin = float(band[0])
        fmax = float(band[1])

        psds, freqs = psd_array_welch(
            data, sfreq, fmin=fmin, fmax=fmax,
            n_fft=n_fft, n_per_seg=n_fft, verbose=False
        )
        if freqs is None or len(freqs) == 0:
            return np.nan
        return float(np.mean(psds))
    except Exception as e:
        logger.debug(f"Band power calculation failed for {band}: {e}")
        return np.nan


def compute_depression_markers(raw):
    """
    计算抑郁症相关的 EEG 生物标志物。
    对非标准EEG数据（少通道、非eeg类型）自动跳过。
    """
    import mne
    try:
        picks = mne.pick_types(raw.info, eeg=True, exclude='bads')
        n_channels = len(raw.ch_names)

        # ========== 新增：低通道模式回退到所有通道 ==========
        if len(picks) == 0 and n_channels <= 5:
            picks = list(range(n_channels))
            logger.debug(f"低通道模式: 无EEG类型标记，使用全部 {n_channels} 个通道计算频段功率")
        # ========== 新增结束 ==========

        if len(picks) == 0:
            return {}
        if raw.n_times < 10:
            return {}

        delta_power = compute_band_power(raw, (1, 4), picks=picks)
        theta_power = compute_band_power(raw, (4, 8), picks=picks)
        alpha_power = compute_band_power(raw, (8, 13), picks=picks)
        beta_power = compute_band_power(raw, (13, 30), picks=picks)

        markers = {
            'delta_power': delta_power,
            'theta_power': theta_power,
            'alpha_power': alpha_power,
            'beta_power': beta_power,
        }

        if not (np.isnan(theta_power) or np.isnan(alpha_power)):
            markers['theta_alpha_ratio'] = float(theta_power / (alpha_power + 1e-12))
        else:
            markers['theta_alpha_ratio'] = np.nan

        if not (np.isnan(delta_power) or np.isnan(theta_power)):
            markers['delta_theta_ratio'] = float(delta_power / (theta_power + 1e-12))
        else:
            markers['delta_theta_ratio'] = np.nan

        if not (np.isnan(alpha_power) or np.isnan(beta_power)):
            markers['alpha_beta_ratio'] = float(alpha_power / (beta_power + 1e-12))
        else:
            markers['alpha_beta_ratio'] = np.nan

        # Frontal alpha asymmetry (F4 - F3 or F8 - F7)
        ch_names = [raw.ch_names[i] for i in picks]
        left_frontal = None
        right_frontal = None

        for ch in ch_names:
            ch_upper = ch.upper().replace('Z', 'z')
            if ch_upper in ('F3', 'F7'):
                left_frontal = ch
            elif ch_upper in ('F4', 'F8'):
                right_frontal = ch

        if left_frontal and right_frontal:
            try:
                from mne.time_frequency import psd_array_welch
                left_data = raw.get_data(picks=[left_frontal])
                right_data = raw.get_data(picks=[right_frontal])
                sfreq = float(raw.info['sfreq'])
                n_fft = min(256, raw.n_times)
                psd_l, freqs_l = psd_array_welch(
                    left_data, sfreq, fmin=8, fmax=13,
                    n_fft=n_fft, n_per_seg=n_fft, verbose=False
                )
                psd_r, freqs_r = psd_array_welch(
                    right_data, sfreq, fmin=8, fmax=13,
                    n_fft=n_fft, n_per_seg=n_fft, verbose=False
                )
                if freqs_l is not None and len(freqs_l) > 0:
                    alpha_l = float(np.mean(psd_l))
                    alpha_r = float(np.mean(psd_r))
                    markers['frontal_alpha_asymmetry'] = float(
                        np.log(alpha_r + 1e-12) - np.log(alpha_l + 1e-12)
                    )
                    markers['frontal_alpha_asymmetry_channels'] = f"{right_frontal} - {left_frontal}"
            except Exception as e:
                logger.debug(f"Frontal asymmetry calculation failed: {e}")

        return markers

    except Exception as e:
        logger.warning(f"Depression marker calculation error: {e}")
        return {}