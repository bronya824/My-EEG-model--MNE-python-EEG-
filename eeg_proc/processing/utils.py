import mne
import numpy as np

def calculate_snr(raw: mne.io.BaseRaw) -> float:
    """计算信噪比（基于网页5的质量评估要求）"""
    data = raw.get_data()
    signal_power = np.mean(np.abs(data))
    noise_power = np.std(data - np.mean(data, axis=1, keepdims=True))
    return 20 * np.log10(signal_power / noise_power)

def _generate_events(raw: mne.io.BaseRaw) -> np.ndarray:
    """事件检测工具函数（内部使用）"""
    return mne.find_events(raw, stim_channel="Status")