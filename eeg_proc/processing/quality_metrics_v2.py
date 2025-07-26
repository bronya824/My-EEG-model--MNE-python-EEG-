# eeg_proc/processing/quality_metrics_v2.py

import numpy as np
from scipy import signal


def calculate_snr(raw, low=45, high=55):
    """
    计算信噪比（SNR）的改进版本，避免工频干扰。
    """
    try:
        data = raw.get_data()
        fs = raw.info['sfreq']

        # 计算归一化截止频率
        nyquist = fs / 2
        cutoff = np.array([low, high]) / nyquist

        # 确保截止频率在有效范围内
        cutoff = np.clip(cutoff, 0.001, 0.999)

        # 设计带阻滤波器去除工频干扰
        b, a = signal.butter(4, cutoff, 'bandstop')

        # 应用滤波器
        filtered_data = signal.filtfilt(b, a, data)

        # 计算噪声
        noise = data - filtered_data

        # 计算功率
        signal_power = np.mean(data ** 2, axis=1)
        noise_power = np.mean(noise ** 2, axis=1)

        # 避免除以零
        noise_power = np.maximum(noise_power, 1e-10)

        # 计算SNR（分贝）
        snr_db = 10 * np.log10(signal_power / noise_power)
        return np.mean(snr_db)

    except Exception as e:
        # 添加错误处理
        print(f"SNR计算错误: {str(e)}")
        return 0  # 返回默认值避免中断流程