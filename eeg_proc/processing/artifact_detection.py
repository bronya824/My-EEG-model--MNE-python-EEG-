# eeg_proc/processing/artifact_detection.py

import mne
import numpy as np
import logging


def compute_bad_channels(raw):
    """检测坏通道的替代实现"""
    try:
        # 方法1：使用MNE的自动坏道检测（如果可用）
        if hasattr(mne.preprocessing, 'find_bad_channels'):
            bads, _ = mne.preprocessing.find_bad_channels(raw)
            return bads

        # 方法2：手动检测（基于峰峰值统计）
        data = raw.get_data()
        ptp = np.ptp(data, axis=1)  # 计算每个通道的峰峰值

        # 使用MAD（中位数绝对偏差）检测异常值
        median = np.median(ptp)
        mad = np.median(np.abs(ptp - median))
        z_scores = (ptp - median) / (mad * 1.4826)  # 转换为Z分数

        # 标记Z分数 > 3 的通道为坏道
        bad_idx = np.where(np.abs(z_scores) > 3)[0]
        return [raw.ch_names[i] for i in bad_idx]

    except Exception as e:
        logging.error(f"坏道检测失败: {str(e)}")
        return []  # 返回空列表避免中断流程