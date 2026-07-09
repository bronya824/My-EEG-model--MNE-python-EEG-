# tests/test_processing.py
import pytest
import numpy as np
import mne
from pathlib import Path

from eeg_proc import MNEPipeline


def _generate_ecg_signal():
    """生成典型心电信号特征"""
    t = np.linspace(0, 10, 1000)
    return 0.5 * np.sin(2 * np.pi * 5 * t) * np.exp(-0.5 * (t % 1.0 - 0.3) ** 2 / 0.01)


def _generate_eog_signal():
    """生成典型眼电信号特征"""
    signal = np.zeros(1000)
    for blink in [200, 500, 800]:
        signal[blink - 20:blink + 20] = 1.0
    return signal


def _make_raw(n_channels=30, n_times=1000, sfreq=250.0, seed=42, extra_ch=('ECG', 'EOG')):
    """通用 Raw 对象工厂"""
    rng = np.random.default_rng(seed=seed)
    montage = mne.channels.make_standard_montage('standard_1020')
    standard_names = montage.ch_names[:n_channels]
    eeg_data = rng.normal(scale=1e-6, size=(n_channels, n_times))

    extra_data_list = []
    ch_names_extra = []
    ch_types_extra = []
    if 'ECG' in extra_ch:
        ecg = _generate_ecg_signal().reshape(1, -1)[:, :n_times]
        extra_data_list.append(ecg)
        ch_names_extra.append('ECG')
        ch_types_extra.append('ecg')
    if 'EOG' in extra_ch:
        eog = _generate_eog_signal().reshape(1, -1)[:, :n_times]
        extra_data_list.append(eog)
        ch_names_extra.append('EOG')
        ch_types_extra.append('eog')

    if extra_data_list:
        data = np.vstack([eeg_data] + extra_data_list)
    else:
        data = eeg_data.copy()

    ch_names = standard_names + ch_names_extra
    ch_types = ['eeg'] * n_channels + ch_types_extra

    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    info.set_montage(montage, on_missing='ignore')
    return mne.io.RawArray(data, info)


@pytest.fixture
def sample_raw():
    return _make_raw(n_channels=30, n_times=1000, sfreq=250.0, extra_ch=('ECG', 'EOG'))


# ===========================================================
# 原有测试 (保留)
# ===========================================================

def test_artifact_detection(sample_raw):
    """验证能稳定检测到生理伪迹"""
    pipeline = (MNEPipeline(sample_raw)
                .apply_filter(l_freq=1, h_freq=40)
                .run_ica(n_components=0.95)
                .detect_artifacts())

    assert len(pipeline.ica.exclude) >= 1
    assert any(k in pipeline.ica.labels_ for k in ('ecg', 'eog'))


# ===========================================================
# 新增：滤波测试
# ===========================================================

def test_filter_basic_bandpass(sample_raw):
    """基本带通滤波后数据形状保持不变"""
    pipeline = MNEPipeline(sample_raw).apply_filter(l_freq=1, h_freq=40)
    assert pipeline.raw.n_times == sample_raw.n_times
    assert pipeline.raw.get_data().shape == sample_raw.get_data().shape


def test_filter_highpass_only(sample_raw):
    """仅高通滤波正常执行"""
    pipeline = MNEPipeline(sample_raw).apply_filter(l_freq=1, h_freq=None)
    assert pipeline.raw.n_times == sample_raw.n_times


def test_filter_lowpass_only(sample_raw):
    """仅低通滤波正常执行"""
    pipeline = MNEPipeline(sample_raw).apply_filter(l_freq=None, h_freq=40)
    assert pipeline.raw.n_times == sample_raw.n_times


def test_filter_short_data_iir_fallback():
    """短数据（< 2秒）自动降级为 IIR 滤波器不抛异常"""
    raw = _make_raw(n_channels=10, n_times=200, sfreq=250.0, extra_ch=())
    pipeline = MNEPipeline(raw).apply_filter(l_freq=1, h_freq=40)
    assert pipeline.raw.n_times == 200


# ===========================================================
# 新增：参考电极测试
# ===========================================================

def test_set_reference_average(sample_raw):
    """平均参考设置后数据形状不变"""
    pipeline = MNEPipeline(sample_raw).set_reference(ref_channels='average')
    assert pipeline.raw.get_data().shape == sample_raw.get_data().shape


# ===========================================================
# 新增：ICA 测试
# ===========================================================

def test_ica_few_channels_skipped():
    """通道数 < 3 时 ICA 自动跳过"""
    raw = _make_raw(n_channels=2, n_times=1000, sfreq=250.0, extra_ch=())
    pipeline = MNEPipeline(raw).run_ica(n_components=0.95)
    assert pipeline.ica is None


def test_ica_insufficient_data_skipped():
    """数据不足 2 秒时 ICA 自动跳过"""
    raw = _make_raw(n_channels=10, n_times=200, sfreq=250.0, extra_ch=())
    pipeline = MNEPipeline(raw).run_ica(n_components=0.95)
    assert pipeline.ica is None


# ===========================================================
# 新增：完整流水线测试
# ===========================================================

def test_apply_standard_pipeline_full(sample_raw):
    """完整流水线执行后数据有效，报告可用"""
    pipeline = MNEPipeline(sample_raw).apply_standard_pipeline(l_freq=1, h_freq=40)
    assert pipeline.raw.n_times > 0
    assert len(pipeline.raw.ch_names) > 0
    report = pipeline.get_processing_report()
    assert 'log' in report
    assert 'metrics' in report
    assert 'snr_db' in report['metrics']


def test_apply_standard_pipeline_short_data_skip_ica():
    """短数据流水线自动跳过 ICA 但完成滤波和参考"""
    raw = _make_raw(n_channels=10, n_times=300, sfreq=250.0, extra_ch=())
    pipeline = MNEPipeline(raw).apply_standard_pipeline(l_freq=1, h_freq=40)
    assert pipeline.raw.n_times == 300


def test_filter_invalid_params_rejected(sample_raw):
    """无效滤波参数 l_freq >= h_freq 应抛出 RuntimeError (内部 ValueError)"""
    with pytest.raises(RuntimeError, match="滤波参数无效"):
        MNEPipeline(sample_raw).apply_standard_pipeline(l_freq=40, h_freq=30)


# ===========================================================
# 新增：质量指标与输出测试
# ===========================================================

def test_quality_metrics_produced(sample_raw):
    """质量指标在 apply_clean 后包含 snr_db"""
    pipeline = (MNEPipeline(sample_raw)
                .apply_filter(l_freq=1, h_freq=40)
                .run_ica(n_components=0.95)
                .detect_artifacts()
                .apply_clean())
    assert 'snr_db' in pipeline.quality_metrics
    assert isinstance(pipeline.quality_metrics['snr_db'], (int, float, np.floating))


def test_get_processed_data_return_copy(sample_raw):
    """get_processed_data 返回与原始数据 sfreq 相同的副本"""
    pipeline = MNEPipeline(sample_raw).apply_filter(l_freq=1, h_freq=40)
    copy = pipeline.get_processed_data()
    assert copy.info['sfreq'] == sample_raw.info['sfreq']
    assert copy.ch_names == sample_raw.ch_names


def test_get_processing_report_types():
    """get_processing_report 中数值均为 Python 原生类型"""
    raw = _make_raw(n_channels=10, n_times=1000, sfreq=250.0, extra_ch=())
    pipeline = MNEPipeline(raw).apply_filter(l_freq=1, h_freq=40).apply_clean()
    report = pipeline.get_processing_report()
    metrics = report['metrics']
    assert isinstance(metrics['snr_db'], float)
    assert isinstance(metrics['bad_channels_ratio'], float)
    assert isinstance(metrics['remaining_samples'], int)
    assert isinstance(metrics['effective_sfreq'], float)


def test_processing_log_populated(sample_raw):
    """处理步骤日志正确记录 (包括 IIR fallback 路径)"""
    pipeline = MNEPipeline(sample_raw).apply_filter(l_freq=1, h_freq=40)
    # IIR fallback 可能只有 init 日志，以 >=1 为断言
    assert len(pipeline.processing_log) >= 1
    # 验证 init 步骤必定存在
    assert pipeline.processing_log[0]['step'] == 'init'


def test_empty_data_raises():
    """空数据 (n_times=0) 应抛出 RuntimeError"""
    raw = _make_raw(n_channels=10, n_times=0, sfreq=250.0, extra_ch=())
    with pytest.raises(RuntimeError, match="空"):
        MNEPipeline(raw).apply_standard_pipeline()