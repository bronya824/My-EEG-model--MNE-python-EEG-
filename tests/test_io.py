import pytest
import numpy as np
import mne
import base64
from pathlib import Path
from pydantic import ValidationError
from eeg_proc.io.receiver import EEGReceiver
from eeg_proc.models.eeg_model import WechatEEGInput


@pytest.fixture
def valid_wechat_data():
    rng = np.random.default_rng(seed=42)
    standard_names = ['FP1', 'FP2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2']
    return {
        "device_id": "eeg-12345",
        "user_id": "usr-20240415",
        "timestamp": "2024-04-15T09:30:00+08:00",
        "sampling_rate": 500.0,
        "n_channels": 10,
        "channel_names": standard_names,
        "eeg_data": base64.b64encode(
            rng.standard_normal(10 * 1000).astype(np.float32).tobytes()
        ).decode()
    }


def test_invalid_user_id(valid_wechat_data):
    invalid_data = valid_wechat_data.copy()
    invalid_data["user_id"] = "invalid_id"
    with pytest.raises(ValidationError) as exc:
        WechatEEGInput(**invalid_data)
    assert "String should match pattern '^usr-\\d{8}$'" in str(exc.value)


def test_parse_wechat_data(valid_wechat_data):
    receiver = EEGReceiver()
    raw = receiver.parse_wechat_input(valid_wechat_data)
    assert raw.ch_names == ['FP1', 'FP2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2']
    assert raw.info['sfreq'] == 500.0


def test_edf_loading(tmp_path: Path):
    pytest.importorskip("pyedflib")

    from mne import export
    edf_path = tmp_path / "test.edf"

    data = np.random.default_rng().standard_normal((32, 1000))
    info = mne.create_info(32, sfreq=250, ch_types='eeg')
    raw = mne.io.RawArray(data, info)

    export.export_raw(edf_path, raw, fmt='edf', overwrite=True)

    receiver = EEGReceiver()
    loaded_raw = receiver.load_edf(str(edf_path))
    assert loaded_raw.info['sfreq'] == 250.0


def test_channel_renaming(tmp_path: Path):
    pytest.importorskip("pyedflib")

    from mne import export
    edf_path = tmp_path / "test.edf"

    data = np.random.default_rng().standard_normal((2, 1000))
    info = mne.create_info(['EEG1', 'EEG2'], sfreq=250, ch_types='eeg')
    raw = mne.io.RawArray(data, info)

    export.export_raw(edf_path, raw, fmt='edf', overwrite=True)

    receiver = EEGReceiver()
    loaded_raw = receiver.load_edf(
        str(edf_path),
        channel_map={'EEG1': 'FP1', 'EEG2': 'FP2'}
    )
    assert loaded_raw.ch_names == ['FP1', 'FP2']


# ===========================================================
# 新增：FIF 文件加载测试
# ===========================================================

def test_fif_loading(tmp_path: Path):
    """验证 FIF 文件基本加载功能"""
    fif_path = tmp_path / "test_raw.fif"
    data = np.random.default_rng(seed=42).standard_normal((8, 1000))
    info = mne.create_info(8, sfreq=250, ch_types='eeg')
    raw = mne.io.RawArray(data, info)
    raw.save(fif_path, overwrite=True)

    receiver = EEGReceiver()
    loaded = receiver.load_fif(str(fif_path))
    assert loaded.info['sfreq'] == 250.0


def test_fif_file_not_found(tmp_path: Path):
    """FIF 文件不存在应抛出 FileNotFoundError (被 RuntimeError 包裹)"""
    receiver = EEGReceiver()
    with pytest.raises((FileNotFoundError, RuntimeError), match="不存在"):
        receiver.load_fif(str(tmp_path / "nonexistent.fif"), allow_missing=False)


# ===========================================================
# 新增：采样率与边界条件测试
# ===========================================================

def test_edf_high_sampling_rate_resample(tmp_path: Path):
    """EDF 采样率超出 max_sampling_rate 时触发降采样"""
    pytest.importorskip("pyedflib")
    from mne import export

    edf_path = tmp_path / "high_sfreq.edf"
    sfreq = 1000.0
    data = np.random.default_rng(seed=42).standard_normal((4, 5000))
    info = mne.create_info(4, sfreq=sfreq, ch_types='eeg')
    raw = mne.io.RawArray(data, info)
    export.export_raw(edf_path, raw, fmt='edf', overwrite=True)

    receiver = EEGReceiver(max_sampling_rate=250)
    loaded = receiver.load_edf(str(edf_path))
    assert loaded.info['sfreq'] <= 250.0


def test_edf_missing_channel_in_map(tmp_path: Path):
    """channel_map 指定了 EDF 中不存在的通道应抛出 RuntimeError (包裹内部 ValueError)"""
    pytest.importorskip("pyedflib")
    from mne import export

    edf_path = tmp_path / "test.edf"
    data = np.random.default_rng().standard_normal((2, 1000))
    info = mne.create_info(['CH1', 'CH2'], sfreq=250, ch_types='eeg')
    raw = mne.io.RawArray(data, info)
    export.export_raw(edf_path, raw, fmt='edf', overwrite=True)

    receiver = EEGReceiver()
    with pytest.raises(RuntimeError, match="缺少"):
        receiver.load_edf(str(edf_path), channel_map={'NOT_EXIST': 'Fp1'})


def test_parse_wechat_data_invalid_bytes_length(valid_wechat_data):
    """解码后字节数不是通道数的整数倍应拒绝 (eeg_data min_length=100 限制)"""
    receiver = EEGReceiver()
    invalid_data = valid_wechat_data.copy()
    # 113 字节 (质数), base64 编码后 > 100 字符, 且 113 % 10 ≠ 0
    invalid_data["eeg_data"] = base64.b64encode(b'\x00' * 113).decode()
    invalid_data["n_channels"] = 10
    with pytest.raises(ValueError, match="无法整除"):
        receiver.parse_wechat_input(invalid_data)