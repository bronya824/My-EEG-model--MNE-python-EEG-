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
    pytest.importorskip("pyedflib")  # 关键依赖检查

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