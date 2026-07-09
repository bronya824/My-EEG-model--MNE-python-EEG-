import pytest
import numpy as np
import mne
import base64
import shutil
from pathlib import Path
from io import BytesIO
import os
from mne import export
from eeg_proc.api.rest_api import app
import logging

pytestmark = pytest.mark.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('test_api')


@pytest.fixture
def client(tmp_path):
    logger = logging.getLogger('test_api')
    logger.setLevel(logging.INFO)

    class SizeFilter(logging.Filter):
        def filter(self, record):
            return len(record.getMessage()) < 1024

    logger.addFilter(SizeFilter())

    upload_dir = tmp_path / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    app.config.update({
        "TESTING": True,
        "UPLOAD_FOLDER": str(upload_dir),
        "MAX_CONTENT_LENGTH": 100 * 1024 * 1024
    })

    with app.test_client() as client:
        yield client

    if upload_dir.exists():
        shutil.rmtree(upload_dir)


def _generate_valid_eeg():
    """生成有效的EEG测试数据（30秒数据）"""
    montage = mne.channels.make_standard_montage('standard_1020')
    standard_names = montage.ch_names[:21]
    n_channels = len(standard_names)
    n_samples = n_channels * 15000
    rng = np.random.default_rng(42)
    eeg_data = rng.standard_normal(n_samples).astype(np.float32)

    return {
        "device_id": "eeg-2024",
        "user_id": "usr-20240423",
        "timestamp": "2024-04-23T09:30:00+08:00",
        "sampling_rate": 500.0,
        "n_channels": n_channels,
        "channel_names": standard_names,
        "eeg_data": base64.b64encode(eeg_data.tobytes()).decode()
    }


# ===========================================================
# 原有测试 (保留)
# ===========================================================

def test_process_endpoint(client):
    """测试/process端点"""
    valid_data = _generate_valid_eeg()
    response = client.post('/process', json=valid_data)

    assert response.status_code == 200, (
        f"Expected status 200, got {response.status_code}. "
        f"Response: {response.get_data(as_text=True)}"
    )
    assert response.json['metadata']['ch_names'] == valid_data['channel_names']


def test_edf_upload(client, tmp_path: Path):
    """测试EDF文件上传端点 (需要 edfio)"""
    pytest.importorskip("edfio", reason="edfio not installed for EDF export")

    edf_path = tmp_path / "valid.edf"
    valid_data = _generate_valid_eeg()
    channel_names = valid_data["channel_names"]
    n_channels = len(channel_names)

    n_times = 7500
    rng = np.random.default_rng(42)
    data = rng.standard_normal((n_channels, n_times))

    info = mne.create_info(ch_names=channel_names, sfreq=250, ch_types='eeg')
    raw = mne.io.RawArray(data, info)

    montage = mne.channels.make_standard_montage('standard_1020')
    raw.set_montage(montage, match_case=False, on_missing='ignore')

    export.export_raw(edf_path, raw, fmt='edf', overwrite=True)

    with open(edf_path, 'rb') as f:
        response = client.post('/api/v1/upload', data={'file': (f, 'valid.edf')})

    assert response.status_code == 200, (
        f"Expected status 200, got {response.status_code}. "
        f"Response: {response.get_data(as_text=True)}"
    )


def test_invalid_uploads(client):
    """测试无效文件上传"""
    fake_file = (BytesIO(b'invalid'), 'test.txt')
    response = client.post('/api/v1/upload', data={'file': fake_file})
    assert response.status_code == 415


def test_error_handling(client):
    """测试错误处理"""
    response = client.post(
        '/process',
        data="not json",
        headers={'Content-Type': 'application/json'}
    )
    assert response.status_code == 415


def test_channel_standardization():
    from eeg_proc.api.rest_api import _standardize_ch_names

    cases = [
        (['FP1', 'FP2'], ['Fp1', 'Fp2']),
        (['fp1', 'fp2'], ['Fp1', 'Fp2']),
        (['eeg fp1', 'eeg fp2'], ['Fp1', 'Fp2']),
        (['AF3', 'PO4'], ['AF3', 'PO4'])
    ]

    for input_names, expected in cases:
        assert _standardize_ch_names(input_names) == expected


# ===========================================================
# 新增：FIF 文件上传测试
# ===========================================================

def test_fif_upload(client, tmp_path: Path):
    """测试 FIF 文件上传端点"""
    fif_path = tmp_path / "test_raw.fif"

    montage = mne.channels.make_standard_montage('standard_1020')
    ch_names = montage.ch_names[:10]
    n_times = 5000
    sfreq = 250.0
    data = np.random.default_rng(seed=42).standard_normal((len(ch_names), n_times))
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')
    raw = mne.io.RawArray(data, info)
    raw.set_montage(montage, on_missing='ignore')
    raw.save(fif_path, overwrite=True)

    with open(fif_path, 'rb') as f:
        response = client.post('/api/v1/upload', data={'file': (f, 'test_raw.fif')})

    assert response.status_code != 415, f"FIF should be accepted, got {response.status_code}"


# ===========================================================
# 新增：更多错误注入测试
# ===========================================================

def test_upload_no_file(client):
    """无文件上传应返回 400"""
    response = client.post('/api/v1/upload', data={})
    assert response.status_code == 400


def test_upload_empty_filename(client):
    """空文件名应返回 400"""
    fake_file = (BytesIO(b'data'), '')
    response = client.post('/api/v1/upload', data={'file': fake_file})
    assert response.status_code == 400


def test_process_invalid_json_structure(client):
    """JSON 结构有效但缺少必填字段应返回 422"""
    response = client.post('/process', json={"device_id": "eeg-1"})
    assert response.status_code == 422


def test_process_missing_channel_names(client):
    """缺少 channel_names 字段应拒绝"""
    data = _generate_valid_eeg()
    del data['channel_names']
    response = client.post('/process', json=data)
    assert response.status_code == 422


def test_process_invalid_sampling_rate(client):
    """采样率超出范围应拒绝"""
    data = _generate_valid_eeg()
    data['sampling_rate'] = 3000.0
    response = client.post('/process', json=data)
    assert response.status_code == 422


def test_allowed_file_helper():
    """allowed_file 工具函数边界测试"""
    from eeg_proc.api.rest_api import allowed_file
    assert allowed_file('test.edf') is True
    assert allowed_file('test.fif') is True
    assert allowed_file('test.txt') is False
    assert allowed_file('no_extension') is False
    assert allowed_file('test.csv') is False


if __name__ == "__main__":
    pytest.main(["-v", "--log-level=DEBUG"])