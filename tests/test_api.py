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
# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('test_api')


@pytest.fixture
def client(tmp_path):
    logger = logging.getLogger('test_api')
    logger.setLevel(logging.INFO)  # 提高日志级别

    # 添加强制性大小过滤器
    class SizeFilter(logging.Filter):
        def filter(self, record):
            return len(record.getMessage()) < 1024  # 限制1KB

    logger.addFilter(SizeFilter())

    # 配置应用
    upload_dir = tmp_path / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    app.config.update({
        "TESTING": True,
        "UPLOAD_FOLDER": str(upload_dir),
        "MAX_CONTENT_LENGTH": 100 * 1024 * 1024
    })

    with app.test_client() as client:
        yield client

    # 清理
    if upload_dir.exists():
        shutil.rmtree(upload_dir)


def _generate_valid_eeg():
    """生成有效的EEG测试数据（30秒数据）"""
    montage = mne.channels.make_standard_montage('standard_1020')
    standard_names = montage.ch_names[:21]
    n_channels = len(standard_names)

    # 30秒数据（500Hz × 30秒 = 15000采样点）
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


def test_process_endpoint(client):
    """测试/process端点"""
    valid_data = _generate_valid_eeg()

    # 打印测试数据详情
    logger.info("=== 测试数据详情 ===")
    logger.info(f"设备ID: {valid_data['device_id']}")
    logger.info(f"用户ID: {valid_data['user_id']}")
    logger.info(f"采样率: {valid_data['sampling_rate']} Hz")
    logger.info(f"通道数: {valid_data['n_channels']}")
    logger.info(f"通道名称: {valid_data['channel_names']}")

    eeg_bytes = base64.b64decode(valid_data['eeg_data'])
    logger.info(f"EEG数据长度: {len(eeg_bytes)} 字节")
    logger.info(f"预期采样点数: {len(eeg_bytes) // 4 // valid_data['n_channels']}")

    # 发送请求
    response = client.post('/process', json=valid_data)

    # 打印响应信息
    logger.info(f"响应状态码: {response.status_code}")
    if response.status_code != 200:
        logger.error(f"错误响应: {response.get_data(as_text=True)}")

    # 断言
    assert response.status_code == 200, (
        f"Expected status 200, got {response.status_code}. "
        f"Response: {response.get_data(as_text=True)}"
    )
    assert response.json['metadata']['ch_names'] == valid_data['channel_names']


def test_edf_upload(client, tmp_path: Path):
    """测试EDF文件上传端点"""
    edf_path = tmp_path / "valid.edf"
    valid_data = _generate_valid_eeg()
    channel_names = valid_data["channel_names"]
    n_channels = len(channel_names)

    # 生成30秒数据（250Hz × 30秒 = 7500采样点）
    n_times = 7500
    rng = np.random.default_rng(42)
    data = rng.standard_normal((n_channels, n_times))

    # 打印调试信息
    logger.info(f"测试使用的通道名称: {channel_names}")
    logger.info(f"生成数据: {n_channels}通道, {n_times}采样点")

    # 创建Raw对象
    info = mne.create_info(
        ch_names=channel_names,
        sfreq=250,
        ch_types='eeg'
    )
    raw = mne.io.RawArray(data, info)

    # 设置电极位置
    montage = mne.channels.make_standard_montage('standard_1020')
    raw.set_montage(montage, match_case=False, on_missing='ignore')

    # 导出EDF
    export.export_raw(edf_path, raw, fmt='edf', overwrite=True)

    # 打印EDF文件信息
    logger.info(f"生成的EDF文件路径: {edf_path}")
    logger.info(f"EDF文件大小: {os.path.getsize(edf_path)}字节")

    # 验证EDF文件
    try:
        raw_test = mne.io.read_raw_edf(str(edf_path), preload=False)
        logger.info(f"EDF文件验证成功: {len(raw_test.ch_names)}通道, {raw_test.n_times}采样点")
        logger.info(f"通道名称: {raw_test.ch_names[:10]}...")
    except Exception as e:
        logger.error(f"EDF文件验证失败: {str(e)}")
        pytest.fail(f"生成的EDF文件无效: {str(e)}")

    # 上传文件
    with open(edf_path, 'rb') as f:
        response = client.post('/api/v1/upload', data={'file': (f, 'valid.edf')})

    # 打印响应信息
    logger.info(f"响应状态码: {response.status_code}")
    logger.info(f"响应内容: {response.get_data(as_text=True)}")

    # 断言
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

    # 测试各种命名变体
    cases = [
        (['FP1', 'FP2'], ['Fp1', 'Fp2']),
        (['fp1', 'fp2'], ['Fp1', 'Fp2']),
        (['eeg fp1', 'eeg fp2'], ['Fp1', 'Fp2']),
        (['AF3', 'PO4'], ['AF3', 'PO4'])
    ]

    for input_names, expected in cases:
        assert _standardize_ch_names(input_names) == expected


if __name__ == "__main__":
    pytest.main(["-v", "--log-level=DEBUG"])