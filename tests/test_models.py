# tests/test_models.py
import pytest
import numpy as np
import base64
from pydantic import ValidationError
from eeg_proc.models.eeg_model import WechatEEGInput, AliyunOutputModel


def _build_valid_input(**overrides):
    """构建有效的WechatEEGInput原始数据"""
    rng = np.random.default_rng(seed=42)
    n_channels = 10
    n_samples = 1000
    standard_names = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2']
    data = {
        "device_id": "eeg-9999",
        "user_id": "usr-20240423",
        "timestamp": "2024-04-23T09:30:00+08:00",
        "sampling_rate": 500.0,
        "n_channels": n_channels,
        "channel_names": standard_names,
        "eeg_data": base64.b64encode(
            rng.standard_normal(n_channels * n_samples).astype(np.float32).tobytes()
        ).decode()
    }
    data.update(overrides)
    return data


class TestWechatEEGInput:
    """WechatEEGInput 模型核心验证测试"""

    def test_valid_input(self):
        """正常输入应通过所有校验"""
        data = _build_valid_input()
        obj = WechatEEGInput(**data)
        assert obj.device_id == "eeg-9999"
        assert obj.user_id == "usr-20240423"
        assert obj.sampling_rate == 500.0
        assert obj.n_channels == 10
        assert len(obj.channel_names) == 10

    def test_invalid_device_id_short(self):
        """device_id 格式错误'XXX'应拒绝"""
        with pytest.raises(ValidationError, match="String should match pattern"):
            WechatEEGInput(**_build_valid_input(device_id="XXX"))

    def test_invalid_device_id_no_number(self):
        """device_id 缺少数字后缀应拒绝"""
        with pytest.raises(ValidationError, match="String should match pattern"):
            WechatEEGInput(**_build_valid_input(device_id="eeg-abc"))

    def test_invalid_user_id_format(self):
        """user_id 格式错误应拒绝"""
        with pytest.raises(ValidationError, match="String should match pattern"):
            WechatEEGInput(**_build_valid_input(user_id="invalid"))

    def test_invalid_user_id_too_short(self):
        """user_id 日期位数不足应拒绝"""
        with pytest.raises(ValidationError, match="String should match pattern"):
            WechatEEGInput(**_build_valid_input(user_id="usr-123"))

    def test_sampling_rate_zero(self):
        """采样率为 0 应拒绝"""
        with pytest.raises(ValidationError):
            WechatEEGInput(**_build_valid_input(sampling_rate=0.0))

    def test_sampling_rate_exceeds_max(self):
        """采样率超过 2000 Hz 应拒绝"""
        with pytest.raises(ValidationError):
            WechatEEGInput(**_build_valid_input(sampling_rate=2001.0))

    def test_n_channels_zero(self):
        """n_channels 为 0 应拒绝"""
        with pytest.raises(ValidationError):
            WechatEEGInput(**_build_valid_input(n_channels=0))

    def test_n_channels_exceeds_max(self):
        """n_channels 超过 128 应拒绝"""
        with pytest.raises(ValidationError):
            WechatEEGInput(**_build_valid_input(n_channels=129))

    def test_channel_name_standardization_case_insensitive(self):
        """通道名大小写不敏感标准化 (validator 返回全大写)"""
        data = _build_valid_input(channel_names=['fp1', 'FP2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2'])
        obj = WechatEEGInput(**data)
        # 校验器统一转大写: fp1 -> FP1
        assert 'FP1' in obj.channel_names
        assert 'FP2' in obj.channel_names
        assert len(obj.channel_names) == 10

    def test_channel_name_t3_t4_alias(self):
        """T3/T4 别名自动映射为 T7/T8"""
        data = _build_valid_input(channel_names=['Fp1', 'Fp2', 'T3', 'T4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2'])
        obj = WechatEEGInput(**data)
        assert 'T7' in obj.channel_names
        assert 'T8' in obj.channel_names
        assert 'T3' not in obj.channel_names
        assert 'T4' not in obj.channel_names

    def test_channel_count_mismatch(self):
        """channel_names 长度与 n_channels 不匹配应拒绝"""
        data = _build_valid_input(
            n_channels=10,
            channel_names=['Fp1', 'Fp2', 'F3']
        )
        with pytest.raises(ValueError, match="通道数量"):
            WechatEEGInput(**data)

    def test_invalid_channel_names(self):
        """非标准通道名应拒绝"""
        data = _build_valid_input(
            channel_names=['A1', 'B2', 'C3', 'D4'] + ['Fp1'] * 6
        )
        with pytest.raises(ValueError, match="非标准通道名称"):
            WechatEEGInput(**data)


class TestAliyunOutputModel:
    """AliyunOutputModel 元数据校验测试"""

    def test_valid_output(self):
        """正常输出模型构造成功"""
        output = AliyunOutputModel(
            processed_data="dGVzdA==",
            metadata={
                "sfreq": 250.0,
                "ch_names": ["Fp1", "C3", "O1"],
                "n_samples": 1000
            }
        )
        assert output.metadata['sfreq'] == 250.0
        assert output.metadata['n_samples'] == 1000

    def test_missing_metadata_field(self):
        """缺失 sfreq 字段应拒绝"""
        with pytest.raises(ValueError, match="sfreq"):
            AliyunOutputModel(
                processed_data="dGVzdA==",
                metadata={
                    "ch_names": ["Fp1"],
                    "n_samples": 100
                }
            )

    def test_invalid_sfreq_negative(self):
        """sfreq 为负数应拒绝"""
        with pytest.raises(ValueError, match="采样率"):
            AliyunOutputModel(
                processed_data="dGVzdA==",
                metadata={
                    "sfreq": -1.0,
                    "ch_names": ["Fp1"],
                    "n_samples": 100
                }
            )

    def test_invalid_n_samples_zero(self):
        """n_samples 为 0 应拒绝"""
        with pytest.raises(ValueError, match="样本数"):
            AliyunOutputModel(
                processed_data="dGVzdA==",
                metadata={
                    "sfreq": 250.0,
                    "ch_names": ["Fp1"],
                    "n_samples": 0
                }
            )

    def test_empty_channel_names(self):
        """空通道名列表应拒绝"""
        with pytest.raises(ValueError, match="为空"):
            AliyunOutputModel(
                processed_data="dGVzdA==",
                metadata={
                    "sfreq": 250.0,
                    "ch_names": [],
                    "n_samples": 100
                }
            )