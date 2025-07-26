# eeg_proc/models/eeg_model.py
from pydantic import BaseModel, Field, field_validator, ConfigDict
from typing import List, Dict, Any
import base64
import numpy as np
import mne
import datetime
import logging

from pydantic_core.core_schema import ValidationInfo

logger = logging.getLogger(__name__)

class WechatEEGInput(BaseModel):
    """符合国际10-20系统最新标准的EEG输入模型"""
    model_config = ConfigDict(
        str_to_lower=False,
        arbitrary_types_allowed=True,
        validate_default=True,
        extra='forbid'
    )

    device_id: str = Field(
        ...,
        pattern=r'^eeg-\d+$',
        examples=["eeg-2024"],
        json_schema_extra={"error_msg": "设备ID必须以eeg-开头"}
    )

    user_id: str = Field(
        ...,
        pattern=r'^usr-\d{8}$',
        examples=["usr-20240423"],
        json_schema_extra={"error_msg": "用户ID格式错误，示例：usr-20240423"}
    )

    timestamp: str = Field(
        ...,
        examples=["2024-04-23T14:30:00+08:00"],
        description="ISO 8601格式时间戳"
    )

    sampling_rate: float = Field(..., gt=0, le=2000)
    n_channels: int = Field(..., gt=0, le=128)
    channel_names: List[str] = Field(
        ...,
        min_length=1,
        examples=[["FP1", "F3", "C3"]],
        description="国际10-20系统标准通道名称"
    )

    eeg_data: str = Field(
        ...,
        min_length=100,
        description="Base64编码的EEG数据"
    )

    # eeg_proc/models/eeg_model.py
    @field_validator('channel_names', mode='after')
    @classmethod
    def validate_channels(cls, v: List[str], values: ValidationInfo) -> List[str]:
        """标准化通道名称并验证（兼容大小写）"""
        # 定义标准名称的小写形式集合（扩展版）
        STANDARD_10_20_LOWER = {
            'fp1', 'fpz', 'fp2',
            'af1', 'af2', 'af3', 'af4', 'af5', 'af6', 'af7', 'af8', 'af9', 'af10', 'afz',
            'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'f10', 'fz',
            'fc1', 'fc2', 'fc3', 'fc4', 'fc5', 'fc6', 'fcz',
            'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'cz',
            'cp1', 'cp2', 'cp3', 'cp4', 'cp5', 'cp6', 'cpz',
            'p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7', 'p8', 'p9', 'p10', 'pz',
            'po3', 'po4', 'po7', 'po8', 'poz',
            'o1', 'o2', 'oz',
            't7', 't8', 'tp7', 'tp8', 'tp9', 'tp10',
            'm1', 'm2'
        }

        # 1. 统一处理：转换为大写并替换T3/T4
        processed = []
        for original in v:
            # 去除空白并处理T3/T4别名
            cleaned = original.strip().upper()
            cleaned = cleaned.replace("T3", "T7").replace("T4", "T8")
            processed.append(cleaned)

        # 2. 验证不区分大小写（使用小写比较）
        lower_names = {name.lower() for name in processed}
        if invalid := lower_names - STANDARD_10_20_LOWER:
            # 仅报告前5个无效通道名
            invalid_list = list(invalid)[:5]
            invalid_str = ', '.join(invalid_list)
            if len(invalid) > 5:
                invalid_str += f" 等{len(invalid)}个"
            raise ValueError(f"非标准通道名称: {invalid_str}")

        # 3. 检查通道数量（安全访问）
        n_channels_declared = values.data.get('n_channels')
        if n_channels_declared is not None and len(processed) != n_channels_declared:
            raise ValueError(
                f"通道数量声明{n_channels_declared} 实际检测到{len(processed)}"
            )

        # 4. 返回标准化后的通道名称（全大写）
        return processed

    def to_raw(self) -> mne.io.BaseRaw:
        """增强数据转换的健壮性"""
        try:
            if len(self.eeg_data) < self.n_channels * 100:
                raise ValueError("EEG数据长度不足")

            decoded = base64.b64decode(self.eeg_data, validate=True)
            eeg_array = np.frombuffer(decoded, dtype=np.float32)

            expected_samples = eeg_array.size // self.n_channels
            eeg_array = eeg_array[:self.n_channels * expected_samples].reshape(self.n_channels, -1)

            info = mne.create_info(
                ch_names=self.channel_names,
                sfreq=self.sampling_rate,
                ch_types='eeg'
            )
            montage = mne.channels.make_standard_montage('standard_1005')
            info.set_montage(montage, on_missing='raise')

            return mne.io.RawArray(eeg_array, info)
        except Exception as e:
            raise RuntimeError(f"EEG数据转换失败: {str(e)}") from e

    def to_info(self) -> mne.Info:
        """创建MNE Info对象"""
        return mne.create_info(
            ch_names=self.channel_names,
            sfreq=self.sampling_rate,
            ch_types=['eeg'] * self.n_channels
        )


class AliyunOutputModel(BaseModel):
    """增强元数据校验和上传功能"""
    model_config = ConfigDict(extra='forbid')

    processed_data: str = Field(..., description="Base64编码的预处理数据")
    metadata: Dict[str, Any] = Field(
        ...,
        description="必须包含sfreq/ch_names/n_samples字段"
    )

    @field_validator('metadata')
    @classmethod
    def validate_metadata(cls, v: Dict) -> Dict:
        # 定义字段期望类型（兼容NumPy类型）
        required_types = {
            'sfreq': (float, np.floating),
            'ch_names': (list, np.ndarray),
            'n_samples': (int, np.integer)
        }

        # 检查缺失字段
        missing = [field for field in required_types if field not in v]
        if missing:
            raise ValueError(f"元数据缺失必要字段：{', '.join(missing)}")

        # 检查字段类型（兼容NumPy类型）
        type_errors = []
        for field, expected_types in required_types.items():
            if not any(isinstance(v[field], t) for t in expected_types):
                actual_type = type(v[field]).__name__
                expected_names = [t.__name__ for t in expected_types]
                type_errors.append(
                    f"字段 '{field}' 应为 {' 或 '.join(expected_names)}，实际为 {actual_type}"
                )

        if type_errors:
            raise TypeError("; ".join(type_errors))

        # 额外验证
        if not v['ch_names']:
            raise ValueError("通道名称列表不能为空")

        if v['sfreq'] <= 0:
            raise ValueError(f"无效的采样率: {v['sfreq']}")

        if v['n_samples'] <= 0:
            raise ValueError(f"无效的样本数: {v['n_samples']}")

        return v

    def decode_features(self) -> Dict:
        import json
        decoded = base64.b64decode(self.processed_data)
        return json.loads(decoded.decode('utf-8'))

    def upload(self, bucket_name: str = "eeg-data"):
        """上传数据到阿里云（模拟实现）"""
        try:
            # 在实际应用中，这里会有阿里云OSS的上传代码

            # 记录上传信息
            upload_info = {
                "bucket": bucket_name,
                "data_size": len(self.processed_data),
                "channels": len(self.metadata['ch_names']),
                "samples": self.metadata['n_samples'],
                "sfreq": self.metadata['sfreq']
            }

            logger.info(f"数据上传成功到存储桶: {bucket_name}")
            logger.debug(f"上传详情: {upload_info}")

            return {
                "status": "success",
                **upload_info
            }
        except Exception as e:
            logger.error(f"上传失败: {str(e)}", exc_info=True)
            return {
                "status": "error",
                "message": str(e)
            }