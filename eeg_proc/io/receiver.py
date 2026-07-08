# eeg_proc/io/receiver.py
import base64
import numpy as np
from pathlib import Path
import mne
import warnings
from pydantic import ValidationError
from ..models.eeg_model import WechatEEGInput
import logging

class EEGReceiver:
    def __init__(self, max_sampling_rate=2000):
        self.max_sampling_rate = max_sampling_rate
        self.logger = logging.getLogger('EEGReceiver')

    def parse_wechat_input(self, json_data: dict) -> mne.io.BaseRaw:
        """微信数据解析增强版"""
        try:
            validated = WechatEEGInput(**json_data)

            eeg_bytes = base64.b64decode(validated.eeg_data)
            expected_elements = len(eeg_bytes) // np.dtype(np.float32).itemsize
            if expected_elements % validated.n_channels != 0:
                raise ValueError(
                    f"数据长度{len(eeg_bytes)}字节无法整除{validated.n_channels}个通道"
                )

            eeg_array = np.frombuffer(eeg_bytes, dtype=np.float32)
            eeg_array = eeg_array.reshape(validated.n_channels, -1)

            info = mne.create_info(
                ch_names=validated.channel_names,
                sfreq=validated.sampling_rate,
                ch_types='eeg'
            )

            with warnings.catch_warnings():
                warnings.simplefilter("once")
                try:
                    montage = mne.channels.make_standard_montage('standard_1020')
                    info.set_montage(
                        montage,
                        on_missing='warn',
                        match_case=False
                    )
                except ValueError as e:
                    warnings.warn(f"Montage配置异常: {str(e)}", category=UserWarning)

            return mne.io.RawArray(eeg_array, info)

        except ValidationError as e:
            error_details = [f"{err['loc'][0]}: {err['msg']}" for err in e.errors()]
            raise ValueError(f"输入校验失败: {'; '.join(error_details)}") from e

    def load_edf(self, edf_path: str, channel_map: dict = None, max_sfreq: float = None) -> mne.io.BaseRaw:
        """增强EDF加载器"""
        try:
            max_sfreq = max_sfreq or self.max_sampling_rate
            edf_path = Path(edf_path).resolve()
            if not edf_path.exists():
                raise FileNotFoundError(f"EDF文件不存在: {edf_path.as_posix()!r}")

            raw = mne.io.read_raw_edf(
                edf_path,
                preload=True,
                exclude=['Misc'],
                infer_types=True
            )

            if channel_map:
                self._validate_channels(raw, channel_map)
                raw.rename_channels({
                    str(k).strip().upper(): str(v).strip().upper()
                    for k, v in channel_map.items()
                })

            current_sfreq = raw.info['sfreq']
            if current_sfreq > max_sfreq:
                safe_cutoff = max_sfreq * 0.45
                raw.filter(
                    l_freq=None,
                    h_freq=safe_cutoff,
                    method='fir',
                    fir_window='hamming'
                )
                raw.resample(max_sfreq, npad='auto')

            return raw.pick_types(eeg=True).load_data()

        except Exception as e:
            if isinstance(e, FileNotFoundError):
                raise
            raise RuntimeError(f"EDF加载失败: {str(e)}") from e

    def load_fif(self, fif_path: str, allow_missing: bool = False) -> mne.io.BaseRaw:
        """增强FIF文件加载"""
        try:
            fif_path = Path(fif_path).resolve()
            if not fif_path.exists() and not allow_missing:
                raise FileNotFoundError(f"FIF文件不存在: {fif_path}")

            raw = mne.io.read_raw_fif(
                fif_path,
                preload=True,
                verbose='error'
            )

            if not raw.info['lowpass']:
                self.logger.warning("未检测到低通滤波记录")

            return raw.pick_types(eeg=True).load_data()

        except Exception as e:
            raise RuntimeError(f"FIF加载失败: {str(e)}") from e

    def _validate_channels(self, raw, channel_map: dict):
        """通道映射校验增强版"""
        available = {ch.strip().upper() for ch in raw.ch_names}
        required = {str(k).strip().upper() for k in channel_map.keys()}

        if missing := required - available:
            available_list = '\n'.join(f"- {ch}" for ch in raw.ch_names)
            raise ValueError(
                f"缺少{len(missing)}个映射通道:\n"
                f"缺失通道: {missing}\n"
                f"当前文件包含{len(raw.ch_names)}个通道:\n{available_list}"
            )