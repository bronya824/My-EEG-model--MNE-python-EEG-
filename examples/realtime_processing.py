# examples/realtime_processing.py
import numpy as np
import mne
import base64
import warnings
from eeg_proc.io.receiver import EEGReceiver
from eeg_proc.models.eeg_model import AliyunOutputModel
from eeg_proc.processing.preprocessor import MNEPipeline
from pydantic import ValidationError
import json


class RealTimeProcessor:
    def __init__(self, buffer_duration=5.0, l_freq=1.0, h_freq=40.0):  # ✅ 改进1：缓冲区时长从30秒改为1秒
        self.buffer = None
        self.ica_model = None  # 新增ICA模型缓存
        self.frame_count = 0  # 新增帧计数器
        self.skip_ica_counter = 0
        self.min_filter_samples = None  # 添加初始化
        self.min_filter_duration = max(3.0, 3 / l_freq) if l_freq > 0 else 3.0
        self.processing_params = {
            'l_freq': l_freq,
            'h_freq': h_freq,
            'ica_method': 'infomax',
            'buffer_duration': buffer_duration,  # 适应测试数据长度[3](@ref)
            'retain_duration': 2.0  # 新增：保留历史数据时长
        }
        print(f"初始化RealTimeProcessor，缓冲区时长={buffer_duration}秒")  # ✅ 改进4：添加调试输出

    def _initialize_buffer(self, n_channels: int, sfreq: float):
        """初始化缓冲区（使用match_case避免大小写转换）"""
        # 在初始化缓冲区时设置 min_filter_samples
        self.min_filter_samples = int(self.min_filter_duration * sfreq)
        try:
            # 创建原始通道名（保留大写）
            ch_names = [f"CH{i + 1}" for i in range(n_channels)]

            # 创建空缓冲区
            self.buffer = mne.io.RawArray(
                np.empty((n_channels, 0)),
                mne.create_info(
                    ch_names=ch_names,
                    sfreq=sfreq,
                    ch_types='eeg'
                )
            )

            # ✅ 改进2：直接使用match_case忽略大小写[2](@ref)
            montage = mne.channels.make_standard_montage('standard_1005')
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.buffer.set_montage(montage, match_case=False, on_missing='ignore')

            # 初始化处理管道
            self.expected_channels = n_channels
            self.base_sfreq = sfreq
            print(f"缓冲区初始化完成: {n_channels}通道, {sfreq}Hz采样率")  # ✅ 改进4

        except Exception as e:
            self._reset_buffer()
            raise RuntimeError(f"Buffer初始化失败: {str(e)}") from e

    def process_chunk(self, chunk_data: np.ndarray, sfreq: float):
        """处理实时数据块（增强鲁棒性）"""
        if chunk_data.size == 0 or sfreq <= 0:
            print("警告: 无效数据块，跳过处理")
            return
        if self.buffer and self.buffer.n_times < self.min_filter_samples:
            # 小数据块直接合并到缓冲区，不单独处理
            self._append_to_buffer(chunk_data, sfreq)
            return
        try:
            if chunk_data.size == 0:
                print("警告: 收到空数据块，跳过处理")  # ✅ 改进4
                return

            # 统一数据维度 (n_samples, n_channels)
            if chunk_data.ndim == 1:
                chunk_data = chunk_data.reshape(-1, 1)

            n_channels = chunk_data.shape[1]

            # 动态初始化检查
            if self.buffer is None:
                print(f"初始化缓冲区: 检测到{n_channels}通道")  # ✅ 改进4
                self._initialize_buffer(n_channels, sfreq)
            elif n_channels != self.expected_channels:
                raise ValueError(f"通道数不匹配！预期: {self.expected_channels}，实际: {n_channels}")

            # 数据转换
            scaled_data = chunk_data.T  # (n_channels, n_samples)

            # 创建临时Raw对象
            temp_raw = mne.io.RawArray(
                scaled_data,
                mne.create_info(
                    ch_names=self.buffer.ch_names,
                    sfreq=sfreq,
                    ch_types='eeg'
                )
            )

            # 追加数据到缓冲区
            self.buffer.append(temp_raw)
            buffer_duration = self.buffer.n_times / sfreq
            print(f"当前缓冲区时长: {buffer_duration:.1f}s")

            # ===== 关键修复：确保总是处理数据 =====
            # 第一层：达到基本缓冲区时长
            if buffer_duration >= self.processing_params['buffer_duration']:
                # 第二层：检查滤波器要求（仅作记录，不阻止处理）
                filter_ready = buffer_duration >= self.min_filter_duration
                status = "充足" if filter_ready else f"不足({buffer_duration:.1f}s < {self.min_filter_duration:.1f}s)"
                print(f"滤波器条件: {status}")

                # 执行处理（无论是否满足滤波器要求）
                print(f"缓冲区满，开始处理...")
                processed = self._apply_processing()
                self._upload_to_aliyun(processed)

                # 裁剪缓冲区
                retain_samples = int(
                    self.processing_params['retain_duration'] * sfreq
                )
                retain_samples = min(retain_samples, self.buffer.n_times)
                data = self.buffer.get_data()[:, -retain_samples:]
                self.buffer = mne.io.RawArray(data, self.buffer.info)
                print(f"裁剪后缓冲区: {retain_samples / sfreq:.1f}s")

        except Exception as e:
            print(f"实时处理失败: {type(e).__name__}: {str(e)}")
            self._reset_buffer()

    def _apply_processing(self) -> mne.io.BaseRaw:
        """执行处理流程（绑定最新缓冲区）"""
        if self.buffer is None or self.buffer.info is None:
            print("警告: 缓冲区无效，跳过处理")
            return None

            # 确保缓冲区有足够数据
        if self.buffer.n_times < 10:  # 至少10个样本
            print("警告: 缓冲区数据不足，跳过处理")
            return self.buffer.copy()
        print("应用处理: 滤波->参考->ICA->伪迹去除")

        min_samples = int(self.min_filter_duration * self.buffer.info['sfreq'])

        if self.buffer.n_times < min_samples:
            print(f"跳过处理: 数据不足{min_samples}点")
            return self.buffer.copy()

        self.frame_count += 1

        # 使用当前缓冲区副本
        pipeline = MNEPipeline(raw=self.buffer.copy())

        # 应用滤波和参考
        pipeline = (
            pipeline
            .apply_filter(
                l_freq=self.processing_params['l_freq'],
                h_freq=self.processing_params['h_freq']
            )
            .set_reference()
        )

        # 获取采样率
        sfreq = self.buffer.info['sfreq']  # ✅ 修复：定义sfreq变量

        # ICA处理策略
        min_ica_samples = int(1.5 * sfreq)  # 1.5秒最低要求
        skip_ica = self.buffer.n_times < min_ica_samples

        if not skip_ica or self.skip_ica_counter >= 5:
            if self.ica_model is None or self.frame_count % 30 == 0:
                print("更新ICA模型...")
                pipeline = pipeline.run_ica(
                    method=self.processing_params['ica_method']
                )
                self.ica_model = pipeline.ica
                self.skip_ica_counter = 0
            else:
                print("应用缓存ICA模型...")
                pipeline.ica = self.ica_model

            pipeline = pipeline.detect_artifacts().apply_clean()
        else:
            self.skip_ica_counter += 1
            # ✅ 修复：使用定义的sfreq变量
            print(f"跳过ICA（数据不足{self.buffer.n_times / sfreq:.1f}s < {min_ica_samples / sfreq:.1f}s）")

        return pipeline.get_processed_data()
    def _upload_to_aliyun(self, processed: mne.io.BaseRaw):
        """数据上传（改为特征上传）"""
        try:
            # ✅ 提取特征并转换为Base64字符串
            features = self._extract_features(processed)
            total_features = features.pop('_feature_count', 0)
            encoded = base64.b64encode(json.dumps(features).encode()).decode()

            # ✅ 创建符合AliyunOutputModel的元数据
            metadata = {
                'sfreq': float(processed.info['sfreq']),
                'ch_names': processed.ch_names,
                'n_samples': int(processed.n_times),
                'feature_version': "1.0"  # 新增特征版本标识
            }

            # ✅ 使用Base64编码的特征数据创建模型实例
            output = AliyunOutputModel(
                processed_data=encoded,  # 传入Base64字符串
                metadata=metadata
            )
            result = output.upload()

            if result["status"] == "success":
                print(f"上传成功: {total_features}个特征值")
            else:
                print(f"上传失败: {result.get('message', '未知错误')}")

        except ValidationError as e:
            print(f"阿里云数据校验失败: {str(e)}")
        except Exception as e:
            print(f"上传过程中发生意外错误: {str(e)}")

    def _extract_features(self, raw: mne.io.BaseRaw) -> dict:
        """提取时域/频域特征"""
        # 确保数据有效
        if raw is None or raw.info is None or not raw.ch_names:
            print("警告: 无效的EEG数据，跳过特征提取")
            return {'_feature_count': 0}

        n_channels = len(raw.ch_names)
        data = raw.get_data()
        features = {}

        # 1. 时域特征（总是计算）
        features['mean'] = np.mean(data, axis=1).tolist()
        features['std'] = np.std(data, axis=1).tolist()

        # 2. 频域特征
        freqs = np.array([1, 4, 8, 13, 30])

        # 确保有足够数据计算PSD
        min_psd_samples = int(1.5 * raw.info['sfreq'])  # 至少1.5秒数据
        band_features = []

        if raw.n_times >= min_psd_samples:
            try:
                # 使用更可靠的PSD计算方法
                psd = raw.compute_psd(
                    method='multitaper',
                    fmin=1,
                    fmax=30,
                    n_jobs=1,  # 避免并行问题
                    verbose=False
                )

                for i in range(len(freqs) - 1):
                    # 获取频段功率
                    band_data = psd.get_data(fmin=freqs[i], fmax=freqs[i + 1])

                    # 确保有有效数据
                    if band_data.size > 0:
                        band_power = band_data.mean(axis=1).tolist()
                    else:
                        band_power = [0.0] * n_channels

                    # 验证结果长度
                    if len(band_power) != n_channels:
                        print(f"警告: 频段功率数量({len(band_power)})不等于通道数({n_channels})")
                        band_power = [0.0] * n_channels

                    features[f'band_{freqs[i]}-{freqs[i + 1]}'] = band_power
                    band_features.extend(band_power)
            except Exception as e:
                print(f"PSD计算错误: {str(e)}")
                # 出错时填充默认值
                for i in range(len(freqs) - 1):
                    features[f'band_{freqs[i]}-{freqs[i + 1]}'] = [0.0] * n_channels
                    band_features.extend([0.0] * n_channels)
        else:
            # 数据不足，填充默认值
            for i in range(len(freqs) - 1):
                features[f'band_{freqs[i]}-{freqs[i + 1]}'] = [0.0] * n_channels
                band_features.extend([0.0] * n_channels)

        # 3. 计算总特征数（确保一致性）
        total_features = 2 * n_channels + len(band_features)
        features['_feature_count'] = total_features

        # 调试输出
        print(f"特征统计: 通道={n_channels}, 时域={2 * n_channels}, 频域={len(band_features)}, 总计={total_features}")

        return features
    def _reset_buffer(self):
        """安全重置缓冲区"""
        print("重置缓冲区")  # ✅ 改进4
        self.buffer = None
        self.pipeline = None
        self.expected_channels = None
        self.base_sfreq = None

    def _append_to_buffer(self, chunk_data: np.ndarray, sfreq: float):
        """追加数据到缓冲区而不触发处理"""
        if chunk_data.size == 0:
            return

        # 统一数据维度
        if chunk_data.ndim == 1:
            chunk_data = chunk_data.reshape(-1, 1)

        n_channels = chunk_data.shape[1]

        # 数据转换
        scaled_data = chunk_data.T  # (n_channels, n_samples)

        # 创建临时Raw对象
        temp_raw = mne.io.RawArray(
            scaled_data,
            mne.create_info(
                ch_names=self.buffer.ch_names,
                sfreq=sfreq,
                ch_types='eeg'
            )
        )

        # 追加数据到缓冲区
        self.buffer.append(temp_raw)
        print(f"追加数据: {chunk_data.shape[0]}点, 总时长: {self.buffer.n_times / sfreq:.1f}s")


def start_processing():
    """测试运行函数"""
    receiver = EEGReceiver()

    # 构造符合大写通道名称要求的测试数据
    mock_data = {
        "device_id": "eeg-2024",
        "user_id": "usr-20240423",
        "timestamp": "2024-04-23T14:30:00+08:00",
        "sampling_rate": 1000,
        "n_channels": 3,
        "channel_names": ["FP1", "F3", "C3"],  # 标准大写名称
        "eeg_data": base64.b64encode(np.random.randn(3 * 9000).astype(np.float32)).decode()
    }

    try:
        # 解析输入数据
        raw = receiver.parse_wechat_input(mock_data)
        print("成功解析数据：")
        print(f"通道数：{len(raw.ch_names)}")
        print(f"采样率：{raw.info['sfreq']}Hz")
        print(f"数据时长：{raw.times[-1]:.1f}秒")

        # 初始化处理器
        processor = RealTimeProcessor()

        import time
        process_times = []
        import psutil
        process = psutil.Process()

        # 模拟实时数据输入
        chunk_size = 100  # 每次处理100个样本点
        for i in range(0, raw.n_times, chunk_size):
            start_time = time.time()
            chunk = raw.get_data()[:, i:i + chunk_size].T
            processor.process_chunk(chunk, raw.info['sfreq'])

            # 记录处理时间
            elapsed = time.time() - start_time
            process_times.append(elapsed)

            # 每5帧报告一次资源使用
            if i // chunk_size % 5 == 0:
                mem_usage = process.memory_info().rss / (1024 ** 2)
                print(f"帧 {i // chunk_size}: 处理延迟={elapsed * 1000:.1f}ms, 内存={mem_usage:.1f}MB")

            # 添加缓冲区状态检查
            if processor.buffer:
                print(f"当前缓冲区状态: {processor.buffer.n_times}采样点")

    except ValueError as e:
        print(f"处理失败：{str(e)}")


if __name__ == "__main__":
    start_processing()