# eeg_proc/processing/preprocessor.py
from datetime import datetime
import mne
import logging
from mne.preprocessing import ICA
from typing import Optional, Dict, Any
from .artifact_detection import compute_bad_channels
from .quality_metrics_v2 import calculate_snr
import numpy as np
logger = logging.getLogger('eeg_pipeline')


class MNEPipeline:
    """符合MNE标准的预处理流水线（支持方法链调用）"""

    DEFAULT_PARAMS = {
        'filter': {
            'l_freq': 1.0,
            'h_freq': 40.0,
            'method': 'fir',
            'l_trans_bandwidth': 0.5,  # 低频过渡带宽(Hz)
            'h_trans_bandwidth': 0.5,  # 高频过渡带宽(Hz)
            'phase': 'zero-double',
            'fir_window': 'hann'
        },
        'ica': {
            'method': 'infomax',
            'random_state': 42,
            'n_components': 0.99
        },
        'artifact': {
            'ecg_threshold': 0.3,
            'eog_threshold': 0.5,
            'ecg_ch_name': 'ECG',  # 兼容旧版mne
            'eog_ch_name': 'EOG'  # 新版参数名
        }
    }

    def __init__(self, raw: mne.io.BaseRaw):
        """
        :param raw: 原始数据对象
        """
        self.raw = raw.copy()
        self.ica = None
        self.processing_log = []
        self.quality_metrics = {}
        self._log_step('init', {'input_shape': self.raw.get_data().shape})
        logger.info(f"初始化MNEPipeline: {len(raw.ch_names)}通道, {raw.n_times}采样点")

    def _log_step(self, step_name: str, params: dict) -> None:
        """记录处理步骤和参数"""
        log_entry = {
            'step': step_name,
            'params': params.copy() if params else {},
            'timestamp': datetime.now().isoformat(),
            'data_hash': hash(self.raw.get_data().tobytes())
        }
        self.processing_log.append(log_entry)

    def apply_filter(self, l_freq: Optional[float] = None,
                     h_freq: Optional[float] = None, **kwargs) -> 'MNEPipeline':
        """带通滤波（兼容新旧MNE版本，优化衰减特性）"""
        params = {**self.DEFAULT_PARAMS['filter'], **kwargs}
        params.update({
            'l_freq': l_freq if l_freq is not None else params['l_freq'],
            'h_freq': h_freq if h_freq is not None else params['h_freq']
        })

        # 确保过渡带宽参数存在
        params.setdefault('l_trans_bandwidth', self.DEFAULT_PARAMS['filter']['l_trans_bandwidth'])
        params.setdefault('h_trans_bandwidth', self.DEFAULT_PARAMS['filter']['h_trans_bandwidth'])

        # 旧版兼容处理
        if 'fir_design' not in params:
            params['fir_design'] = 'firwin2'

            # 动态调整滤波器长度（核心修复）
            n_samples = self.raw.n_times
            # 计算最小滤波器长度（至少3倍周期）
            min_length = 0
            if l_freq and l_freq > 0:
                min_length = int(3 * self.raw.info['sfreq'] / l_freq)
            min_length = max(101, min_length)  # 确保最小长度

            # 计算最大滤波器长度（不超过数据长度的1/3）
            max_length = min(10000, n_samples // 2)
            max_length = min(max_length, int(n_samples * 0.1))  # 新增：限制最大长度 ≤10% 数据时长

            # 如果用户没有指定滤波器长度，则使用动态计算的值
            if 'filter_length' not in params:
                params['filter_length'] = min_length  # 使用最小长度保证滤波效果

            # 确保滤波器长度在合理范围内且为奇数
            if isinstance(params['filter_length'], (int, float)):
                filter_length = int(params['filter_length'])
                filter_length = max(min_length, min(filter_length, max_length))
                filter_length = filter_length // 2 * 2 + 1  # 确保是奇数
                params['filter_length'] = filter_length

            n_samples = self.raw.n_times

            # 1. 计算最小长度要求
            min_cycles = 3  # 至少3个周期
            min_length_by_freq = 0

            if l_freq and l_freq > 0:
                min_length_by_freq = int(min_cycles * self.raw.info['sfreq'] / l_freq)
            elif h_freq and h_freq > 0:
                min_length_by_freq = int(min_cycles * self.raw.info['sfreq'] / h_freq)

            # 2. 确定FIR滤波器长度
            if 'filter_length' not in params:
                # 使用频率要求的最小长度，但不超过数据长度的70%
                filter_length = min(min_length_by_freq, int(n_samples * 0.7)) if min_length_by_freq > 0 else 101
            else:
                filter_length = params['filter_length']

            # 3. 确保长度合理
            filter_length = max(101, min(filter_length, min(10000, n_samples)))

            # 4. 如果滤波器长度 > 数据长度或数据长度 < 2秒，使用IIR滤波器
            if filter_length > n_samples or n_samples < int(2 * self.raw.info['sfreq']):
                # 使用IIR滤波器（更适合短数据）
                logger.warning(f"使用IIR滤波: 数据={n_samples}点 < 滤波器要求={filter_length}点")
                self.raw.filter(
                    l_freq=l_freq,
                    h_freq=h_freq,
                    method='iir',
                    iir_params=dict(order=4, ftype='butter'),
                    verbose=False
                )
                return self

            # 5. 确保是奇数（FIR滤波器要求）
            filter_length = filter_length // 2 * 2 + 1
            params['filter_length'] = filter_length

            logger.info(
                f"滤波器: 长度={filter_length}点 (数据={n_samples}点, "
                f"频率要求={min_length_by_freq}点)"
            )

            # ===== 安全处理：当滤波器长度 > 数据长度时 =====
            if filter_length > n_samples:
                # 1. 使用分段滤波（更先进的方法）
                try:
                    logger.warning(f"使用分段滤波: {filter_length}点 > {n_samples}点")
                    self.raw.filter(
                        l_freq=l_freq,
                        h_freq=h_freq,
                        method='fir',
                        filter_length=filter_length,
                        fir_design='firwin2',
                        phase='zero',
                        fir_window='hann',
                        pad='reflect_limited',  # 更好的边界处理
                        verbose=False
                    )
                    return self
                except Exception as e:
                    logger.error(f"分段滤波失败: {str(e)}")

                # 2. 回退方案：使用数据长度
                filter_length = n_samples if n_samples % 2 == 1 else n_samples - 1
                logger.warning(f"回退到最小滤波器长度: {filter_length}点")

            # ===== 智能滤波策略选择 =====
            min_required_samples = int(5 * self.raw.info['sfreq'] / min(l_freq, 1) if l_freq else 1000)

            if n_samples < min_required_samples:
                # 使用IIR滤波器（更适合短数据）
                logger.warning(f"使用IIR滤波: 数据={n_samples}点 < 要求={min_required_samples}点")
                self.raw.filter(
                    l_freq=l_freq,
                    h_freq=h_freq,
                    method='iir',
                    iir_params=dict(order=4, ftype='butter'),
                    verbose=False
                )
                return self

        # 记录使用的滤波器参数
        self._log_step('filter', params)

        # 应用滤波器（添加过渡带宽参数）
        self.raw.filter(
            l_freq=params['l_freq'],
            h_freq=params['h_freq'],
            method=params['method'],
            phase=params['phase'],
            fir_window=params['fir_window'],
            fir_design=params['fir_design'],
            filter_length=params['filter_length'],
            l_trans_bandwidth=params['l_trans_bandwidth'],  # 添加低频过渡带宽
            h_trans_bandwidth=params['h_trans_bandwidth'],  # 添加高频过渡带宽
            verbose=False
        )
        return self

    def set_reference(self, ref_channels: str = 'average') -> 'MNEPipeline':
        """智能重参考设置（修复应用问题）"""
        try:
            # 确保数据已加载
            if not self.raw.preload:
                logger.info("预加载数据以应用参考")
                self.raw.load_data()

            if ref_channels == 'average':
                logger.info("应用平均参考")

                # 创建数据副本以避免修改原始数据
                raw_copy = self.raw.copy()

                # 应用平均参考
                raw_copy.set_eeg_reference(ref_channels='average')
                raw_copy.apply_proj()  # 显式应用投影

                # 更新原始数据
                self.raw = raw_copy
                logger.info("平均参考已应用")
            else:
                logger.info(f"应用{ref_channels}参考")
                self.raw.set_eeg_reference(ref_channels=ref_channels)

            self._log_step('rereference', {'method': ref_channels})
            return self
        except Exception as e:
            logger.error(f"重参考设置失败: {str(e)}", exc_info=True)
            raise RuntimeError(f"重参考设置错误: {str(e)}")

    def run_ica(self, n_components: Optional[float] = None,
                method: Optional[str] = None, **kwargs) -> 'MNEPipeline':
        """增强版ICA分解（添加空成分检查）"""
        params = {**self.DEFAULT_PARAMS['ica'], **kwargs}
        params.update({
            'n_components': n_components or params['n_components'],
            'method': method or params['method']
        })

        min_samples = int(2 * self.raw.info['sfreq'])  # 至少2秒数据
        if self.raw.n_times < min_samples:
            logger.warning(f"数据不足({self.raw.n_times}采样点 < 2秒)，跳过ICA处理")
            self._log_step('ica', {'status': 'skipped (insufficient data)'})
            return self

        self.ica = ICA(
            n_components=params['n_components'],
            method=params['method'],
            random_state=params['random_state'],
            max_iter=params.get('max_iter', 500)
        )

        self.ica.fit(
            self.raw,
            reject_by_annotation=True,
            verbose='warning'  # 显示重要警告
        )

        # ===== 关键修复：检查ICA是否成功生成成分 =====
        if not hasattr(self.ica, 'n_components_') or self.ica.n_components_ == 0:
            logger.error("ICA拟合失败: 未生成任何独立成分。可能是数据不足或数据质量问题。")
            self.ica = None  # 标记ICA失效，避免后续操作
        else:
            logger.info(f"ICA拟合成功，生成{self.ica.n_components_}个成分")

        self._log_step('ica', params)
        return self

    def detect_artifacts(self, ecg_threshold: Optional[float] = None,
                         eog_threshold: Optional[float] = None, **kwargs) -> 'MNEPipeline':
        """多模态伪迹检测"""
        if self.ica is None or (hasattr(self.ica, 'n_components_') and self.ica.n_components_ == 0):
            logger.warning("ICA未初始化或拟合失败，跳过伪迹检测")
            self._log_step('artifact_detect', {'status': 'skipped (no ICA model)'})
            return self

        params = {**self.DEFAULT_PARAMS['artifact'], **kwargs}

        # ECG检测（多版本兼容）
        ecg_components = []
        if params['ecg_ch_name'] in self.raw.ch_names:
            try:
                ecg_components, _ = self.ica.find_bads_ecg(
                    self.raw,
                    threshold=params['ecg_threshold'],
                    ch_name=params['ecg_ch_name']
                )
            except TypeError:  # 旧版参数名回退
                ecg_components, _ = self.ica.find_bads_ecg(
                    self.raw,
                    threshold=params['ecg_threshold'],
                    channel=params['ecg_ch_name']
                )

        # EOG检测（同理）
        eog_components = []
        if params['eog_ch_name'] in self.raw.ch_names:
            try:
                eog_components, _ = self.ica.find_bads_eog(
                    self.raw,
                    threshold=params['eog_threshold'],
                    ch_name=params['eog_ch_name']
                )
            except TypeError:  # 旧版参数名回退
                eog_components, _ = self.ica.find_bads_eog(
                    self.raw,
                    threshold=params['eog_threshold'],
                    channel=params['eog_ch_name']
                )

        self.ica.exclude = list(set(ecg_components + eog_components))
        self._log_step('artifact_detect', {
            'excluded_components': self.ica.exclude,
            'artifact_params': params
        })
        return self

    def apply_clean(self) -> 'MNEPipeline':
        """数据清洗综合处理"""
        # 应用ICA清洗
        if self.ica and len(self.ica.exclude) > 0:
            self.ica.apply(self.raw)

        # 注释掉坏道处理（实时处理中保留所有通道）
        # bad_channels = compute_bad_channels(self.raw)
        # if bad_channels:
        #     self.raw.drop_channels(bad_channels)

        # 质量评估
        self._calculate_quality_metrics()

        self._log_step('final_clean', {
            'remaining_channels': self.raw.info['ch_names'],
            'quality_metrics': self.quality_metrics
        })
        return self

    def apply_standard_pipeline(self, l_freq=1, h_freq=40, ica_method='fastica') -> 'MNEPipeline':
        """支持参数传递的标准化流程（带详细日志和错误处理）"""
        try:
            logger.info("开始标准预处理流程")

            # ===== 关键修复 1：空数据防护 =====
            if self.raw.n_times == 0:
                error_msg = "输入数据为空 (n_times=0)，无法执行处理流程"
                logger.critical(error_msg)
                raise ValueError(error_msg)

            # ===== 关键修复 2：状态快照增强 =====
            orig_state = {
                'ch_names': self.raw.ch_names.copy(),
                'n_times': self.raw.n_times,
                'sfreq': self.raw.info['sfreq'],
                'data_hash': hash(self.raw._data.tobytes())  # 增加数据哈希校验
            }
            logger.info(f"原始数据状态: {orig_state['n_times']}采样点, {len(orig_state['ch_names'])}通道")

            # 参数验证（保持原有逻辑）
            if l_freq is not None and h_freq is not None and l_freq >= h_freq:
                error_msg = f"滤波参数无效: 低通{l_freq}Hz ≥ 高通{h_freq}Hz"
                logger.error(error_msg)
                raise ValueError(error_msg)

            min_samples_ica = int(self.raw.info['sfreq'] * 2)  # 至少2秒数据
            skip_ica = False

            if self.raw.n_times < min_samples_ica:
                logger.warning(f"数据量不足({self.raw.n_times}采样点 < 2秒)，跳过ICA处理")
                skip_ica = True

                # ===== 执行处理链 =====
                logger.info(">> 步骤1: 应用滤波器")
                self.apply_filter(l_freq=l_freq, h_freq=h_freq)

                logger.info(">> 步骤2: 设置参考电极")
                self.set_reference()

                if not skip_ica:
                    logger.info(f">> 步骤3: 执行ICA ({ica_method})")
                    self.run_ica(method=ica_method)

                    logger.info(">> 步骤4: 伪迹检测")
                    self.detect_artifacts()
                else:
                    logger.info(">> 跳过步骤3-4 (ICA和伪迹检测)")

                logger.info(">> 步骤5: 应用清理")
                self.apply_clean()

            # ===== 关键修复 3：状态变化验证 =====
            current_state = {
                'n_times': self.raw.n_times,
                'n_channels': len(self.raw.ch_names)
            }
            if current_state['n_times'] == 0:
                logger.error(f"处理后数据为空! 原始数据: {orig_state['n_times']}采样点")
                raise RuntimeError("处理流程导致数据丢失")

            logger.info(
                f"预处理完成: 通道 {len(orig_state['ch_names'])}→{current_state['n_channels']}, "
                f"采样点 {orig_state['n_times']}→{current_state['n_times']}, "
                f"采样率 {orig_state['sfreq']}Hz"
            )
            return self

        except Exception as e:
            # ===== 增强错误诊断 =====
            current_step = self.processing_log[-1]['step'] if self.processing_log else "未知"
            error_details = {
                'step': current_step,
                'error_type': type(e).__name__,
                'error_msg': str(e),
                'data_state': f"{len(self.raw.ch_names)}通道/{self.raw.n_times}采样点"
            }
            logger.critical(f"预处理失败: {error_details}")

            # 记录关键参数
            logger.debug(f"滤波器参数: l_freq={l_freq}, h_freq={h_freq}")
            logger.debug(f"ICA方法: {ica_method}")

            raise RuntimeError(f"预处理失败于[{current_step}]: {str(e)}") from e

    def _calculate_quality_metrics(self) -> None:
        """综合质量评估"""
        self.quality_metrics = {
            'snr_db': calculate_snr(self.raw),
            'bad_channels_ratio': len(compute_bad_channels(self.raw)) / len(self.raw.ch_names),
            'ica_artifact_ratio': len(self.ica.exclude) / self.ica.n_components_ if self.ica else 0,
            'remaining_samples': self.raw.n_times,
            'effective_sfreq': self.raw.info['sfreq']
        }

    def get_processed_data(self) -> mne.io.BaseRaw:
        """获取处理后的数据副本"""
        return self.raw.copy()

    def get_processing_report(self) -> Dict[str, Any]:
        """生成处理报告，确保所有类型可序列化"""
        # 转换numpy类型为Python原生类型
        metrics = {
            'snr_db': float(self.quality_metrics['snr_db']),
            'bad_channels_ratio': float(self.quality_metrics['bad_channels_ratio']),
            'ica_artifact_ratio': float(self.quality_metrics['ica_artifact_ratio']),
            'remaining_samples': int(self.quality_metrics['remaining_samples']),
            'effective_sfreq': float(self.quality_metrics['effective_sfreq'])
        }

        # 转换日志中的numpy类型
        serializable_log = []
        for entry in self.processing_log:
            # 深度复制并转换
            entry_copy = entry.copy()
            if 'params' in entry_copy:
                for key, value in entry_copy['params'].items():
                    # 修复1: 使用安全的NumPy类型检查方法
                    try:
                        # 检查是否是NumPy标量类型
                        if hasattr(value, 'item') and callable(getattr(value, 'item', None)):
                            # 处理 NumPy 标量类型
                            entry_copy['params'][key] = value.item()
                        # 修复2: 使用NumPy的issubdtype替代直接类型检查
                        elif isinstance(value, np.generic):
                            if np.issubdtype(value.dtype, np.integer):
                                entry_copy['params'][key] = int(value)
                            elif np.issubdtype(value.dtype, np.floating):
                                entry_copy['params'][key] = float(value)
                        # 处理标准Python类型
                        elif isinstance(value, (int, float)):
                            # 原生类型不需要转换
                            pass
                        else:
                            try:
                                # 尝试直接转换为 Python 原生类型
                                # 改进浮点数检测逻辑
                                if isinstance(value, str) and '.' in value:
                                    entry_copy['params'][key] = float(value)
                                else:
                                    entry_copy['params'][key] = int(value)
                            except (ValueError, TypeError):
                                # 无法转换则保留原值（可能是字符串或其他可序列化类型）
                                pass
                    except Exception as e:
                        # 添加错误日志记录以便调试
                        logger.warning(f"类型转换错误: 键={key}, 值={value}(类型:{type(value)}), 错误: {str(e)}")
                        # 保持原值，确保函数继续执行
            serializable_log.append(entry_copy)

        return {
            'log': serializable_log,
            'metrics': metrics,
            'final_info': {
                'n_channels': int(len(self.raw.ch_names)),
                'sfreq': float(self.raw.info['sfreq']),
                'duration': float(self.raw.times[-1])
            }
        }