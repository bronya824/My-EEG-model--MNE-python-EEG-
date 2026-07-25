# eeg_proc/processing/preprocessor.py
from datetime import datetime
import mne
import logging
from mne.preprocessing import ICA
from typing import Optional, Dict, Any
from .artifact_detection import compute_bad_channels, classify_ica_components
from .quality_metrics_v2 import calculate_snr, compute_depression_markers
import numpy as np

logger = logging.getLogger('eeg_pipeline')


class MNEPipeline:
    """符合MNE标准的预处理流水线（抑郁症检测优化版）"""

    DEFAULT_PARAMS = {
        'filter': {
            'l_freq': 0.1,
            'h_freq': 40.0,
            'method': 'fir',
            'l_trans_bandwidth': 0.5,
            'h_trans_bandwidth': 0.5,
            'phase': 'zero-double',
            'fir_window': 'hann'
        },
        'notch': {
            'freqs': 50,
            'method': 'fir',
            'filter_length': '10s'
        },
        'ica': {
            'method': 'infomax',
            'random_state': 42,
            'n_components': 0.99
        },
        'artifact': {
            'ecg_threshold': 0.3,
            'eog_threshold': 0.5,
            'ecg_ch_name': 'ECG',
            'eog_ch_name': 'EOG',
            'use_spectral_classification': True
        }
    }

    def __init__(self, raw: mne.io.BaseRaw):
        self.raw = raw.copy()
        self.ica = None
        self.processing_log = []
        self.quality_metrics = {}
        self.depression_markers = {}
        self._log_step('init', {'input_shape': self.raw.get_data().shape})
        logger.info(f"初始化MNEPipeline: {len(raw.ch_names)}通道, {raw.n_times}采样点")

    def _log_step(self, step_name: str, params: dict) -> None:
        log_entry = {
            'step': step_name,
            'params': params.copy() if params else {},
            'timestamp': datetime.now().isoformat(),
            'data_hash': hash(self.raw.get_data().tobytes())
        }
        self.processing_log.append(log_entry)

    # ========== 修复：自动幅度归一化（正确处理 V/µV 单位） ==========
    def auto_scale(self, target_mad: float = 50.0, max_scale: float = 1e6) -> 'MNEPipeline':
        """
        自动幅度归一化 — 无外部增益参数依赖。
        MNE raw.get_data() 返回单位为 Volts，需先转为 µV 评估。
        """
        data = self.raw.get_data()  # 单位: Volts
        # 转为 µV 进行稳健统计
        data_uv = data * 1e6
        median_uv = float(np.median(data_uv))
        mad_uv = float(np.median(np.abs(data_uv - median_uv)))

        # 如果已经在合理范围，跳过
        if 5.0 <= mad_uv <= 200.0:
            logger.info(f"幅度正常 (MAD={mad_uv:.2f} µV)，跳过auto_scale")
            self._log_step('auto_scale', {'status': 'skipped', 'original_mad_uv': mad_uv})
            return self

        scale = target_mad / (mad_uv + 1e-12)
        scale = float(np.clip(scale, 1.0 / max_scale, max_scale))

        # 直接在 Volts 数据上应用缩放（减去中位数消除DC，再乘scale）
        self.raw._data = (data - np.median(data)) * scale

        logger.warning(
            f"auto_scale执行: 原median={np.median(data):.2e} V, "
            f"MAD={mad_uv:.2e} µV, scale={scale:.2e}, 目标MAD={target_mad:.2f} µV"
        )
        self._log_step('auto_scale', {
            'original_median_v': float(np.median(data)),
            'original_mad_uv': mad_uv,
            'scale_factor': scale,
            'target_mad_uv': target_mad
        })
        return self
    # ========== 修复结束 ==========

    def apply_filter(self, l_freq: Optional[float] = None,
                     h_freq: Optional[float] = None, **kwargs) -> 'MNEPipeline':
        """带通滤波（兼容新旧MNE版本，优化衰减特性）"""
        params = {**self.DEFAULT_PARAMS['filter'], **kwargs}
        params.update({
            'l_freq': l_freq if l_freq is not None else params['l_freq'],
            'h_freq': h_freq if h_freq is not None else params['h_freq']
        })
        params.setdefault('l_trans_bandwidth', self.DEFAULT_PARAMS['filter']['l_trans_bandwidth'])
        params.setdefault('h_trans_bandwidth', self.DEFAULT_PARAMS['filter']['h_trans_bandwidth'])

        if 'fir_design' not in params:
            params['fir_design'] = 'firwin2'

            n_samples = self.raw.n_times
            min_cycles = 3
            min_length_by_freq = 0

            if params['l_freq'] and params['l_freq'] > 0:
                min_length_by_freq = int(min_cycles * self.raw.info['sfreq'] / params['l_freq'])
            elif params['h_freq'] and params['h_freq'] > 0:
                min_length_by_freq = int(min_cycles * self.raw.info['sfreq'] / params['h_freq'])

            if 'filter_length' not in params:
                filter_length = min(min_length_by_freq, int(n_samples * 0.7)) if min_length_by_freq > 0 else 101
            else:
                filter_length = params['filter_length']

            filter_length = max(101, min(filter_length, min(10000, n_samples)))

            if filter_length > n_samples or n_samples < int(2 * self.raw.info['sfreq']):
                logger.warning(f"使用IIR滤波: 数据={n_samples}点 < 滤波器要求={filter_length}点")
                self.raw.filter(
                    l_freq=params['l_freq'],
                    h_freq=params['h_freq'],
                    method='iir',
                    iir_params=dict(order=4, ftype='butter'),
                    verbose=False
                )
                self._log_step('filter', params)
                return self

            filter_length = filter_length // 2 * 2 + 1
            params['filter_length'] = filter_length
            logger.info(f"滤波器: 长度={filter_length}点")

            if filter_length > n_samples:
                try:
                    logger.warning(f"使用分段滤波: {filter_length}点 > {n_samples}点")
                    self.raw.filter(
                        l_freq=params['l_freq'],
                        h_freq=params['h_freq'],
                        method='fir',
                        filter_length=filter_length,
                        fir_design='firwin2',
                        phase='zero',
                        fir_window='hann',
                        pad='reflect_limited',
                        verbose=False
                    )
                    self._log_step('filter', params)
                    return self
                except Exception as e:
                    logger.error(f"分段滤波失败: {str(e)}")
                    filter_length = n_samples if n_samples % 2 == 1 else n_samples - 1

            min_required_samples = int(5 * self.raw.info['sfreq'] / min(params['l_freq'], 1) if params['l_freq'] else 1000)
            if n_samples < min_required_samples:
                logger.warning(f"使用IIR滤波: 数据={n_samples}点 < 要求={min_required_samples}点")
                self.raw.filter(
                    l_freq=params['l_freq'],
                    h_freq=params['h_freq'],
                    method='iir',
                    iir_params=dict(order=4, ftype='butter'),
                    verbose=False
                )
                self._log_step('filter', params)
                return self

        self._log_step('filter', params)
        self.raw.filter(
            l_freq=params['l_freq'],
            h_freq=params['h_freq'],
            method=params['method'],
            phase=params['phase'],
            fir_window=params['fir_window'],
            fir_design=params['fir_design'],
            filter_length=params['filter_length'],
            l_trans_bandwidth=params['l_trans_bandwidth'],
            h_trans_bandwidth=params['h_trans_bandwidth'],
            verbose=False
        )
        return self

    def apply_notch(self, freqs: Optional[float] = None, **kwargs) -> 'MNEPipeline':
        """工频陷波滤波（限制谐波数量，避免高采样率数据产生过多陷波）"""
        params = {**self.DEFAULT_PARAMS['notch'], **kwargs}
        if freqs is not None:
            params['freqs'] = freqs

        notch_freqs = params['freqs']
        if isinstance(notch_freqs, (int, float)):
            notch_freqs = [notch_freqs]

        sfreq = self.raw.info['sfreq']

        harmonics = []
        max_notch = min(sfreq / 2 - 5, 250)
        for f in notch_freqs:
            h = f
            count = 0
            while h < max_notch and count < 3:
                harmonics.append(h)
                h += f
                count += 1

        if not harmonics:
            logger.info("无需陷波（采样率过低或无有效频率）")
            return self

        logger.info(f"应用陷波滤波: {harmonics} Hz")
        try:
            self.raw.notch_filter(
                freqs=harmonics,
                method=params.get('method', 'fir'),
                filter_length=params.get('filter_length', '10s'),
                verbose=False
            )
        except Exception as e:
            logger.warning(f"陷波滤波失败: {e}，跳过")
            return self

        self._log_step('notch', {'freqs': harmonics, **params})
        return self

    def detect_bad_channels(self, interpolate: bool = True) -> 'MNEPipeline':
        """坏道检测与插值（通道数感知）"""
        bads = compute_bad_channels(self.raw)
        n_channels = len(self.raw.ch_names)

        if bads:
            logger.info(f"检测到坏道: {bads}")
            self.raw.info['bads'] = bads

            if n_channels <= 5:
                logger.warning(f"低通道模式 ({n_channels}ch): 坏道仅标记不插值: {bads}")
                self._log_step('bad_channels', {
                    'detected': bads,
                    'action': 'marked_only_low_channel',
                    'n_channels': n_channels
                })
                return self

            if interpolate and len(bads) < len(self.raw.ch_names) * 0.2:
                try:
                    if self.raw.info.get('dig') is None:
                        try:
                            self.raw.set_montage('standard_1020', match_case=False, on_missing='ignore')
                            logger.info("已自动设置 standard_1020 montage 用于插值")
                        except Exception:
                            pass

                    if self.raw.info.get('dig') is not None:
                        self.raw = self.raw.interpolate_bads(reset_bads=True)
                        logger.info(f"已插值坏道: {bads}")
                        self._log_step('bad_channels', {'detected': bads, 'action': 'interpolated'})
                    else:
                        logger.warning(f"无电极位置信息，无法插值，仅标记坏道: {bads}")
                        self._log_step('bad_channels', {'detected': bads, 'action': 'marked_only_no_dig'})
                except Exception as e:
                    logger.warning(f"坏道插值失败: {e}，仅标记")
                    self._log_step('bad_channels', {'detected': bads, 'action': 'marked_only', 'error': str(e)})
            else:
                logger.warning(f"坏道过多({len(bads)}), 仅标记不插值")
                self._log_step('bad_channels', {'detected': bads, 'action': 'marked_only'})
        else:
            self._log_step('bad_channels', {'detected': [], 'action': 'none'})
        return self

    def set_reference(self, ref_channels: str = 'average') -> 'MNEPipeline':
        """智能重参考设置（通道数感知）"""
        try:
            n_channels = len(self.raw.ch_names)

            if n_channels <= 5:
                logger.info(f"低通道模式 ({n_channels}ch): 跳过平均参考（通道高度相关，会被抵消）")
                self._log_step('rereference', {'method': 'skipped_low_channel', 'reason': 'channels_too_correlated'})
                return self

            if not self.raw.preload:
                logger.info("预加载数据以应用参考")
                self.raw.load_data()

            if ref_channels == 'average':
                logger.info("应用平均参考")
                raw_copy = self.raw.copy()
                raw_copy.set_eeg_reference(ref_channels='average')
                raw_copy.apply_proj()
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
        """增强版ICA分解（通道数感知）"""
        params = {**self.DEFAULT_PARAMS['ica'], **kwargs}
        params.update({
            'n_components': n_components or params['n_components'],
            'method': method or params['method']
        })

        n_channels = len(self.raw.ch_names)

        if n_channels <= 5:
            logger.info(f"低通道模式 ({n_channels}ch): 跳过ICA分解")
            self.ica = None
            self._log_step('ica', {'status': f'skipped (low_channel_mode: {n_channels})'})
            return self

        if n_channels < 3:
            logger.warning(f"通道数不足({n_channels} < 3)，ICA无意义，跳过")
            self._log_step('ica', {'status': f'skipped (too few channels: {n_channels})'})
            self.ica = None
            return self

        min_samples = int(2 * self.raw.info['sfreq'])
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

        try:
            self.ica.fit(self.raw, reject_by_annotation=True, verbose='warning')
        except ImportError as e:
            logger.warning(f"ICA方法 '{params['method']}' 不可用: {str(e)}，降级为 'infomax'")
            params['method'] = 'infomax'
            self.ica = ICA(n_components=params['n_components'], method='infomax',
                           random_state=params['random_state'], max_iter=params.get('max_iter', 500))
            self.ica.fit(self.raw, reject_by_annotation=True, verbose='warning')
        except RuntimeError as e:
            if 'PCA' in str(e) or 'threshold' in str(e):
                logger.warning(f"ICA不适合当前数据(PCA问题): {str(e)}，跳过ICA处理")
                self.ica = None
                self._log_step('ica', {'status': 'skipped (PCA variance issue)'})
                return self
            raise

        if not hasattr(self.ica, 'n_components_') or self.ica.n_components_ == 0:
            logger.error("ICA拟合失败: 未生成任何独立成分")
            self.ica = None
        else:
            logger.info(f"ICA拟合成功，生成{self.ica.n_components_}个成分")

        self._log_step('ica', params)
        return self

    def detect_artifacts(self, ecg_threshold: Optional[float] = None,
                         eog_threshold: Optional[float] = None, **kwargs) -> 'MNEPipeline':
        """多模态伪迹检测（通道数感知）"""
        if self.ica is None or (hasattr(self.ica, 'n_components_') and self.ica.n_components_ == 0):
            logger.warning("ICA未初始化或拟合失败，跳过伪迹检测")
            self._log_step('artifact_detect', {'status': 'skipped (no ICA model)'})
            return self

        n_channels = len(self.raw.ch_names)
        if n_channels <= 5:
            logger.info(f"低通道模式 ({n_channels}ch): 跳过ICA伪迹检测")
            self._log_step('artifact_detect', {
                'status': 'skipped (low_channel_mode)',
                'n_channels': n_channels
            })
            return self

        params = {**self.DEFAULT_PARAMS['artifact'], **kwargs}
        excluded = set()

        if params['ecg_ch_name'] in self.raw.ch_names:
            try:
                ecg_comp, _ = self.ica.find_bads_ecg(self.raw, threshold=params['ecg_threshold'],
                                                      ch_name=params['ecg_ch_name'])
                excluded.update(ecg_comp)
            except TypeError:
                ecg_comp, _ = self.ica.find_bads_ecg(self.raw, threshold=params['ecg_threshold'],
                                                      channel=params['ecg_ch_name'])
                excluded.update(ecg_comp)
            except Exception as e:
                logger.warning(f"ECG检测失败: {e}")

        if params['eog_ch_name'] in self.raw.ch_names:
            try:
                eog_comp, _ = self.ica.find_bads_eog(self.raw, threshold=params['eog_threshold'],
                                                      ch_name=params['eog_ch_name'])
                excluded.update(eog_comp)
            except TypeError:
                eog_comp, _ = self.ica.find_bads_eog(self.raw, threshold=params['eog_threshold'],
                                                      channel=params['eog_ch_name'])
                excluded.update(eog_comp)
            except Exception as e:
                logger.warning(f"EOG检测失败: {e}")

        if params.get('use_spectral_classification', True):
            spectral_labels = classify_ica_components(self.ica, self.raw)
            excluded.update(spectral_labels.get('eog', []))
            excluded.update(spectral_labels.get('muscle', []))
            excluded.update(spectral_labels.get('line', []))
            logger.info(f"频谱分类结果: EOG={spectral_labels['eog']}, "
                        f"Muscle={spectral_labels['muscle']}, "
                        f"Line={spectral_labels['line']}, "
                        f"Brain={spectral_labels['brain']}")

        self.ica.exclude = sorted(list(excluded))
        self._log_step('artifact_detect', {
            'excluded_components': self.ica.exclude,
            'artifact_params': params
        })
        return self

    def apply_clean(self) -> 'MNEPipeline':
        """数据清洗综合处理"""
        if self.ica and len(self.ica.exclude) > 0:
            self.ica.apply(self.raw)
            logger.info(f"已剔除ICA成分: {self.ica.exclude}")

        self._calculate_quality_metrics()
        self.depression_markers = compute_depression_markers(self.raw)

        self._log_step('final_clean', {
            'remaining_channels': self.raw.info['ch_names'],
            'quality_metrics': self.quality_metrics,
            'depression_markers': self.depression_markers
        })
        return self

    def apply_standard_pipeline(self, l_freq=0.1, h_freq=40, ica_method='infomax',
                                apply_notch=True, notch_freqs=50) -> 'MNEPipeline':
        """抑郁症检测标准化流程（通道数感知 + 自动幅度归一化）"""
        try:
            n_channels = len(self.raw.ch_names)
            is_low_channel = n_channels <= 5
            mode_str = "LOW_CHANNEL" if is_low_channel else "STANDARD"
            logger.info(f"=== 预处理模式: {mode_str} ({n_channels}通道) ===")

            logger.info("开始标准预处理流程 (抑郁症检测优化版)")

            if self.raw.n_times == 0:
                raise ValueError("输入数据为空 (n_times=0)")

            orig_state = {
                'ch_names': self.raw.ch_names.copy(),
                'n_times': self.raw.n_times,
                'sfreq': self.raw.info['sfreq'],
                'data_hash': hash(self.raw._data.tobytes())
            }
            logger.info(f"原始数据状态: {orig_state['n_times']}采样点, {len(orig_state['ch_names'])}通道")

            if l_freq is not None and h_freq is not None and l_freq >= h_freq:
                raise ValueError(f"滤波参数无效: 低通{l_freq}Hz >= 高通{h_freq}Hz")

            logger.info(">> 步骤0: 自动幅度归一化")
            self.auto_scale(target_mad=50.0)

            logger.info(">> 步骤1: 应用带通滤波器")
            self.apply_filter(l_freq=l_freq, h_freq=h_freq)

            if apply_notch:
                logger.info(">> 步骤2: 应用陷波滤波器")
                self.apply_notch(freqs=notch_freqs)
            else:
                logger.info(">> 跳过陷波 (用户指定)")

            logger.info(">> 步骤3: 检测并处理坏道")
            self.detect_bad_channels(interpolate=True)

            logger.info(">> 步骤4: 设置参考电极")
            self.set_reference()

            min_samples_ica = int(self.raw.info['sfreq'] * 2)
            if is_low_channel:
                logger.info(">> 低通道模式: 跳过步骤5-6 (ICA和伪迹检测)")
                self.ica = None
                self._log_step('ica', {'status': 'skipped (low_channel_mode)'})
                self._log_step('artifact_detect', {'status': 'skipped (low_channel_mode)'})
            elif self.raw.n_times >= min_samples_ica:
                logger.info(f">> 步骤5: 执行ICA ({ica_method})")
                self.run_ica(method=ica_method)
                logger.info(">> 步骤6: 伪迹检测")
                self.detect_artifacts()
            else:
                logger.info(">> 跳过步骤5-6 (ICA和伪迹检测，数据不足)")

            logger.info(">> 步骤7: 应用清理")
            self.apply_clean()

            current_state = {
                'n_times': self.raw.n_times,
                'n_channels': len(self.raw.ch_names)
            }
            if current_state['n_times'] == 0:
                raise RuntimeError("处理流程导致数据丢失")

            logger.info(
                f"预处理完成: 通道 {len(orig_state['ch_names'])}->{current_state['n_channels']}, "
                f"采样点 {orig_state['n_times']}->{current_state['n_times']}"
            )
            return self

        except Exception as e:
            current_step = self.processing_log[-1]['step'] if self.processing_log else "未知"
            logger.critical(f"预处理失败于[{current_step}]: {str(e)}")
            raise RuntimeError(f"预处理失败于[{current_step}]: {str(e)}") from e

    def _calculate_quality_metrics(self) -> None:
        """综合质量评估（含抑郁症标记）"""
        self.quality_metrics = {
            'snr_db': calculate_snr(self.raw),
            'bad_channels_ratio': len(compute_bad_channels(self.raw)) / len(self.raw.ch_names),
            'ica_artifact_ratio': len(self.ica.exclude) / self.ica.n_components_ if self.ica else 0,
            'remaining_samples': self.raw.n_times,
            'effective_sfreq': self.raw.info['sfreq']
        }

    def get_processed_data(self) -> mne.io.BaseRaw:
        return self.raw.copy()

    def get_processing_report(self) -> Dict[str, Any]:
        """生成处理报告"""
        metrics = {
            'snr_db': float(self.quality_metrics['snr_db']),
            'bad_channels_ratio': float(self.quality_metrics['bad_channels_ratio']),
            'ica_artifact_ratio': float(self.quality_metrics['ica_artifact_ratio']),
            'remaining_samples': int(self.quality_metrics['remaining_samples']),
            'effective_sfreq': float(self.quality_metrics['effective_sfreq'])
        }

        depression = {}
        for k, v in self.depression_markers.items():
            if isinstance(v, (np.integer, np.floating)):
                depression[k] = float(v)
            else:
                depression[k] = v

        serializable_log = []
        for entry in self.processing_log:
            entry_copy = entry.copy()
            if 'params' in entry_copy:
                for key, value in entry_copy['params'].items():
                    try:
                        if hasattr(value, 'item') and callable(getattr(value, 'item', None)):
                            entry_copy['params'][key] = value.item()
                        elif isinstance(value, np.generic):
                            if np.issubdtype(value.dtype, np.integer):
                                entry_copy['params'][key] = int(value)
                            elif np.issubdtype(value.dtype, np.floating):
                                entry_copy['params'][key] = float(value)
                    except Exception:
                        pass
            serializable_log.append(entry_copy)

        return {
            'log': serializable_log,
            'metrics': metrics,
            'depression_markers': depression,
            'final_info': {
                'n_channels': int(len(self.raw.ch_names)),
                'sfreq': float(self.raw.info['sfreq']),
                'duration': float(self.raw.times[-1])
            }
        }