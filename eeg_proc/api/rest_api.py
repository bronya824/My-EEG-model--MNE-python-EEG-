from flask import Flask, jsonify, request, abort
from werkzeug.exceptions import HTTPException, BadRequest
import mne
import os
import logging
import numpy as np
from pathlib import Path
from datetime import datetime
from werkzeug.utils import secure_filename
from eeg_proc.models.eeg_model import WechatEEGInput, AliyunOutputModel
from eeg_proc.processing.preprocessor import MNEPipeline
from pydantic import ValidationError
import base64
from mne.io import read_raw_edf, read_raw_fif
import traceback
from alibabacloud_oss20190517.client import Client as OssClient
from alibabacloud_tea_openapi import models as tea_models
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', write_through=True)
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', write_through=True)

__all__ = ['app', 'process_wechat', 'AliyunUploader', 'handle_edf_upload']


class AliyunUploader:
    _client = None

    @classmethod
    def get_client(cls):
        if not cls._client:
            config = tea_models.Config(
                access_key_id=os.getenv('ALIYUN_ACCESS_KEY'),
                access_key_secret=os.getenv('ALIYUN_SECRET_KEY'),
                region_id=os.getenv('ALIYUN_REGION', 'cn-shanghai')
            )
            cls._client = OssClient(config)
        return cls._client


# 替换现有的日志配置
app = Flask(__name__)

# 只在直接运行时配置日志
if __name__ == '__main__':
    # 配置详细的日志记录
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger('eeg_api')

    # 创建文件处理器
    file_handler = logging.FileHandler('api_errors.log', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'))
    app.logger.addHandler(file_handler)
    app.logger.setLevel(logging.DEBUG)
else:
    # 测试环境使用基本配置
    logging.basicConfig(level=logging.WARNING)


def _standardize_ch_names(ch_names):
    """通道名称标准化（扩展兼容性）"""
    # 特殊替换规则（扩展兼容性）
    replacements = {
        'FP1': 'Fp1', 'FP2': 'Fp2', 'FPZ': 'Fpz',
        'FZ': 'Fz', 'CZ': 'Cz', 'PZ': 'Pz', 'OZ': 'Oz',
        'T3': 'T7', 'T4': 'T8', 'T5': 'P7', 'T6': 'P8',
        'A1': 'M1', 'A2': 'M2',
        'fp1': 'Fp1', 'fp2': 'Fp2', 'fpz': 'Fpz',
        'fz': 'Fz', 'cz': 'Cz', 'pz': 'Pz', 'oz': 'Oz',
        'FP1.': 'Fp1', 'FP2.': 'Fp2', 'FPZ.': 'Fpz',
        'AFZ': 'AFz', 'POZ': 'POz', 'FCZ': 'FCz', 'CPZ': 'CPz',
        # 添加更多替换规则...
        'PO1': 'PO1', 'PO2': 'PO2', 'PO3': 'PO3', 'PO4': 'PO4',
        'PO5': 'PO5', 'PO6': 'PO6', 'PO7': 'PO7', 'PO8': 'PO8',
        'PO9': 'PO9', 'PO10': 'PO10'
    }

    # 双字母前缀列表（应保持大写）
    uppercase_prefixes = ['AF', 'FC', 'FT', 'CP', 'TP', 'PO', 'CB']

    standardized = []
    for name in ch_names:
        clean_name = name.strip().upper().split('.')[0]

        # 特殊处理：移除"EEG"前缀
        if clean_name.startswith("EEG "):
            clean_name = clean_name[4:]

        # 应用替换规则
        if clean_name in replacements:
            standardized_name = replacements[clean_name]
        else:
            # 检查是否是双字母前缀
            is_uppercase_prefix = any(
                clean_name.startswith(prefix) for prefix in uppercase_prefixes
            )

            if is_uppercase_prefix and len(clean_name) > 2:
                # 双字母前缀保持大写，后续部分小写
                prefix = clean_name[:2]
                suffix = clean_name[2:].lower()
                standardized_name = prefix + suffix
            else:
                # 其他情况：首字母大写，其余小写
                if clean_name:
                    standardized_name = clean_name[0] + clean_name[1:].lower()
                else:
                    standardized_name = clean_name

        standardized.append(standardized_name)

    return standardized


@app.errorhandler(HTTPException)
def handle_http_error(e):
    app.logger.error(f"HTTP错误 {e.code}: {e.description}")
    return jsonify({"error": e.description, "status_code": e.code}), e.code


@app.errorhandler(ValidationError)
def handle_validation_error(e):
    app.logger.error(f"数据校验失败: {str(e)}\n{traceback.format_exc()}")
    return jsonify({"error": "请求数据格式错误", "details": str(e)}), 422


@app.errorhandler(BadRequest)
def handle_bad_request(e):
    if 'JSON' in str(e):
        app.logger.error("无效的JSON格式")
        return jsonify({"error": "无效的JSON格式"}), 415
    app.logger.error(f"错误请求: {str(e)}")
    return jsonify({"error": str(e)}), 400


@app.errorhandler(Exception)
def handle_generic_error(e):
    app.logger.exception(f"未处理异常: {str(e)}\n{traceback.format_exc()}")
    return jsonify({"error": "系统内部错误"}), 500


@app.route('/process', methods=['POST'])
def process_wechat():
    try:
        app.logger.info("收到/process请求")

        # 检查JSON格式
        if not request.is_json:
            app.logger.error("请求不是JSON格式")
            abort(415, description="仅支持application/json格式")

        try:
            data = request.get_json()
            app.logger.debug(f"请求JSON数据: keys={list(data.keys())}")
        except BadRequest as e:
            app.logger.error(f"JSON解析失败: {str(e)}")
            abort(415, description=f"无效的JSON格式: {str(e)}")

        # 标准化通道名称
        if "channel_names" in data:
            original_names = data["channel_names"]
            data["channel_names"] = _standardize_ch_names(data["channel_names"])
            app.logger.debug(f"通道名称标准化: {original_names} -> {data['channel_names']}")

        # 验证输入数据
        try:
            validated = WechatEEGInput(**data)
            app.logger.info(f"数据验证成功: 设备={validated.device_id}, 用户={validated.user_id}")
            app.logger.debug(f"通道数={validated.n_channels}, 采样率={validated.sampling_rate}Hz")
            app.logger.debug(f"通道名称: {validated.channel_names}")
        except ValidationError as e:
            app.logger.error(f"数据验证失败: {str(e)}")
            abort(422, description=f"数据校验失败: {str(e)}")

        try:
            # 解码EEG数据
            try:
                eeg_bytes = base64.b64decode(validated.eeg_data)
                app.logger.debug(f"解码后数据长度: {len(eeg_bytes)}字节")

                # 转换为numpy数组
                eeg_array = np.frombuffer(eeg_bytes, dtype=np.float32)
                app.logger.debug(f"原始数据点数: {len(eeg_array)}")

                # 计算预期数据点数
                expected_points = validated.n_channels * (len(eeg_array) // validated.n_channels)
                app.logger.debug(f"预期数据点数: {expected_points}")

                # 重塑数据前检查长度
                if len(eeg_array) % validated.n_channels != 0:
                    error_msg = f"数据长度{len(eeg_array)}不是通道数{validated.n_channels}的整数倍"
                    app.logger.error(error_msg)
                    abort(422, description=error_msg)

                # 重塑数据
                eeg_array = eeg_array[:expected_points]  # 确保长度匹配
                eeg_array = eeg_array.reshape(validated.n_channels, -1)
                app.logger.info(f"数据重塑成功: {eeg_array.shape}")
            except Exception as e:
                app.logger.error(f"数据解码/重塑失败: {str(e)}\n{traceback.format_exc()}")
                abort(422, description="数据处理失败: 数据格式错误")

            # 创建Raw对象
            try:
                info = mne.create_info(
                    ch_names=validated.channel_names,
                    sfreq=validated.sampling_rate,
                    ch_types=['eeg'] * validated.n_channels
                )
                raw = mne.io.RawArray(eeg_array, info)
                app.logger.info(f"创建Raw对象成功: {len(raw.ch_names)}通道, {raw.n_times}采样点")
            except Exception as e:
                app.logger.error(f"创建Raw对象失败: {str(e)}\n{traceback.format_exc()}")
                abort(422, description="数据处理失败: 无法创建EEG对象")

            # 设置蒙太奇
            try:
                # 获取原始通道名
                original_ch_names = raw.ch_names
                # 生成标准化后的通道名列表
                standardized_list = _standardize_ch_names(original_ch_names)
                # 创建映射字典 {原始名称: 标准化名称}
                channel_mapping = dict(zip(original_ch_names, standardized_list))
                # 重命名通道
                raw.rename_channels(channel_mapping)

                raw.set_montage('standard_1020', on_missing='ignore', match_case=False)
                app.logger.info("蒙太奇设置成功")
            except Exception as e:
                app.logger.warning(f"蒙太奇设置警告: {str(e)}\n{traceback.format_exc()}")
                # 继续处理但不中断流程

            # 应用处理流程
            try:
                app.logger.info("开始应用预处理流程")
                processed = MNEPipeline(raw).apply_standard_pipeline()
                app.logger.info(f"处理流程完成: {len(processed.raw.ch_names)}通道, {processed.raw.n_times}采样点")
            except Exception as e:
                app.logger.error(f"处理流程失败: {str(e)}\n{traceback.format_exc()}")
                abort(500, description="处理流程内部错误")

            # 准备输出
            try:
                processed_data = processed.raw.get_data().tobytes()

                # 确保所有数字类型都是 Python 原生类型
                metadata = {
                    'sfreq': float(processed.raw.info['sfreq']),  # 转换为 float
                    'ch_names': processed.raw.ch_names,
                    'n_samples': int(processed.raw.n_times)  # 转换为 int
                }

                output = AliyunOutputModel(
                    processed_data=base64.b64encode(processed_data).decode(),
                    metadata=metadata
                )

                app.logger.info("处理成功完成")
                return jsonify(output.model_dump()), 200
            except Exception as e:
                app.logger.error(f"输出序列化失败: {str(e)}", exc_info=True)
                abort(500, description="数据处理结果序列化错误")

        except ValueError as e:
            app.logger.error(f"处理失败: {str(e)}\n{traceback.format_exc()}")
            abort(422, description=str(e))

    except HTTPException:
        # 已处理的HTTP异常，直接抛出
        raise
    except Exception as e:
        app.logger.error(f"未处理的处理失败: {str(e)}\n{traceback.format_exc()}")
        abort(500, description="内部处理错误")


@app.route('/api/v1/upload', methods=['POST'])
def handle_edf_upload():
    filepath = None
    try:
        app.logger.info("收到EDF上传请求")

        if 'file' not in request.files:
            app.logger.error("未检测到文件上传")
            abort(400, description="未检测到文件上传")

        file = request.files['file']
        if not file or file.filename == '':
            app.logger.error("空文件内容")
            abort(400, description="空文件内容")

        if not allowed_file(file.filename):
            app.logger.error(f"不支持的文件格式: {file.filename}")
            abort(415, description="仅支持EDF/FIF格式")

        temp_dir = Path(app.config['UPLOAD_FOLDER'])
        try:
            temp_dir.mkdir(parents=True, exist_ok=True)
            os.chmod(temp_dir, 0o755)
            app.logger.debug(f"创建/访问临时目录: {temp_dir}")
        except PermissionError as e:
            app.logger.error(f"权限错误: {str(e)}\n{traceback.format_exc()}")
            abort(500, description="服务器配置错误：无法写入临时目录")
        except Exception as e:
            app.logger.error(f"临时目录错误: {str(e)}\n{traceback.format_exc()}")
            abort(500, description="服务器配置错误：无法创建临时目录")

        filename = secure_filename(f"{datetime.now():%Y%m%d%H%M%S}_{file.filename}")
        file_extension = filename.rsplit('.', 1)[1].lower() if '.' in filename else ''
        filepath = temp_dir / filename

        try:
            file.save(str(filepath))
            app.logger.info(f"文件保存成功: {filepath}")
        except Exception as e:
            app.logger.error(f"文件保存失败: {str(e)}\n{traceback.format_exc()}")
            abort(500, description="无法保存上传文件")

        # 添加详细的日志记录
        file_size = os.path.getsize(filepath)
        app.logger.info(f"开始处理上传文件: {filename} ({file_size}字节)")
        app.logger.debug(f"文件路径: {filepath}")

        try:
            if file_extension == 'edf':
                app.logger.debug("尝试加载EDF文件")
                # 使用通用异常处理替代特定EDF错误
                try:
                    raw = read_raw_edf(
                        str(filepath),
                        preload=True,
                        verbose='debug' if app.debug else 'error',
                        exclude=['Status', 'Event', 'EDF Annotations'],
                        stim_channel=None
                    )
                    app.logger.info(f"成功加载EDF文件: {len(raw.ch_names)}通道, {raw.n_times}采样点")
                except (OSError, ValueError) as e:
                    # 捕获文件读取错误和格式错误
                    error_msg = f"EDF文件解析失败: {str(e)}"
                    app.logger.error(f"{error_msg}\n{traceback.format_exc()}")
                    abort(422, description=f"EDF文件格式错误: {str(e)}")

            elif file_extension == 'fif':
                app.logger.debug("尝试加载FIF文件")
                try:
                    raw = read_raw_fif(str(filepath), preload=True, verbose='error')
                    app.logger.info(f"成功加载FIF文件: {len(raw.ch_names)}通道, {raw.n_times}采样点")
                except Exception as e:
                    app.logger.error(f"FIF文件解析失败: {str(e)}\n{traceback.format_exc()}")
                    abort(422, description=f"FIF文件格式错误: {str(e)}")
            else:
                app.logger.error(f"不支持的文件格式: {file_extension}")
                abort(415, description="不支持的文件格式")

            app.logger.debug(f"原始通道名: {raw.ch_names}")

            # 标准化通道名称
            try:
                app.logger.debug("尝试标准化通道名称")

                # 获取原始通道名
                original_ch_names = raw.ch_names
                # 生成标准化后的通道名列表
                standardized_list = _standardize_ch_names(original_ch_names)
                # 创建映射字典 {原始名称: 标准化名称}
                channel_mapping = dict(zip(original_ch_names, standardized_list))

                # 应用重命名
                raw.rename_channels(channel_mapping)

                app.logger.info(f"标准化后通道名: {raw.ch_names}")
            except Exception as e:
                app.logger.error(f"通道标准化失败: {str(e)}\n{traceback.format_exc()}")
                abort(422, description=f"通道名称处理失败: {str(e)}")

            # 设置电极位置
            try:
                app.logger.debug("尝试设置电极位置")
                montage = mne.channels.make_standard_montage('standard_1020')
                raw.set_montage(montage, on_missing='warn', match_case=False)
                app.logger.info("成功设置电极位置")
            except Exception as e:
                app.logger.warning(f"电极位置设置失败: {str(e)}\n{traceback.format_exc()}")
                # 继续处理但不中断流程

            # 应用处理流程
            try:
                app.logger.info("开始预处理流程")
                processed = MNEPipeline(raw).apply_standard_pipeline()
                app.logger.info(f"预处理完成: {len(processed.raw.ch_names)}通道, {processed.raw.n_times}采样点")
            except Exception as e:
                app.logger.error(f"预处理失败: {str(e)}\n{traceback.format_exc()}")
                abort(500, description=f"数据处理错误: {str(e)}")

            # 准备输出
            try:
                processed_data = processed.raw.get_data().tobytes()

                # 确保所有数字类型都是 Python 原生类型
                metadata = {
                    'sfreq': float(processed.raw.info['sfreq']),  # 转换为 float
                    'ch_names': processed.raw.ch_names,
                    'n_samples': int(processed.raw.n_times)  # 转换为 int
                }

                output = AliyunOutputModel(
                    processed_data=base64.b64encode(processed_data).decode(),
                    metadata=metadata
                )

                app.logger.info("文件处理成功完成")
                return jsonify(output.model_dump()), 200
            except Exception as e:
                app.logger.error(f"输出序列化失败: {str(e)}", exc_info=True)
                abort(500, description="数据处理结果序列化错误")

        except Exception as e:
            app.logger.critical(f"文件处理错误: {str(e)}\n{traceback.format_exc()}")
            abort(500, description="文件处理内部错误")

    finally:
        if filepath and filepath.exists():
            try:
                filepath.unlink()
                app.logger.info(f"临时文件已删除: {filepath}")
            except Exception as e:
                app.logger.warning(f"文件删除失败: {str(e)}\n{traceback.format_exc()}")


def allowed_file(filename: str) -> bool:
    """检查文件扩展名是否允许"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'edf', 'fif'}


if __name__ == '__main__':
    # 修复标准输出流问题
    try:
        # 尝试写入标准输出以确保它是打开的
        sys.stdout.write("")
    except (ValueError, OSError):
        # 如果标准输出已关闭，则重新打开
        sys.stdout = open(os.devnull, 'w')

    try:
        sys.stderr.write("")
    except (ValueError, OSError):
        sys.stderr = open(os.devnull, 'w')

    app.run(host='0.0.0.0', port=5000, debug=False)