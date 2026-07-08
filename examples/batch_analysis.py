from pathlib import Path
import base64
import logging
import numpy as np
import mne
from tqdm import tqdm
import sys
from eeg_proc.processing.preprocessor import MNEPipeline
from eeg_proc.models.eeg_model import AliyunOutputModel
import pyedflib

# 关键修复1: 最先配置MNE日志级别
mne.set_log_level("INFO")  # 必须先设置MNE的日志级别

# 增强日志配置 - 显示更多信息
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)-8s [%(name)s] - %(message)s',
    handlers=[
        logging.FileHandler('batch_processing.log', mode='w'),
        logging.StreamHandler()
    ],
    force=True  # 关键修复: 强制覆盖现有配置
)
logger = logging.getLogger('EEG_Batch_Processor')

# 关键修复2: 配置根日志器而不仅仅是当前日志器
root_logger = logging.getLogger()

# 修复 Windows GBK 控制台 Unicode 编码问题
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# 调整第三方库日志级别 - 允许INFO级别
for lib in ['alibabacloud_credentials', 'mne', 'urllib3', 'pyedflib']:
    logging.getLogger(lib).setLevel(logging.INFO)

# 新增：彩色日志支持 (应用到所有处理器)
try:
    from colorlog import ColoredFormatter

    color_formatter = ColoredFormatter(
        "%(log_color)s[%(asctime)s] %(levelname)-8s [%(name)s] - %(message)s",
        datefmt='%H:%M:%S',
        reset=True,
        log_colors={
            'DEBUG': 'cyan',
            'INFO': 'green',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'red,bg_white',
        }
    )

    # 关键修复3: 应用到所有处理器，包括根日志器
    for handler in root_logger.handlers:
        if isinstance(handler, logging.StreamHandler):
            handler.setFormatter(color_formatter)

    logger.info("彩色日志已启用")
except ImportError:
    logger.info("未安装colorlog，使用标准日志格式")

# 关键修复4: 禁用MNE的冗余输出
mne.set_config('MNE_LOGGING_LEVEL', 'INFO')


def generate_sample_data():
    """生成测试数据并输出详细日志"""
    data_dir = Path("./data/raw")
    data_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"生成测试数据到: {data_dir.absolute()}")

    for i in range(1, 4):
        fif_filename = f"sub-{i:02d}_task-test_eeg.fif"
        data = np.random.randn(32, 1000)
        info = mne.create_info(32, sfreq=1000, ch_types='eeg')
        raw = mne.io.RawArray(data, info)
        file_path = data_dir / fif_filename
        raw.save(file_path, overwrite=True)
        logger.info(f"创建FIF测试文件: {file_path.name} ({file_path.stat().st_size / 1024:.1f} KB)")

        edf_filename = f"sub-{i:02d}_task-test_eeg.edf"
        file_path = data_dir / edf_filename
        raw.save(file_path, overwrite=True, fmt='edf')
        logger.info(f"创建EDF测试文件: {file_path.name} ({file_path.stat().st_size / 1024:.1f} KB)")

    logger.info(f"测试数据生成完成，共创建 {3 * 2} 个文件")


def load_eeg_file(file_path: Path):
    """加载EEG文件并添加详细日志"""
    try:
        file_size = file_path.stat().st_size
        logger.info(f"加载文件: {file_path.name} ({file_size / 1024:.1f} KB)")

        if file_path.suffix.lower() == '.fif':
            logger.debug("使用MNE加载FIF文件")
            # 关键修复5: 禁用MNE的冗余控制台输出
            raw = mne.io.read_raw_fif(file_path, preload=True, verbose=False)
            logger.info(f"成功加载FIF文件: {len(raw.ch_names)}通道, {raw.n_times}采样点")
            return raw
        elif file_path.suffix.lower() == '.edf':
            logger.debug("使用增强方法加载EDF文件")
            return load_edf_robustly(file_path)
        else:
            raise ValueError(f"不支持的EEG文件格式: {file_path.suffix}")
    except Exception as e:
        logger.error(f"文件加载失败: {file_path.name} - {type(e).__name__}: {str(e)}", exc_info=True)
        raise


def load_edf_robustly(file_path: Path):
    """EDF加载增强版，添加详细日志"""
    logger.info("尝试方法1: 标准EDF加载")
    try:
        # 关键修复6: 禁用MNE的冗余输出
        raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
        logger.info(f"方法1成功: 加载{len(raw.ch_names)}通道EDF数据")
        return raw
    except Exception as e1:
        logger.warning(f"方法1失败: {str(e1)[:100]}... 尝试方法2")

    logger.info("尝试方法2: 禁用刺激通道")
    try:
        raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False, stim_channel=None)
        logger.info(f"方法2成功: 加载{len(raw.ch_names)}通道EDF数据")
        return raw
    except Exception as e2:
        logger.warning(f"方法2失败: {str(e2)[:100]}... 尝试方法3")

    logger.info("尝试方法3: pyedflib手动加载")
    try:
        return load_edf_with_pyedflib(file_path)
    except Exception as e3:
        logger.error(f"方法3失败: {str(e3)[:100]}")
        raise RuntimeError(f"所有EDF加载方法均失败: {file_path.name}")


def load_edf_with_pyedflib(file_path: Path):
    """pyedflib加载添加详细日志"""
    logger.info("使用pyedflib加载EDF文件")
    try:
        with pyedflib.EdfReader(str(file_path)) as edf_file:
            n = edf_file.signals_in_file
            signal_labels = edf_file.getSignalLabels()
            fs = edf_file.getSampleFrequency(0)
            n_samples = edf_file.getNSamples()[0]

            logger.info(f"EDF信息: {n}通道, 采样率={fs}Hz, 样本数={n_samples}")

            data = np.zeros((n, n_samples))
            for i in range(n):
                data[i, :] = edf_file.readSignal(i)
                if i % 5 == 0:
                    logger.debug(f"读取通道 {i + 1}/{n}: {signal_labels[i]}")

        # 创建MNE Raw对象
        ch_names = signal_labels[:n]
        info = mne.create_info(ch_names=ch_names, sfreq=fs, ch_types='eeg')
        raw = mne.io.RawArray(data, info)
        logger.info(f"成功创建Raw对象: {len(raw.ch_names)}通道, {raw.n_times}采样点")
        return raw
    except Exception as e:
        logger.error(f"pyedflib加载失败: {type(e).__name__} - {str(e)}", exc_info=True)
        raise


def run_batch_analysis(input_dir: str = "./data/raw"):
    """批处理流程 - 详细日志增强版"""
    try:
        logger.info("\n" + "=" * 60)
        logger.info("EEG批处理分析启动")
        logger.info("=" * 60)

        input_path = Path(input_dir).absolute()
        logger.info(f"输入目录: {input_path}")

        if not input_path.exists():
            logger.error(f"路径不存在: {input_path}")
            raise FileNotFoundError(f"路径不存在: {input_path}")

        fif_files = list(input_path.rglob("*.fif"))
        edf_files = list(input_path.rglob("*.edf"))
        valid_files = fif_files + edf_files

        logger.info(f"找到 {len(fif_files)} 个FIF文件, {len(edf_files)} 个EDF文件")

        if not valid_files:
            logger.warning("未发现有效EEG文件，跳过处理")
            return 0

        logger.info("待处理文件列表:")
        for i, file in enumerate(valid_files[:10]):
            logger.info(f"  {i + 1}. {file.name}")
        if len(valid_files) > 10:
            logger.info(f"  其余 {len(valid_files) - 10} 个文件...")

        success_count = 0
        failed_files = []

        output_dir = Path("./data/processed").absolute()
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"输出目录: {output_dir}")

        disable_tqdm = len(valid_files) <= 3
        pbar = tqdm(valid_files, desc='处理EEG文件', disable=disable_tqdm)

        for file_idx, file in enumerate(pbar, 1):
            file_name = file.name
            try:
                logger.info("\n" + "-" * 50)
                logger.info(f"处理文件 ({file_idx}/{len(valid_files)}): {file_name}")
                logger.info("-" * 50)

                logger.info(f"步骤1/4: 加载文件")
                raw = load_eeg_file(file)

                logger.info(f"采样率: {raw.info['sfreq']}Hz, 通道数: {len(raw.ch_names)}, 时长: {raw.times[-1]:.1f}s")

                logger.info(f"步骤2/4: 预处理流水线")
                pipeline = MNEPipeline(raw=raw)
                pipeline.apply_standard_pipeline(
                    l_freq=0.5,
                    h_freq=50.0,
                    ica_method="infomax"
                )
                raw_clean = pipeline.raw
                logger.info(f"预处理完成: 剩余通道 {len(raw_clean.ch_names)}个, 时长 {raw_clean.times[-1]:.1f}s")

                logger.info(f"步骤3/4: 保存结果")
                output_path = output_dir / f"proc-{file.stem}_eeg.fif"
                output_path.parent.mkdir(parents=True, exist_ok=True)

                if raw_clean.info.get('meas_date') is not None:
                    try:
                        # 检查测量日期是否超出MNE允许范围
                        meas_sec = raw_clean.info['meas_date'][0]  # 获取秒数部分
                        if meas_sec < -2147483648 or meas_sec > 2147483647:
                            logger.warning(f"测量日期 {meas_sec} 超出范围，重置为None")
                            raw_clean.set_meas_date(None)  # 清除有问题的测量日期
                    except (IndexError, TypeError) as e:
                        logger.warning(f"测量日期格式异常，重置为None: {e}")
                        raw_clean.set_meas_date(None)

                raw_clean.save(output_path, overwrite=True)
                logger.info(f"✅ 保存成功: {output_path.name} ({output_path.stat().st_size / 1024:.1f} KB)")

                logger.info(f"步骤4/4: 上传到阿里云")
                data_bytes = raw_clean.get_data().astype(np.float32).tobytes()
                AliyunOutputModel(
                    processed_data=base64.b64encode(data_bytes).decode(),
                    metadata={
                        'sfreq': float(raw_clean.info['sfreq']),
                        'ch_names': raw_clean.ch_names,
                        'n_samples': int(raw_clean.n_times),
                        'source_format': file.suffix.lower()[1:]
                    }
                ).upload(bucket_name='eeg-data')
                logger.info(f"☁️ 上传成功: {file_name}")

                success_count += 1
                logger.info(f"文件处理完成: {file_name}")

            except ValueError as ve:
                if "sampling_rate" in str(ve):
                    logger.warning(f"⚠️ 采样率问题跳过文件: {file_name} - {str(ve)[:100]}")
                else:
                    logger.error(f"❌ 处理错误: {file_name} - {str(ve)}", exc_info=True)
                failed_files.append(file_name)
            except Exception as e:
                logger.error(f"❌ 文件处理失败: {file_name} - {type(e).__name__}: {str(e)}", exc_info=True)
                failed_files.append(file_name)

        logger.info("\n" + "=" * 60)
        logger.info(f"批处理完成: 成功 {success_count}/{len(valid_files)} 文件")
        if failed_files:
            logger.info(f"失败文件 ({len(failed_files)}):")
            for f in failed_files:
                logger.info(f"  - {f}")
        logger.info("=" * 60)

        return success_count

    except Exception as e:
        logger.critical(f"‼️ 系统级错误: {type(e).__name__} - {str(e)}", exc_info=True)
        return 0


if __name__ == "__main__":
    # 关键修复7: 确保所有日志输出都经过格式化
    root_logger = logging.getLogger()
    for handler in root_logger.handlers:
        if isinstance(handler, logging.StreamHandler):
            handler.setLevel(logging.INFO)

    logger.info("\n" + "=" * 60)
    logger.info(f"EEG批处理分析程序 v1.3")
    logger.info(f"工作目录: {Path.cwd()}")
    logger.info(f"Python版本: {sys.version}")
    logger.info(f"MNE版本: {mne.__version__}")
    logger.info("=" * 60)

    raw_dir = Path("./data/raw").absolute()
    logger.info(f"原始数据目录: {raw_dir}")

    if not raw_dir.exists():
        logger.warning(f"目录不存在，创建: {raw_dir}")
        raw_dir.mkdir(parents=True, exist_ok=True)

    fif_files = list(raw_dir.glob("*.fif"))
    edf_files = list(raw_dir.glob("*.edf"))
    total_files = fif_files + edf_files

    dir_contents = [f.name for f in raw_dir.iterdir() if f.is_file()]
    logger.info(f"目录内容 ({len(dir_contents)} 个文件):")
    for file in dir_contents[:5]:
        logger.info(f"  - {file}")
    if len(dir_contents) > 5:
        logger.info(f"  ... 其余 {len(dir_contents) - 5} 个文件")

    if not total_files:
        logger.warning("未找到EEG文件，生成测试数据...")
        generate_sample_data()
        fif_files = list(raw_dir.glob("*.fif"))
        edf_files = list(raw_dir.glob("*.edf"))
        total_files = fif_files + edf_files
        logger.info(f"生成后找到 {len(total_files)} 个EEG文件")
    else:
        logger.info(f"找到 {len(total_files)} 个EEG文件")

    logger.info("启动批处理分析")
    success_count = run_batch_analysis()

    logger.info("\n" + "=" * 60)
    logger.info(f"处理完成: 成功 {success_count} 个文件")
    logger.info(f"退出代码: 0")
    logger.info("=" * 60)
    sys.exit(0)