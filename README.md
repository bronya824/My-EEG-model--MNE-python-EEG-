# EEG Processor

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![MNE](https://img.shields.io/badge/MNE-%3E%3D1.6.0-green)](https://mne.tools/)
[![Flask](https://img.shields.io/badge/Flask-3.0%2B-lightgrey)](https://flask.palletsprojects.com/)
[![License](https://img.shields.io/badge/Apache-2.0-blue.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-50%20passed-brightgreen)](.)

基于 MNE-Python 的**脑电图（EEG）高级预处理流水线**，面向抑郁症研究，支持多源数据输入、自动化伪迹去除、阿里云 OSS 集成与实时流处理。

---

## 目录

- [特性概览](#特性概览)
- [架构设计](#架构设计)
- [项目结构](#项目结构)
- [快速开始](#快速开始)
- [核心模块](#核心模块)
- [REST API](#rest-api)
- [测试](#测试)
- [配置](#配置)
- [依赖项](#依赖项)

---

## 特性概览

| 功能 | 说明 |
|------|------|
| 📡 **多源数据输入** | 微信小程序 JSON/Base64 数据、EDF/FIF 文件上传 |
| 🧹 **自动化预处理** | 带通滤波 → 重参考 → ICA 分解 → 伪迹检测(ECG/EOG) → 数据清洗 |
| 🎛️ **方法链调用** | `MNEPipeline(raw).apply_filter().run_ica().detect_artifacts().apply_clean()` |
| 🌐 **REST API** | Flask 端点: `POST /process` 和 `POST /api/v1/upload` |
| ☁️ **阿里云 OSS** | `AliyunUploader` 单例客户端，`AliyunOutputModel` 序列化输出 |
| 🔍 **质量评估** | 自动计算 SNR (带工频陷波)、坏道比例、ICA 伪迹成分比 |
| ⚡ **实时流** | WebSocket 适配器 (`eeg_proc.api.streaming.WebSocketAdapter`) |
| 🧪 **测试完善** | 50 个 pytest 用例，覆盖 models / processing / io / api |

---

## 架构设计

```
微信端 JSON     EDF / FIF 文件
     │               │
     ▼               ▼
┌──────────────────────────────────────┐
│            EEGReceiver               │  ← eeg_proc/io/receiver.py
│  parse_wechat_input() / load_edf()  │      load_fif()
│   → mne.io.Raw                      │
└──────────────┬───────────────────────┘
               │
               ▼
┌──────────────────────────────────────┐
│           MNEPipeline                │  ← eeg_proc/processing/preprocessor.py
│  apply_filter() → set_reference()    │
│  → run_ica() → detect_artifacts()    │
│  → apply_clean()                     │
│                                      │
│  质量指标: calculate_snr()           │  ← quality_metrics_v2.py
│          compute_bad_channels()      │  ← artifact_detection.py
└──────────────┬───────────────────────┘
               │
               ▼
┌──────────────────────────────────────┐
│  WechatEEGInput / AliyunOutputModel  │  ← eeg_proc/models/eeg_model.py
│  ErrorResponse / ProcessingResult    │  ← eeg_proc/models/result_schema.py
└──────────────┬───────────────────────┘
               │
               ▼
┌──────────────────────────────────────┐
│         Flask REST API               │  ← eeg_proc/api/rest_api.py
│  AliyunUploader (OSS 单例客户端)     │
│  POST /process  → 微信 JSON 处理     │
│  POST /api/v1/upload → 文件上传      │
└──────────────────────────────────────┘
```

---

## 项目结构

```
EEG_Processor/
├── eeg_proc/
│   ├── __init__.py              # _SafeLazyLoader 延迟加载器
│   ├── api/
│   │   ├── __init__.py          # Flask app + 组件注册 + 路由注册
│   │   ├── rest_api.py          # 端点: process_wechat, handle_edf_upload
│   │   └── streaming.py         # WebSocketAdapter 实时流适配器
│   ├── io/
│   │   ├── __init__.py
│   │   └── receiver.py          # EEGReceiver 多源数据加载器
│   ├── models/
│   │   ├── __init__.py
│   │   ├── eeg_model.py         # WechatEEGInput, AliyunOutputModel
│   │   └── result_schema.py     # ErrorResponse, ProcessingResult
│   └── processing/
│       ├── __init__.py
│       ├── preprocessor.py      # MNEPipeline 核心预处理流水线
│       ├── artifact_detection.py # compute_bad_channels 坏道检测
│       └── quality_metrics_v2.py # calculate_snr SNR 计算(含工频陷波)
├── config/
│   ├── dev.yaml
│   └── prod.yaml
├── tests/
│   ├── test_models.py           # 18 个测试: 数据模型字段/校验/序列化
│   ├── test_processing.py       # 13 个测试: 滤波/ICA/参考/流水线/边界
│   ├── test_io.py               # 8 个测试: EDF/FIF 加载 + 通道映射 + 边界
│   └── test_api.py              # 11 个测试: 端点 + 错误注入 + 工具函数
├── examples/
│   ├── batch_analysis.py
│   └── realtime_processing.py
├── pyproject.toml               # 项目元数据 / 构建 / 依赖
├── requirements.txt             # pip 一键安装清单
├── setup.py
├── pytest.ini
├── Dockerfile
└── README.md
```

---

## 快速开始

### 环境要求

- Python ≥ 3.10

### 安装

```bash
git clone git@github.com:bronya824/My-EEG-model--MNE-python-EEG-.git
cd EEG_Processor

# 创建虚拟环境
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate

# 安装依赖
pip install -r requirements.txt

# (可选) 开发模式安装 + dev/depression 扩展
pip install -e ".[dev,depression]"
```

### 5 分钟本地体验

```python
import numpy as np
import mne
from eeg_proc import MNEPipeline

# 1. 生成模拟 EEG 数据 (10 通道, 250Hz, 4 秒)
montage = mne.channels.make_standard_montage('standard_1020')
ch_names = montage.ch_names[:10]
data = np.random.default_rng(42).normal(scale=1e-6, size=(10, 1000))
info = mne.create_info(ch_names=ch_names, sfreq=250., ch_types='eeg')
raw = mne.io.RawArray(data, info)

# 2. 一键预处理
pipeline = MNEPipeline(raw).apply_standard_pipeline(l_freq=1, h_freq=40)

# 3. 质量报告
report = pipeline.get_processing_report()
print(f"SNR: {report['metrics']['snr_db']:.2f} dB")
print(f"通道数: {report['final_info']['n_channels']}")
```

### 启动 REST API 服务

```bash
python -m eeg_proc.api.rest_api
# 默认监听 0.0.0.0:5000
```

---

## 核心模块

### 1. `MNEPipeline` — 预处理流水线

源码: `eeg_proc/processing/preprocessor.py` (524 行)

```python
from eeg_proc import MNEPipeline

pipeline = (MNEPipeline(raw)
    .apply_filter(l_freq=1, h_freq=40)       # 带通 (默认 FIR, 短数据自动降级 IIR)
    .set_reference(ref_channels='average')    # 平均参考
    .run_ica(n_components=0.95)               # ICA (保留 95% 方差, method='infomax')
    .detect_artifacts()                       # ECG/EOG 伪迹检测
    .apply_clean()                            # ICA 清洗 + 质量评估
)

# 方法链返回值均为 self, 支持逐步组合
# 完整流水线快捷方法:
pipeline = MNEPipeline(raw).apply_standard_pipeline(l_freq=1, h_freq=40)

# 获取结果
processed = pipeline.get_processed_data()
report = pipeline.get_processing_report()
# report 结构: {"log": [...], "metrics": {snr_db, bad_channels_ratio, ...}, "final_info": {...}}
```

**自动保护机制 (源码中实现):**
- 短数据 (< 2 秒) → 跳过 ICA, 滤波降级为 IIR (`preprocessor.py:122-132`)
- 通道数 < 3 → ICA 跳过 (`preprocessor.py:242-246`)
- 无效滤波参数 (`l_freq >= h_freq`) → `RuntimeError` (`preprocessor.py:391-393`)
- 空数据 (`n_times=0`) → `RuntimeError` (`preprocessor.py:377-379`)
- ICA PCA 方差问题 → 自动跳过 (`preprocessor.py:283-289`)

### 2. `EEGReceiver` — 数据加载器

源码: `eeg_proc/io/receiver.py` (127 行)

```python
from eeg_proc.io.receiver import EEGReceiver

receiver = EEGReceiver(max_sampling_rate=2000)

# 微信 JSON 数据
raw = receiver.parse_wechat_input(json_dict)

# EDF 文件 (支持通道映射 + 自动降采样)
raw = receiver.load_edf("recording.edf",
    channel_map={'EEG1': 'Fp1', 'EEG2': 'Fp2'},
    max_sfreq=250
)

# FIF 文件
raw = receiver.load_fif("recording_raw.fif")
```

**`parse_wechat_input` 处理流程:**
1. `WechatEEGInput(**json_data)` pydantic 校验
2. Base64 解码 → float32 numpy 数组
3. 字节数整除检查 (13 bytes / 10 ch → `ValueError`)
4. `mne.io.RawArray` 构建 + 标准 10-20 montage 设置

**`load_edf` 处理流程:**
1. `Path.exists()` 检查 → `FileNotFoundError`
2. `mne.io.read_raw_edf()` 加载 (排除 Misc 通道)
3. 通道映射校验 (`_validate_channels`) + 重命名
4. 超采样率自动低通滤波 + 降采样

### 3. 数据模型

源码: `eeg_proc/models/eeg_model.py` (220 行), `eeg_proc/models/result_schema.py` (26 行)

**WechatEEGInput** — 输入校验:
| 字段 | 约束 |
|------|------|
| `device_id` | 正则 `^eeg-\d+$` |
| `user_id` | 正则 `^usr-\d{8}$` |
| `sampling_rate` | `(0, 2000]` |
| `n_channels` | `(0, 128]` |
| `channel_names` | 10-20 系统标准名 (大小写不敏感, T3/T4→T7/T8 自动映射) |
| `eeg_data` | Base64 编码 float32, min_length=100 |

**AliyunOutputModel** — 输出校验:
| 字段 | 约束 |
|------|------|
| `processed_data` | Base64 编码的处理后数据 |
| `metadata.sfreq` | float / np.floating, > 0 |
| `metadata.ch_names` | list / np.ndarray, 非空 |
| `metadata.n_samples` | int / np.integer, > 0 |

**ErrorResponse** — 错误序列化:
```python
# error: str (min_length=3), details: Optional[str]
```

**ProcessingResult** — 批处理结果:
```python
# status: "success"|"failed"|"pending", duration: (0,300], artifact_count: ≥0
```

---

## REST API

| 方法 | 路径 | Content-Type | 说明 |
|------|------|-------------|------|
| `POST` | `/process` | `application/json` | 微信 EEG JSON 预处理 |
| `POST` | `/api/v1/upload` | `multipart/form-data` | EDF/FIF 文件上传处理 |

### `POST /process`

**请求体:**
```json
{
  "device_id": "eeg-2024",
  "user_id": "usr-20240423",
  "timestamp": "2024-04-23T09:30:00+08:00",
  "sampling_rate": 500.0,
  "n_channels": 10,
  "channel_names": ["Fp1","Fp2","F3","F4","C3","C4","P3","P4","O1","O2"],
  "eeg_data": "<base64 encoded float32>"
}
```

**成功响应 (200):**
```json
{
  "processed_data": "<base64>",
  "metadata": {
    "sfreq": 500.0,
    "ch_names": ["Fp1","Fp2",...],
    "n_samples": 15000
  }
}
```

**错误响应:**
| 状态码 | 场景 |
|--------|------|
| `415` | 非 JSON / 不支持的文件格式 |
| `422` | 数据校验失败 (通道名/采样率/格式) |
| `400` | 缺少文件 / 空文件名 |
| `500` | 内部处理异常 |

### `POST /api/v1/upload`

```bash
curl -X POST http://localhost:5000/api/v1/upload -F "file=@recording.edf"
curl -X POST http://localhost:5000/api/v1/upload -F "file=@recording_raw.fif"
```

文件扩展名校验: `.edf` / `.fif` (通过 `allowed_file()` 工具函数)

---

## 测试

### 运行所有测试

```bash
# 全部测试
python -m pytest tests/ -v

# 分模块运行
python -m pytest tests/test_models.py -v
python -m pytest tests/test_processing.py -v
python -m pytest tests/test_io.py -v
python -m pytest tests/test_api.py -v
```

### 覆盖概览

| 测试文件 | 数量 | 核心覆盖 |
|----------|------|---------|
| `test_models.py` | 18 | device_id/user_id 正则校验, sampling_rate/n_channels 范围, channel_names 标准化(大小写/T3T4/数量不匹配/非法名), AliyunOutputModel metadata 完整性(sfreq/ch_names/n_samples) |
| `test_processing.py` | 13 | 原始伪迹检测, 带通/高通/低通滤波, 短数据 IIR 降级, 平均参考, ICA 少通道跳过, ICA 短数据跳过, 完整流水线, 无效滤波参数拒绝, 质量指标(apply_clean后生成), get_processed_data/get_processing_report 输出类型, 处理日志, 空数据拒绝 |
| `test_io.py` | 8 | 微信数据解析, EDF 加载(pyedflib skip), 通道重命名映射, FIF 加载, FIF 文件不存在, EDF 高采样率降采样, 缺失通道映射拒绝(RuntimeError包裹), 字节数不可整除拒绝 |
| `test_api.py` | 11 | /process 正常流程, EDF 上传(edfio skip), 无效文件格式, 非 JSON 错误, 通道名标准化, FIF 上传, 无文件/空文件名, 缺字段 JSON, 缺 channel_names, 采样率超限, allowed_file 边界 |

**skip 说明:** `test_edf_loading` / `test_channel_renaming` / `test_edf_high_sampling_rate_resample` 等 5 个测试因缺少 `pyedflib` / `edfio` 而 skip，安装后全量通过:
```bash
pip install pyedflib edfio
python -m pytest tests/ -v   # 预计 55 passed
```

---

## 配置

`config/dev.yaml` / `config/prod.yaml` 用于环境级参数覆盖。`MNEPipeline.DEFAULT_PARAMS` (源码 `preprocessor.py:16-37`) 提供默认值:

```python
DEFAULT_PARAMS = {
    'filter': {'l_freq': 1.0, 'h_freq': 40.0, 'method': 'fir', ...},
    'ica':    {'method': 'infomax', 'random_state': 42, 'n_components': 0.99},
    'artifact': {'ecg_threshold': 0.3, 'eog_threshold': 0.5, ...}
}
```

---

## 依赖项

### 核心依赖

| 包 | 版本 | 用途 |
|----|------|------|
| `mne` | ≥1.6.0 | EEG 数据处理核心库 (pyproject.toml & requirements.txt) |
| `Flask` | ≥3.0.2 | REST API Web 框架 |
| `werkzeug` | ≥2.0.0 | Flask WSGI 工具库 |
| `pydantic` | ≥2.0 | 数据模型/字段校验 |
| `numpy` | ≥1.24 | 多维数组计算 |
| `scipy` | ≥1.10 | 信号处理 (滤波器设计, SNR 计算) |
| `pyedflib` | ≥0.1.32 | EDF 文件读写 |
| `alibabacloud_oss20190517` | — | 阿里云 OSS Python SDK (实际 import 名) |
| `websockets` | ≥11.0 | WebSocket 实时通信 |
| `tqdm` | ≥4.66.0 | 进度条 |

> **说明:** `pyproject.toml` 中 OSS SDK 元数据声明为 `alibabacloud_oss_sdk>=2.1.0`，但运行时实际导入的包名为 `alibabacloud_oss20190517`，后者在 `requirements.txt` 中指定。

### 可选依赖 (pyproject.toml extras)

```bash
pip install -e ".[dev]"         # pytest, pytest-cov, flask-testing
pip install -e ".[depression]"   # scikit-learn, mne-connectivity
```

---

## 许可证

本项目基于 [Apache-2.0](LICENSE) 许可证发布。
