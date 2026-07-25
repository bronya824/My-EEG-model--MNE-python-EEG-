#!/usr/bin/env python3
"""
EEG原始数据转MNE-FIF格式工具
================================
将兰州大学2015年3通道静息态EEG数据转换为MNE-Python可用的FIF格式。

原始数据说明:
- 数据文件: {subject_id}_still.txt (空格分隔的文本文件)
- 三列分别对应: Fp1, Fpz, Fp2
- 值为原始ADC计数，需根据设备增益转换为微伏
- 被试信息: subjects_information_EEG_3channels_resting_lanzhou_2015.xlsx

使用方法:
    python convert_to_fif.py

作者: AI Assistant
"""

import os
import numpy as np
import pandas as pd
import mne
from datetime import datetime, timezone

# ==================== 配置参数 ====================
# 请根据实际EEG设备修改以下参数！
SFREQ = 500.0              # 采样率 (Hz) —— 请根据设备说明书确认！
DATA_DIR = './data/raw'        # 原始数据目录
OUTPUT_DIR = './data/output' # 输出目录
INFO_FILE = 'subjects_information_EEG_3channels_resting_lanzhou_2015.xlsx'

# 通道信息
CH_NAMES = ['Fp1', 'Fpz', 'Fp2']
CH_TYPES = ['eeg', 'eeg', 'eeg']

# 设备增益 (用于ADC→μV转换)。如果不知道，设为1.0保留原始ADC值
# 例如: 如果设备说明书写 "增益=1000, 分辨率=0.0488 μV/LSB"
# 则 GAIN = 0.0488
GAIN_UV_PER_LSB = None # 默认值，请根据设备修改！


def load_subject_info(info_path):
    """读取并清理被试信息表"""
    df_raw = pd.read_excel(info_path)

    # 去掉表头重复行和说明行
    df = df_raw[df_raw['subject id'].notna()].copy()
    df = df[df['subject id'].astype(str) != 'subject id']
    df = df[~df['type'].astype(str).str.contains('MDD =', na=False)]
    df = df[~df['type'].astype(str).str.contains('abbreviation', na=False)]
    df = df[~df['type'].astype(str).str.contains('HC =', na=False)]
    df = df[~df['type'].astype(str).str.contains('PHQ-9 =', na=False)]

    # 统一subject id格式 (去掉前导零差异)
    df['subject_id_clean'] = df['subject id'].astype(str).str.strip().str.lstrip('0')
    df.loc[df['subject_id_clean'] == '', 'subject_id_clean'] = '0'

    return df

# 通道信息
CH_NAMES = ['Fp1', 'Fpz', 'Fp2']
CH_TYPES = ['eeg', 'eeg', 'eeg']


def txt_to_fif(txt_path, subject_id, subject_info_row, output_dir, sfreq=500.0,
               gain_uv_per_lsb=None, remove_dc=True):
    """
    将单个txt文件转换为MNE Raw FIF格式。
    当 gain_uv_per_lsb 为 None 时，自动进行 robust 幅度归一化（MAD → 50 µV）。
    MNE 内部存储单位为 Volts，因此最终数据必须以 V 为单位传入 RawArray。
    """

    # 1. 读取数据
    raw_data = np.loadtxt(txt_path)
    data = raw_data.T
    n_channels, n_times = data.shape
    print(f"  [{subject_id}] 数据形状: {data.shape}, 时长: {n_times / sfreq:.1f}s")

    # 2. 单位转换 / 自动归一化
    if gain_uv_per_lsb is not None:
        # 用户指定了明确增益（如以后拿到设备手册）
        # gain_uv_per_lsb 的单位是 µV/count，需转为 V/count
        scale_v = gain_uv_per_lsb * 1e-6
        data = data * scale_v
        print(f"  [{subject_id}] 应用设备增益: {gain_uv_per_lsb} µV/LSB")
    else:
        # 自动 robust 归一化 —— 不依赖任何外部参数
        # 先去除各通道 DC 偏移
        data = data - np.mean(data, axis=1, keepdims=True)
        global_mad = float(np.median(np.abs(data)))  # 单位: ADC counts

        if global_mad > 1000.0:  # 原始 ADC 计数通常 MAD 在 1e5 量级
            # MNE RawArray 要求数据单位为 Volts
            # 目标 MAD = 50 µV = 50e-6 V
            target_mad_v = 50e-6
            scale = target_mad_v / global_mad
            data = data * scale
            print(f"  [{subject_id}] 自动归一化: 原MAD={global_mad:.2f} counts, "
                  f"scale={scale:.2e} V/count -> 目标MAD=50µV")
        else:
            # 数据可能已经是 µV 级别，直接转 Volts
            data = data * 1e-6
            print(f"  [{subject_id}] 幅度正常，直接转Volts: MAD={global_mad:.2f} counts")

    # 3. 再次去 DC（保险）
    data = data - np.mean(data, axis=1, keepdims=True)

    # 4. 验证幅度（MNE 内部单位为 V，打印时转 µV）
    final_mad_v = float(np.median(np.abs(data)))
    final_ptp_v = float(np.ptp(data))
    print(f"  [{subject_id}] 转换后全局MAD: {final_mad_v * 1e6:.4f} µV")
    print(f"  [{subject_id}] 转换后峰峰值:  {final_ptp_v * 1e6:.4f} µV")

    # 5. 创建 MNE Info 和 Raw
    info = mne.create_info(ch_names=CH_NAMES, sfreq=sfreq, ch_types=CH_TYPES)
    raw = mne.io.RawArray(data, info)

    montage = mne.channels.make_standard_montage('standard_1020')
    raw.set_montage(montage, match_case=False)

    # 7. 添加元数据
    subj_type = str(subject_info_row['type']).strip()
    subj_age = int(subject_info_row['age'])
    subj_gender = str(subject_info_row['gender']).strip()
    subj_edu = int(subject_info_row['education（years）'])
    subj_phq9 = int(subject_info_row['PHQ-9'])
    subj_gad7 = int(subject_info_row['GAD-7'])
    subj_psqi = int(subject_info_row['PSQI'])

    description = f"""Dataset: EEG_3channels_resting_lanzhou_2015
Subject ID: {subject_id}
Group: {subj_type}
Age: {subj_age}
Gender: {subj_gender}
Education: {subj_edu} years
PHQ-9: {subj_phq9} | GAD-7: {subj_gad7} | PSQI: {subj_psqi}
Sampling Rate: {sfreq} Hz
Duration: {n_times/sfreq:.1f} s
Channels: Fp1, Fpz, Fp2
Gain: {gain_uv_per_lsb} μV/LSB
DC removed: {remove_dc}
"""

    raw.info['subject_info'] = {
        'his_id': subject_id,
        'sex': 1 if subj_gender == 'M' else 2,
    }
    raw.info['description'] = description
    raw.info['experimenter'] = 'Lanzhou University 2015'
    raw.set_meas_date(datetime(2015, 1, 1, 12, 0, 0, tzinfo=timezone.utc))

    # 8. 保存为FIF
    os.makedirs(output_dir, exist_ok=True)
    fif_name = f"{subject_id}_raw.fif"
    fif_path = os.path.join(output_dir, fif_name)
    raw.save(fif_path, overwrite=True)

    print(f"  ✅ 已保存: {fif_path}")
    return fif_path


def batch_convert(data_dir, info_file, output_dir, sfreq=500.0):
    """批量转换所有txt文件"""

    # 读取被试信息
    df_info = load_subject_info(info_file)
    print(f"📋 加载了 {len(df_info)} 条被试信息")

    # 查找所有txt文件
    txt_files = [f for f in os.listdir(data_dir) if f.endswith('_still.txt')]
    print(f"📁 找到 {len(txt_files)} 个数据文件")

    success_count = 0
    for txt_file in sorted(txt_files):
        # 从文件名提取subject_id (如 "02010001_still.txt" → "02010001")
        subject_id = txt_file.replace('_still.txt', '')

        # 在信息表中查找匹配 (处理前导零差异)
        search_id = subject_id.lstrip('0')
        if search_id == '':
            search_id = '0'

        match = df_info[df_info['subject_id_clean'] == search_id]

        if len(match) == 0:
            print(f"  ⚠️ 跳过 {txt_file}: 未找到被试信息")
            continue

        txt_path = os.path.join(data_dir, txt_file)
        try:
            txt_to_fif(
                txt_path=txt_path,
                subject_id=subject_id,
                subject_info_row=match.iloc[0],
                output_dir=output_dir,
                sfreq=sfreq,
                gain_uv_per_lsb=GAIN_UV_PER_LSB,
                remove_dc=True
            )
            success_count += 1
        except Exception as e:
            print(f"  ❌ 错误 {txt_file}: {e}")

    print(f"🎉 完成! 成功转换 {success_count}/{len(txt_files)} 个文件")
    print(f"   输出目录: {os.path.abspath(output_dir)}")


# ==================== 主程序 ====================
if __name__ == '__main__':

    # 单文件示例 (02010001)
    print("=" * 60)
    print("示例: 转换单个文件 02010001_still.txt")
    print("=" * 60)

    # 请修改以下路径为您的实际路径
    sample_txt = 'D:/Python_study/EEG_Processor/transform/data/raw/02010001_still.txt'
    sample_info = 'D:/Python_study/EEG_Processor/transform/data/raw/subjects_information_EEG_3channels_resting_lanzhou_2015.xlsx'
    sample_output = 'D:/Python_study/EEG_Processor/transform/data/output'

    df_info = load_subject_info(sample_info)
    subject_row = df_info[df_info['subject_id_clean'] == '2010001'].iloc[0]

    fif_path = txt_to_fif(
        txt_path=sample_txt,
        subject_id='02010001',
        subject_info_row=subject_row,
        output_dir=sample_output,
        sfreq=SFREQ,
        gain_uv_per_lsb=GAIN_UV_PER_LSB,
        remove_dc=True
    )

    # 验证读取
    print("\n验证读取:")
    raw_check = mne.io.read_raw_fif(fif_path, preload=False, verbose='WARNING')
    print(f"  通道: {raw_check.ch_names}")
    print(f"  采样率: {raw_check.info['sfreq']} Hz")
    print(f"  时长: {raw_check.times[-1]:.1f} s")
    print(f"  被试: {raw_check.info['subject_info']}")

    print("\n" + "=" * 60)
    print("使用说明:")
    print("=" * 60)
    print("""
1. 修改脚本顶部的配置参数:
   - SFREQ: 根据您的EEG设备设置正确的采样率
   - GAIN_UV_PER_LSB: 根据设备说明书设置ADC→μV转换系数

2. 批量转换时，设置正确的 DATA_DIR 和调用 batch_convert()

3. 转换后的 .fif 文件可直接用 MNE-Python 读取:

   import mne
   raw = mne.io.read_raw_fif('02010001_raw.fif', preload=True)
   raw.plot()                    # 可视化
   raw.filter(1, 40)             # 滤波
   epochs = mne.make_fixed_length_epochs(raw, duration=2.0)  # 分段

4. 如需处理所有被试，取消注释下面的 batch_convert 调用:
   # batch_convert(DATA_DIR, INFO_FILE, OUTPUT_DIR, SFREQ)
""")
