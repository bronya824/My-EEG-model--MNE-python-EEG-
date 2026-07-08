#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MNE 数据加载与可视化脚本
支持格式：EDF、FIF
功能：
    1. 加载文件并打印基本信息
    2. 绘制原始信号（可交互缩放）
    3. 绘制功率谱密度（PSD）
用法：
    python script.py --file /path/to/your/file.edf
    python script.py --file /path/to/your/file.fif
    python script.py                 # 使用 MNE 示例数据演示
依赖：
    pip install mne matplotlib
"""

import argparse
import sys
import mne
import matplotlib.pyplot as plt

def load_raw(file_path):
    """
    根据文件扩展名自动选择读取方法，返回 mne.io.Raw 对象。
    """
    if file_path.endswith('.edf'):
        print(f"正在读取 EDF 文件: {file_path}")
        raw = mne.io.read_raw_edf(file_path, preload=False)
    elif file_path.endswith('.fif'):
        print(f"正在读取 FIF 文件: {file_path}")
        raw = mne.io.read_raw_fif(file_path, preload=False)
    else:
        raise ValueError("不支持的文件格式，请提供 .edf 或 .fif 文件。")
    return raw

def print_raw_info(raw):
    """
    打印 Raw 对象的关键信息。
    """
    print("\n=== 数据信息 ===")
    print(raw.info)
    print(f"通道数量: {len(raw.ch_names)}")
    print(f"通道名称: {raw.ch_names}")
    print(f"采样频率: {raw.info['sfreq']} Hz")
    print(f"数据时长: {raw.times[-1]:.2f} 秒")
    print("================\n")

def apply_montage_if_needed(raw):
    """
    如果数据没有电极位置信息，尝试应用标准 10-20 蒙太奇（仅当通道名称匹配时）。
    这一步对于绘制地形图或做源定位有用，但可视化原始信号不需要。
    """
    if raw.info['dig'] is None:
        try:
            montage = mne.channels.make_standard_montage('standard_1020')
            raw.set_montage(montage, on_missing='warn')
            print("已应用标准 10-20 蒙太奇。")
        except Exception as e:
            print(f"未能自动设置蒙太奇: {e}")

def visualize_raw(raw, n_channels=10, duration=30.0):
    """
    绘制原始信号。
    参数:
        n_channels: 同时显示的通道数（None 表示全部）
        duration : 初始显示的时间长度（秒）
    """
    print("正在打开交互式原始信号浏览器...")
    # 如果数据尚未载入内存，plot() 会自动载入；这里先 preload 避免重复警告
    if not raw.preload:
        raw.load_data()
    raw.plot(n_channels=n_channels, duration=duration, scalings='auto',
             title=f'原始信号 - {raw.filenames[0]}', block=True)

def plot_psd(raw):
    """
    绘制功率谱密度（PSD）。
    """
    print("正在计算并绘制功率谱密度 (PSD)...")
    if not raw.preload:
        raw.load_data()
    fig = raw.plot_psd(fmax=50, average=True, spatial_colors=True)
    fig.suptitle(f'功率谱密度 (PSD) - {raw.filenames[0]}', size='xx-large')
    plt.show(block=True)

def main():
    parser = argparse.ArgumentParser(description='使用 MNE 加载并可视化 EEG/MEG 数据 (.edf / .fif)')
    parser.add_argument('--file', '-f', type=str, default=None,
                        help='输入文件路径 (.edf 或 .fif)')
    args = parser.parse_args()

    if args.file is None:
        print("未提供文件，将使用 MNE 示例数据 (sample_audvis_raw.fif) 进行演示。")
        # 需要联网下载示例数据（仅首次运行）
        sample_data_folder = mne.datasets.sample.data_path()
        sample_data_raw_file = sample_data_folder / 'MEG' / 'sample' / 'sample_audvis_raw.fif'
        raw = mne.io.read_raw_fif(sample_data_raw_file, preload=False)
        print(f"示例数据路径: {sample_data_raw_file}")
    else:
        try:
            raw = load_raw(args.file)
        except Exception as e:
            print(f"加载文件失败: {e}", file=sys.stderr)
            sys.exit(1)

    # 打印信息
    print_raw_info(raw)

    # 尝试设置蒙太奇（可选）
    apply_montage_if_needed(raw)

    # 可视化原始信号（交互式窗口）
    visualize_raw(raw)

    # 绘制功率谱密度
    plot_psd(raw)

    print("可视化结束。")

if __name__ == "__main__":
    # 如果希望在脚本中直接运行（而非命令行），可以取消下面一行的注释并修改文件路径
    # sys.argv = ['script.py', '--file', 'your_file.edf']
    main()