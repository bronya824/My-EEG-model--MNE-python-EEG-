import numpy as np
import matplotlib
matplotlib.use('Agg')  # 无GUI环境也能跑
import matplotlib.pyplot as plt
import mne

# ==================== 配置 ====================
PROC_PATH = 'D:/Python_study/EEG_Processor/examples/data/processed/proc-02010001_raw_eeg.fif'
UV_PER_V = 1e6  # MNE 内部单位是 Volts，必须转 µV

# ==================== 读取 Proc 文件 ====================
raw = mne.io.read_raw_fif(PROC_PATH, preload=True, verbose='WARNING')
data = raw.get_data() * UV_PER_V  # <-- 关键：转 µV

print(f"Proc 文件幅度 (µV):")
print(f"  median = {np.median(data):.4f}")
print(f"  MAD    = {np.median(np.abs(data - np.median(data))):.4f}")
print(f"  ptp    = {np.ptp(data):.4f}")
print(f"  std    = {np.std(data):.4f}")

# 判断是否正常
mad = np.median(np.abs(data - np.median(data)))
if 0.01 <= mad <= 200:
    print(f"\n✅ 幅度正常 (MAD={mad:.4f} µV)")
else:
    print(f"\n⚠️ 幅度异常 (MAD={mad:.4f} µV)")

# ==================== PSD 检查 ====================
fig = raw.plot_psd(fmax=60, show=False)
fig.savefig('proc_psd_check.png', dpi=150, bbox_inches='tight')
print(f"\n📊 PSD 图已保存: proc_psd_check.png")
plt.close(fig)