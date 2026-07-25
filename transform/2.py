import numpy as np
import mne
import matplotlib.pyplot as plt

raw = mne.io.read_raw_fif(
    'D:/Python_study/EEG_Processor/examples/data/processed/proc-02010001_raw_eeg.fif',
    preload=True
)

data = raw.get_data()
print(f"MAD: {np.median(np.abs(data - np.median(data))) * 1e6:.2f} µV")
print(f"Std: {np.std(data) * 1e6:.2f} µV")
print(f"Range: {np.ptp(data) * 1e6:.2f} µV")

# 画 PSD 看是否有 Alpha 峰
psd = raw.compute_psd(fmax=60)
psd.plot()
plt.show()