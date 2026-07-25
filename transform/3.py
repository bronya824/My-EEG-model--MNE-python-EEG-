import mne
import numpy as np
raw = mne.io.read_raw_fif(r'D:\Python_study\EEG_Processor\transform\data\output\02010001_raw.fif', preload=True)
data = raw.get_data()
print(f"Raw MAD: {np.median(np.abs(data - np.median(data))) * 1e6:.2f} µV")