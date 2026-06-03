"""Script for visualizing LFP traces"""

import os
import numpy as np
from scipy.signal import decimate
from dotenv import load_dotenv
from ephyviewer import mkQApp, MainViewer, TraceViewer

load_dotenv()

fpath = os.path.join(os.getenv("RAW_DATA_DIR"), os.getenv("AWAKE_TEST_DATA_FNAME"))

n_channels = 384
dtype = np.int16
sample_rate = 30000.0
target_rate = 1000.0
ds_factor = int(sample_rate / target_rate)  # 30

n_samples = os.path.getsize(fpath) // (n_channels * np.dtype(dtype).itemsize)
raw = np.memmap(fpath, dtype=dtype, mode='r', shape=(n_samples, n_channels))

# every 24th channel → 16 channels
channel_idx = np.arange(0, n_channels, 24)
raw = raw[:, channel_idx]

# decimate each channel (IIR antialias filter + downsample)
sigs = decimate(raw.astype(np.float32), ds_factor, axis=0, zero_phase=True)

app = mkQApp()
win = MainViewer(debug=True, show_auto_scale=True)

view1 = TraceViewer.from_numpy(sigs, target_rate, t_start=0., name='NP2 ProbeB')
view1.params['scale_mode'] = 'same_for_all'
view1.params['display_labels'] = True
view1.auto_scale()

win.add_view(view1)
win.show()
app.exec()
