"""Script for visualizing LFP traces in different frequency bands"""

import os
import yaml

import ephyviewer
import numpy as np
import matplotlib.pyplot as plt
import spikeinterface.full as si
import spikeinterface.extractors as se
import spikeinterface.preprocessing as sp

from scipy.signal import spectrogram
from probeinterface import get_probe, Probe
from dotenv import load_dotenv, find_dotenv
from pathlib import Path


###############################
### Config and data loading ###
###############################

load_dotenv()
repo_root = Path(find_dotenv()).parent

raw_data_dir = Path(os.getenv("DUAL_RAW_DATA_DIR"))
data_fname = Path(os.getenv("AWAKE_TEST_DATA_FNAME_PL"))
derivatives_dir = Path(os.getenv("DERIVATIVES_DIR"))
full_data_path = os.path.join(raw_data_dir, data_fname)
config_path = os.path.join(repo_root, "config/preprocessing.yml")
lfp_output_dir = os.path.join(derivatives_dir, "lfp")

# Load preprocessing config
with open(Path(config_path)) as f:
    config = yaml.safe_load(f)
    
sample_rate = config['recording']['sample_rate']
n_channels = config['recording']['n_channels']
dtype = config['recording']['dtype']
freq_min = config['lfp']['bandpass_filter']['freq_min']
freq_max = config['lfp']['bandpass_filter']['freq_max']
reference = config['lfp']['common_reference']['reference']
operator = config['lfp']['common_reference']['operator']
resample_rate = config['lfp']['resample_rate']

recording = se.read_binary(full_data_path, sample_rate, dtype, n_channels)
recording_ch0 = recording.select_channels([recording.channel_ids[0]])


#########################
### Create visualiser ###
#########################

sig_source = ephyviewer.SpikeInterfaceRecordingSource(recording=recording_ch0)

# Low frequency (< 1 Hz)
filtered_recording = si.bandpass_filter(recording_ch0, freq_min=0.5, freq_max=1.)
sig_filtered_source_low = ephyviewer.SpikeInterfaceRecordingSource(recording=filtered_recording)

# Delta (1-4 Hz)
filtered_recording = si.bandpass_filter(recording_ch0, freq_min=1., freq_max=4.)
sig_filtered_source_delta = ephyviewer.SpikeInterfaceRecordingSource(recording=filtered_recording)

# Theta (4-8 Hz)
filtered_recording = si.bandpass_filter(recording_ch0, freq_min=4., freq_max=8.)
sig_filtered_source_theta = ephyviewer.SpikeInterfaceRecordingSource(recording=filtered_recording)

# Alpha (8-12 Hz)
filtered_recording = si.bandpass_filter(recording_ch0, freq_min=8., freq_max=12.)
sig_filtered_source_alpha = ephyviewer.SpikeInterfaceRecordingSource(recording=filtered_recording)

# Beta (12-30 Hz)
filtered_recording = si.bandpass_filter(recording_ch0, freq_min=12., freq_max=30.)
sig_filtered_source_beta = ephyviewer.SpikeInterfaceRecordingSource(recording=filtered_recording)

# Gamma (30-100 Hz)
filtered_recording = si.bandpass_filter(recording_ch0, freq_min=30., freq_max=100.)
sig_filtered_source_gamma = ephyviewer.SpikeInterfaceRecordingSource(recording=filtered_recording)

# Ripple (100-180 Hz)
filtered_recording = si.bandpass_filter(recording_ch0, freq_min=100., freq_max=180.)
sig_filtered_source_ripple = ephyviewer.SpikeInterfaceRecordingSource(recording=filtered_recording)

app = ephyviewer.mkQApp()
win = ephyviewer.MainViewer(debug=True, show_auto_scale=True)

view = ephyviewer.TraceViewer(source=sig_source, name='signals')
win.add_view(view)

view_low = ephyviewer.TraceViewer(source=sig_filtered_source_low, name='signals filtered - Low')
win.add_view(view_low)

view_delta = ephyviewer.TraceViewer(source=sig_filtered_source_delta, name='signals filtered - Delta')
win.add_view(view_delta)

view_theta = ephyviewer.TraceViewer(source=sig_filtered_source_theta, name='signals filtered - Theta')
win.add_view(view_theta)

view_alpha = ephyviewer.TraceViewer(source=sig_filtered_source_alpha, name='signals filtered - Alpha')
win.add_view(view_alpha)

view_beta = ephyviewer.TraceViewer(source=sig_filtered_source_beta, name='signals filtered - Beta')
win.add_view(view_beta)

view_gamma = ephyviewer.TraceViewer(source=sig_filtered_source_gamma, name='signals filtered - Gamma')
win.add_view(view_gamma)

view_ripple = ephyviewer.TraceViewer(source=sig_filtered_source_ripple, name='signals filtered - Ripple')
win.add_view(view_ripple)

win.show()
app.exec()