"""
Spike sorting pipeline for Aeon neuropixels data.
Equivalent to spikeinterface_sandbox.ipynb.
"""

import os
import yaml
import numpy as np
import spikeinterface.extractors as se
import spikeinterface.preprocessing as sp
import spikeinterface.sorters as ss

from probeinterface import get_probe, Probe
from dotenv import load_dotenv, find_dotenv
from pathlib import Path

# ---------------------------------------------------------------------------
# Load environment variables and config
# ---------------------------------------------------------------------------
load_dotenv()
repo_root = Path(find_dotenv()).parent

raw_data_dir = Path(os.getenv("RAW_DATA_DIR"))
data_fname = Path(os.getenv("TEST_DATA_FNAME"))
derivatives_dir = Path(os.getenv("LARGE_FILE_DIR"))
full_data_path = os.path.join(raw_data_dir, data_fname)
config_path = os.path.join(repo_root, "config/preprocessing.yml")
sorting_output_dir = os.path.join(derivatives_dir, "sorting_output")

with open(Path(config_path)) as f:
    config = yaml.safe_load(f)

sample_rate = config['recording']['sample_rate']
n_channels  = config['recording']['n_channels']
dtype       = config['recording']['dtype']
freq_min    = config['preprocessing']['bandpass_filter']['freq_min']
freq_max    = config['preprocessing']['bandpass_filter']['freq_max']
reference   = config['preprocessing']['common_reference']['reference']
operator    = config['preprocessing']['common_reference']['operator']

print(f"Data path:          {full_data_path}")
print(f"Sorting output dir: {sorting_output_dir}")
print(f"Sample rate:        {sample_rate}")
print(f"N channels:         {n_channels}")
print(f"dtype:              {dtype}")
print(f"Bandpass:           {freq_min}-{freq_max} Hz")

# ---------------------------------------------------------------------------
# Build probe  (NP2013, shank 0, bottom 384 contacts)
# ---------------------------------------------------------------------------
probe_full = get_probe("imec", "NP2013")

shank0_mask      = probe_full.shank_ids == "0"
shank0_positions = probe_full.contact_positions[shank0_mask]
sorted_idx       = np.argsort(shank0_positions[:, 1])   # ascending = tip first
bottom_384_pos   = shank0_positions[sorted_idx[:384]]

probe = Probe(ndim=2, si_units="um")
probe.set_contacts(
    positions=bottom_384_pos,
    shapes="circle",
    shape_params={"radius": 7.5}
)
probe.set_device_channel_indices(np.arange(384))
probe.create_auto_shape(probe_type="tip")

print("Probe built successfully.")

# ---------------------------------------------------------------------------
# Preprocessing pipeline
# ---------------------------------------------------------------------------
recording = se.read_binary(full_data_path, sample_rate, dtype, n_channels)
print(f"Recording duration: {recording.get_num_samples() / sample_rate / 3600:.2f} hours")

recording_with_probe = recording.set_probe(probe)

# Phase correction — pass shifts as constructor argument so they serialise correctly
intersample_shifts = np.arange(n_channels) % 16 / 16
recording_shifted  = sp.phase_shift(recording_with_probe, inter_sample_shift=intersample_shifts)

recording_filtered = sp.bandpass_filter(recording_shifted, freq_min=freq_min, freq_max=freq_max)

recording_cmr = sp.common_reference(recording_filtered, reference=reference, operator=operator)

print("Preprocessing chain built.")

# ---------------------------------------------------------------------------
# Spike sorting
# ---------------------------------------------------------------------------
print("Starting Kilosort4...")
sorting = ss.run_sorter(
    "kilosort4",
    recording_cmr,
    folder=sorting_output_dir,
    remove_existing_folder=True,
    use_binary_file=False,
    verbose=True,
)

print(f"Sorting complete. Found {len(sorting.get_unit_ids())} units.")
print(f"Output written to: {sorting_output_dir}")