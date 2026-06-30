"""Streamlit ripple explorer: pick a recording + params, view ripple-detection plots.

Run with:  streamlit run scripts/ripple_explorer_app.py
"""

import os
import sys
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt
import streamlit as st
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sleep_sandbox.ripple import (load_recording, compute_psd, band_power,
                                  shank_index, pick_channels_per_shank,
                                  bandpass_envelope, detect_events, peri_event_trace)

REPO_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(REPO_ROOT / ".env")
with open(REPO_ROOT / "config" / "preprocessing.yml") as f:
    CONFIG = yaml.safe_load(f)
alt.data_transformers.disable_max_rows()  # peri-event windows can exceed Altair's 5000-row default

raw = os.getenv("RAW_DATA_DIR_DUAL")
deriv = os.getenv("DERIVATIVES_DIR_DUAL")
lfp_dir = Path(deriv) / "lfp" / "lfp_preprocessed"
PRESETS = {
    "ProbeB sleep #84 — preprocessed LFP (notebook ref)":
        (str(lfp_dir / "NeuropixelsV2_ProbeB_AmplifierData_84"), os.getenv("PROBE_CONFIG_PATH_HPC")),
    "ProbeB sleep #84 — raw":
        (str(Path(raw) / os.getenv("SLEEP_TEST_DATA_FNAME_HPC")), os.getenv("PROBE_CONFIG_PATH_HPC")),
    "ProbeB awake #144 — raw":
        (str(Path(raw) / os.getenv("AWAKE_TEST_DATA_FNAME_HPC")), os.getenv("PROBE_CONFIG_PATH_HPC")),
    "ProbeA sleep #84 — raw":
        (str(Path(raw) / os.getenv("SLEEP_TEST_DATA_FNAME_PL")), os.getenv("PROBE_CONFIG_PATH_PL")),
    "ProbeA awake #144 — raw":
        (str(Path(raw) / os.getenv("AWAKE_TEST_DATA_FNAME_PL")), os.getenv("PROBE_CONFIG_PATH_PL")),
}


@st.cache_resource(show_spinner="Loading recording…")
def get_recording(path, probe):
    return load_recording(path, probe, CONFIG)


@st.cache_data(show_spinner="Reading traces + computing PSD…")
def get_window(path, probe, t_start, t_dur):
    rec = get_recording(path, probe)
    fs = rec.get_sampling_frequency()
    s0 = int(t_start * fs)
    s1 = min(s0 + int(t_dur * fs), rec.get_num_frames())
    traces = rec.get_traces(start_frame=s0, end_frame=s1).astype(np.float32)
    freqs, psd = compute_psd(traces, fs)
    return fs, traces, freqs, psd, rec.get_channel_locations()


st.set_page_config(page_title="Ripple explorer", layout="wide")
st.sidebar.title("Ripple explorer")

preset_name = st.sidebar.selectbox("Recording preset", list(PRESETS))
preset_path, preset_probe = PRESETS[preset_name]
path = st.sidebar.text_input("Recording path (override)", value=preset_path)
probe = st.sidebar.text_input("Probe config path (override)", value=preset_probe)

st.sidebar.subheader("Analysis window")
t_start = st.sidebar.number_input("Start (s)", min_value=0.0, value=0.0, step=10.0)
t_dur = st.sidebar.number_input("Duration (s)", min_value=1.0, value=120.0, step=10.0)

st.sidebar.subheader("Passband (Hz)")
f_lo = st.sidebar.number_input("Low", min_value=1.0, value=100.0, step=10.0)
f_hi = st.sidebar.number_input("High", min_value=2.0, value=200.0, step=10.0)
passband = (f_lo, f_hi)
filter_order = st.sidebar.number_input("Butterworth order", min_value=1, max_value=10, value=4, step=1)

st.sidebar.subheader("Detection")
boundary_sd = st.sidebar.number_input("Boundary SD (event extent)", min_value=0.5, value=2.0, step=0.5)
peak_sd = st.sidebar.number_input("Peak SD (event must reach)", min_value=0.5, value=5.0, step=0.5)
dur_min = st.sidebar.number_input("Event min (ms)", min_value=0.0, value=20.0, step=5.0)
dur_max = st.sidebar.number_input("Event max (ms)", min_value=1.0, value=200.0, step=5.0)
env_plot_s = st.sidebar.number_input("Envelope plot duration (s)", min_value=1.0, value=100.0, step=10.0)
event_window_s = st.sidebar.number_input("Peri-event window (s)", min_value=0.05, value=0.4, step=0.05)

if not Path(path).exists():
    st.error(f"Path does not exist: {path}")
    st.stop()

fs, traces, freqs, psd, locs = get_window(path, probe, t_start, t_dur)
shank = shank_index(locs)
bp = band_power(freqs, psd, passband)
picks = pick_channels_per_shank(bp, shank, n_per_shank=1)
shanks = sorted(picks)

st.caption(f"{path}  |  fs={fs:.0f} Hz  |  {traces.shape[0] / fs:.1f} s × {traces.shape[1]} ch")

peak_opts = {f"shank {s} — ch {int(picks[s][0])}": int(picks[s][0]) for s in shanks}
sel_label = st.sidebar.selectbox("Channel (peak per shank)", list(peak_opts))
channel = peak_opts[sel_label]

# --- depth + PSD plots ---
c1, c2 = st.columns(2)
with c1:
    fig, ax = plt.subplots(figsize=(4, 8))
    for s in shanks:
        sel = shank == s
        ax.scatter(bp[sel], locs[sel, 1], s=12, label=f"shank {s}")
    ax.set_xlabel(f"{f_lo:.0f}-{f_hi:.0f} Hz power")
    ax.set_ylabel("Depth (µm)")
    ax.set_title("Band power by depth")
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)
with c2:
    fmask = (freqs >= 0.25) & (freqs <= 300)
    fig, ax = plt.subplots(figsize=(7, 5))
    for s in shanks:
        ch = int(picks[s][0])
        ax.semilogy(freqs[fmask], psd[fmask, ch], label=f"shank {s} — ch {ch}")
    ax.axvspan(*passband, color="grey", alpha=0.2)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD")
    ax.set_title("PSD of peak channel per shank")
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)

# --- envelope + detection on selected channel ---
trace = traces[:, channel]
filtered, envelope = bandpass_envelope(trace, fs, passband, order=filter_order)
events, z = detect_events(envelope, fs, boundary_sd, peak_sd, dur_min / 1000, dur_max / 1000)
n_show = min(int(min(env_plot_s, t_dur) * fs), len(z))
t_axis = np.arange(n_show) / fs

st.subheader(f"Channel {channel} — {len(events)} events  ·  boundary {boundary_sd:g} SD, peak {peak_sd:g} SD")

thr_z = np.where(z > boundary_sd, z, 0.0)
valid_z = np.zeros_like(z)
for s, e in events:
    valid_z[s:e] = z[s:e]

for title, y in [("Envelope (z-scored)", z),
                 ("Thresholded envelope", thr_z),
                 ("Duration-valid events only", valid_z)]:
    fig, ax = plt.subplots(figsize=(12, 2.5))
    ax.plot(t_axis, y[:n_show], lw=0.6)
    ax.axhline(boundary_sd, color="orange", ls="--", lw=0.8, label=f"boundary {boundary_sd:g} SD")
    ax.axhline(peak_sd, color="red", ls="--", lw=0.8, label=f"peak {peak_sd:g} SD")
    ax.legend(loc="upper right", fontsize=7)
    ax.set_title(title)
    ax.set_ylabel("Amplitude (SD)")
    ax.set_xlabel("Time (s)")
    st.pyplot(fig)
    plt.close(fig)

# --- event browser (interactive) ---
def _step_event(delta, n):
    st.session_state.event_idx = int(min(max(st.session_state.event_idx + delta, 0), n - 1))


st.subheader("Event browser")
if len(events) == 0:
    st.info("No events detected with current parameters.")
else:
    st.session_state.setdefault("event_idx", 0)
    st.session_state.event_idx = min(st.session_state.event_idx, len(events) - 1)
    c_prev, c_next, c_slider = st.columns([1, 1, 6])
    c_prev.button("◀ Prev", on_click=_step_event, args=(-1, len(events)), use_container_width=True)
    c_next.button("Next ▶", on_click=_step_event, args=(1, len(events)), use_container_width=True)
    if len(events) > 1:
        c_slider.slider("Event index", 0, len(events) - 1, key="event_idx")
    idx = st.session_state.event_idx
    s, e = events[idx]

    t_ev, seg = peri_event_trace(trace, events[idx], fs, event_window_s)
    _, seg_filt = peri_event_trace(filtered, events[idx], fs, event_window_s)
    df = pd.DataFrame({"time_s": t_ev, "raw": seg, "filtered": seg_filt})
    raw_line = alt.Chart(df).mark_line(strokeWidth=0.7, color="#4C78A8").encode(
        x=alt.X("time_s:Q", title="Time (s)"),
        y=alt.Y("raw:Q", title="Raw LFP", axis=alt.Axis(titleColor="#4C78A8")))
    filt_line = alt.Chart(df).mark_line(strokeWidth=0.9, color="#F58518").encode(
        x="time_s:Q",
        y=alt.Y("filtered:Q", title=f"Bandpass {f_lo:.0f}-{f_hi:.0f} Hz", axis=alt.Axis(titleColor="#F58518")))
    start = alt.Chart(pd.DataFrame({"t": [s / fs]})).mark_rule(color="green", size=2).encode(x="t:Q")
    end = alt.Chart(pd.DataFrame({"t": [e / fs]})).mark_rule(color="red", size=2).encode(x="t:Q")
    chart = alt.layer(raw_line, filt_line, start, end).resolve_scale(y="independent").interactive(
        bind_y=False).properties(title=f"Event {idx} / {len(events) - 1} — {(e - s) / fs * 1000:.0f} ms", height=350)
    st.altair_chart(chart, use_container_width=True)
    st.caption("🔵 raw LFP (left axis)  ·  🟠 bandpass-filtered (right axis)  ·  🟢 start  🔴 end  ·  drag/scroll = pan/zoom (x)")
