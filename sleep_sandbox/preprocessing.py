"""Lazy SpikeInterface preprocessing chains that return the recording and its canonical
step list together, so the chain and the params used to key its derivative can't desync.
Also holds IMU (BNO055) kinematics derived from the loaded streams."""

import numpy as np
import spikeinterface.preprocessing as sp

from scipy.integrate import cumulative_trapezoid
from scipy.signal import butter, filtfilt


def build_band(recording, freq_min, freq_max, resample_rate, margin_ms,
               cmr=False, reference=None, operator=None, source_steps=()):
    """Bandpass -> optional common reference -> resample. Returns (rec, steps).

    steps is the canonical list get_or_build hashes, prefixed by source_steps (the provenance
    of the input recording, e.g. the concatenate step). margin_ms feeds the bandpass edge margin."""
    rec = sp.bandpass_filter(recording, freq_min=freq_min, freq_max=freq_max, margin_ms=margin_ms)
    steps = [("bandpass_filter", {"freq_min": freq_min, "freq_max": freq_max, "margin_ms": margin_ms})]
    if cmr:
        rec = sp.common_reference(rec, reference=reference, operator=operator)
        steps.append(("common_reference", {"reference": reference, "operator": operator}))
    rec = sp.resample(rec, resample_rate=resample_rate)
    steps.append(("resample", {"resample_rate": resample_rate}))
    return rec, [*source_steps, *steps]


def imu_kinematics(data, t, block_edges, hp_fc=0.3, hp_order=2):
    """Per-block movement signals from concatenated BNO055 data (processed within each block so the
    high-pass / integration don't run across block seams). Returns (speed_trans, speed_ang, accel),
    each aligned to t:
      speed_trans: high-pass(hp_fc) -> integrate -> high-pass -> |v|   (m/s, rough accel-only proxy)
      speed_ang:   angle between consecutive unit quaternions / dt      (deg/s)
      accel:       |LinearAcceleration|                                 (m/s^2, gravity removed)"""
    lin, quat = data["LinearAcceleration"], data["Quaternion"]
    n = len(t)
    accel = np.linalg.norm(lin, axis=1)
    speed_trans = np.zeros(n)
    speed_ang = np.zeros(n)
    edges = list(block_edges) + [n]
    for s, e in zip(edges[:-1], edges[1:]):
        tb = t[s:e]
        fs = 1.0 / np.median(np.diff(tb))
        b, a = butter(hp_order, hp_fc / (fs / 2), btype="high")
        vel = cumulative_trapezoid(filtfilt(b, a, lin[s:e], axis=0), x=tb, axis=0, initial=0)
        speed_trans[s:e] = np.linalg.norm(filtfilt(b, a, vel, axis=0), axis=1)
        q = quat[s:e] / np.linalg.norm(quat[s:e], axis=1, keepdims=True)
        cos_half = np.abs(np.sum(q[1:] * q[:-1], axis=1)).clip(0, 1)
        speed_ang[s + 1:e] = np.degrees(2 * np.arccos(cos_half)) / np.diff(tb)
    return speed_trans, speed_ang, accel
