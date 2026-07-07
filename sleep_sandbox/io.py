"""IO for ephys recordings: raw-segment discovery, remote-mount detection, and
param-keyed save/load of preprocessed SpikeInterface derivatives."""

import re
import json
import hashlib
import warnings

import numpy as np
import spikeinterface.full as si

from pathlib import Path

_SUFFIX_RE = re.compile(r"AmplifierData_(\d+)\.bin$")


def find_amplifier_files(session_dir, probe, suffixes=None):
    """Return (amplifier_paths, probe_config) for one probe in a session directory.

    session_dir directly contains the probe config JSON and a subdir with the .bin files.
    probe is the filename token, e.g. 'ProbeA' / 'ProbeB'. suffixes selects by the trailing
    _N index: None = all, or an int / range / list / array of ints. amplifier_paths is sorted
    by N; probe_config is the single matching JSON (all configs for one probe are identical)."""
    session_dir = Path(session_dir)

    keep = None if suffixes is None else {int(s) for s in np.atleast_1d(suffixes)}
    amp = []
    for p in session_dir.rglob("*AmplifierData_*.bin"):
        m = _SUFFIX_RE.search(p.name)
        if m is None or probe not in p.name:
            continue
        n = int(m.group(1))
        if keep is None or n in keep:
            amp.append((n, p))
    amp_paths = [p for _, p in sorted(amp)]

    if not amp_paths:
        raise FileNotFoundError(
            f"No {probe} AmplifierData files under {session_dir} for suffixes={suffixes}"
        )

    configs = sorted(p for p in session_dir.glob("*.json") if probe in p.name)
    probe_config = configs[0] if configs else None
    return amp_paths, probe_config


def remote_available(root):
    """True if the remote raw-data root is reachable (mounted). None/unset -> False."""
    return root is not None and Path(root).exists()


# --- Param-keyed save/load of preprocessed derivatives -----------------------
# The folder name encodes the preprocessing chain so different parameter sets never
# collide and can be looked up by their params.

def _readable_tag(steps):
    parts = []
    for name, kw in steps:
        if name == "bandpass_filter":
            parts.append(f"bp{kw['freq_min']}-{kw['freq_max']}")
            if "margin_ms" in kw:
                parts.append(f"mg{kw['margin_ms']}")
        elif name == "common_reference":
            parts.append(f"car{kw['reference'].upper()}{kw['operator']}")
        elif name == "resample":
            parts.append(f"rs{kw['resample_rate']}")
    return "_".join(parts)


def _canonical(source, steps):
    return {
        "source": source,
        "steps": [{"name": n, "kwargs": {k: kw[k] for k in sorted(kw)}} for n, kw in steps],
    }


def get_or_build(recording, source, steps, base_dir, stream, verify=True, **save_kwargs):
    """Load the preprocessed recording keyed by its params, else save it; returns the on-disk recording."""
    canon = _canonical(source, steps)
    h = hashlib.sha1(json.dumps(canon, sort_keys=True).encode()).hexdigest()[:8]
    folder = Path(base_dir) / source / f"{stream}__{_readable_tag(steps)}__{h}"

    if folder.exists():
        if verify:
            saved = json.loads((folder / "params.json").read_text())
            if saved != canon:
                raise ValueError(f"Param mismatch at {folder}\n saved={saved}\n wanted={canon}")
        print(f"Loaded existing {stream}: {folder}")
        return si.load(folder)

    print(f"Building {stream} -> {folder}")
    folder.parent.mkdir(parents=True, exist_ok=True)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="There is no Probe attached", category=UserWarning)
        rec = recording.save(folder=folder, **save_kwargs)
    (folder / "params.json").write_text(json.dumps(canon, sort_keys=True, indent=2))
    return rec


def load_preprocessed(parent, stream):
    """Load a preprocessed derivative by stream name from a session's derivative parent dir.

    parent is the seg_source folder holding one subdir per stream (e.g. 'lfp__..._<hash>').
    Matches on the '{stream}__' prefix; raises if there are zero or multiple matches."""
    parent = Path(parent)
    matches = sorted(parent.glob(f"{stream}__*"))
    if not matches:
        raise FileNotFoundError(f"No '{stream}' derivative under {parent}")
    if len(matches) > 1:
        raise ValueError(
            f"Multiple '{stream}' derivatives under {parent}: {[m.name for m in matches]}"
        )
    return si.load(matches[0])


# --- BNO055 IMU streams ------------------------------------------------------
# Per-stream .bin files, one per acquisition block (no embedded timestamps; timing is the Clock).
# (dtype, n components per sample), inferred from file sizes + content.
BNO055_CLOCK_HZ = 250e6   # free-running ONIX hardware counter
_BNO055_STREAMS = {
    "Clock":              ("<u8", 1),
    "HubSyncCounter":     ("<u8", 1),
    "Euler":              ("<f4", 3),
    "Quaternion":         ("<f4", 4),
    "GravityVector":      ("<f4", 3),
    "LinearAcceleration": ("<f4", 3),
}


def load_bno055(data_dir, block, streams=None):
    """Load BNO055 per-stream .bin files for one acquisition block; returns {name: array[-1, ncomp]}.

    data_dir is the NeuropixelsV2 dir holding the raw .bin files. streams: None = all, or a subset
    of _BNO055_STREAMS keys. All streams in one block share the same sample count."""
    data_dir = Path(data_dir)
    names = list(_BNO055_STREAMS) if streams is None else streams
    out = {}
    for name in names:
        dtype, ncomp = _BNO055_STREAMS[name]
        arr = np.fromfile(data_dir / f"NeuropixelsV2_Bno055_{name}_{block}.bin", dtype=dtype)
        out[name] = arr.reshape(-1, ncomp)
    return out


def concat_bno055(data_dir, blocks=None, streams=("Clock", "LinearAcceleration", "Quaternion")):
    """Concatenate BNO055 blocks in numeric order into continuous arrays + an abutted time axis.

    blocks: None = all Clock blocks found under data_dir, else an explicit ordered list.
    Returns (data, t, block_edges): data[name] stacked over blocks; t seconds, each block's Clock
    duration abutted end-to-end (inter-block gaps dropped, mirroring the LFP concatenation);
    block_edges is the start sample index of each block. Pure IO/assembly - no signal processing."""
    data_dir = Path(data_dir)
    if blocks is None:
        blocks = sorted(int(re.search(r"_(\d+)\.bin$", p.name).group(1))
                        for p in data_dir.glob("*Bno055_Clock_*.bin"))
    parts = {name: [] for name in streams}
    t_parts, block_edges, offset, n = [], [], 0.0, 0
    for b in blocks:
        blk = load_bno055(data_dir, b, streams)
        clk = blk["Clock"].ravel().astype(np.float64)
        tb = (clk - clk[0]) / BNO055_CLOCK_HZ
        block_edges.append(n)
        for name in streams:
            parts[name].append(blk[name])
        t_parts.append(tb + offset)
        offset += tb[-1] + np.median(np.diff(tb))   # abut: block duration + one sample step
        n += len(tb)
    data = {name: np.concatenate(v, axis=0) for name, v in parts.items()}
    return data, np.concatenate(t_parts), np.array(block_edges)


def align_bno055_to_lfp(data_dir, blocks=None, probe="ProbeB", fs_raw=30000,
                        streams=("Clock", "LinearAcceleration", "Quaternion")):
    """Approach B: place BNO055 samples on the gap-removed LFP concat timeline via the shared clock.

    Bno055_Clock_N and {probe}_Clock_N are the same free-running 250 MHz ONIX counter (one entry per
    ephys sample). Each block's IMU clock span is mapped linearly onto its ephys segment's nominal
    sample-time span (n_ephys / fs_raw seconds), and segments are abutted gap-free -- so inter-block
    gaps the LFP concat dropped are excluded and the nominal-vs-true rate scale is corrected per
    segment. Returns (data, t, valid, block_edges): t = concat time (s) per IMU sample; valid masks
    samples inside their segment's clock span (edge samples falling in a gap are False)."""
    data_dir = Path(data_dir)
    if blocks is None:
        blocks = sorted(int(re.search(r"_(\d+)\.bin$", p.name).group(1))
                        for p in data_dir.glob("*Bno055_Clock_*.bin"))
    parts = {name: [] for name in streams}
    t_parts, valid_parts, block_edges, offset, n = [], [], [], 0.0, 0
    for b in blocks:
        pc = np.memmap(data_dir / f"NeuropixelsV2_{probe}_Clock_{b}.bin", dtype="<u8", mode="r")
        start_c, end_c, n_eph = int(pc[0]), int(pc[-1]), len(pc)
        scale = (n_eph / fs_raw) / (end_c - start_c)   # nominal seconds per clock count, this segment
        blk = load_bno055(data_dir, b, streams)
        c = blk["Clock"].ravel().astype(np.float64)
        block_edges.append(n)
        t_parts.append(offset + (c - start_c) * scale)
        valid_parts.append((c >= start_c) & (c <= end_c))
        for name in streams:
            parts[name].append(blk[name])
        offset += n_eph / fs_raw
        n += len(c)
    data = {name: np.concatenate(v, axis=0) for name, v in parts.items()}
    return data, np.concatenate(t_parts), np.concatenate(valid_parts), np.array(block_edges)
