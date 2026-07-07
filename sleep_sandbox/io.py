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
