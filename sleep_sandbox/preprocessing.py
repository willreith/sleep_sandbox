"""Lazy SpikeInterface preprocessing chains that return the recording and its canonical
step list together, so the chain and the params used to key its derivative can't desync."""

import spikeinterface.preprocessing as sp


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
