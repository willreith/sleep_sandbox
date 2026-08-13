"""Cross-check REM detections against each other, since there is no ground truth.

Scores every probe x variant x theta convention with the theta threshold on the ~nrem & ~mov pool,
then reports: the NREM span per combo (ProbeA/ProbeB are simultaneous probes on one session, so
their sleep blocks should coincide), pairwise REM Dice among the cases that find any REM, and
whether ProbeB's REM falls inside ProbeA's own sleep block. The last of these is what establishes
that ProbeA's zero-REM is a detection failure rather than an absence of REM.

Usage: python check_rem_agreement.py
"""

import warnings
import argparse
import itertools
from pathlib import Path

import numpy as np
import yaml

from sleep_sandbox.analysis import smooth_norm
from sleep_sandbox.scoring import classify

repo_root = Path(__file__).resolve().parent.parent
parser = argparse.ArgumentParser()
parser.add_argument('--seg', required=True,
                     help="segment range, e.g. 'seg5-148'; reads under data/derivatives/{seg}/")
args = parser.parse_args()
deriv_base = repo_root / 'data/derivatives' / args.seg

cfg = yaml.safe_load(open(repo_root / 'config/sleep_scoring.yml'))
s_s, w = cfg['spectrogram']['step_s'], cfg['smoothing']['window_s']
m, g = cfg['threshold']['method'], cfg['threshold']['kde_grid_n']
bs, bm = cfg['bimodal_threshold']['startbins'], cfg['bimodal_threshold']['maxbins']
dur = cfg['duration_criteria']

res = {}
for p in ['ProbeA', 'ProbeB']:
    for v in ['lfp_cmr', 'lfp_nocmr']:
        d = deriv_base / p / v
        r = dict(np.load(d / 'result.npz')); e = dict(np.load(d / 'result_extras.npz'))
        swt = yaml.safe_load(open(d / 'result.yml'))['result_scalars']['sw_thresh']
        sw, mo, t = r['sw_metric'], r['motion_metric'], r['times']
        for c in cfg['theta']['conventions']:
            th = smooth_norm(e[f'theta_own_ratio_{c}'], step_s=s_s, win_s=w)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                s = classify(sw, th, mo, swt, bs, bm, m, g, dt=1.0,
                             merge_shorter_than_s=dur['merge_shorter_than_s'],
                             min_state_s=dur['min_state_s'],
                             microarousal_max_s=dur['microarousal_max_s'], theta_conditioned=True)
            res[(p, v, c)] = (s, t)

name = lambda k: f"{k[0][-1]}/{k[1].replace('lfp_', '')}/{k[2]}"
print('NREM span (first -> last NREM epoch) -- ProbeA/ProbeB are simultaneous probes, same session:')
for p in ['ProbeA', 'ProbeB']:
    for v in ['lfp_cmr', 'lfp_nocmr']:
        s, t = res[(p, v, 'watson')]; idx = np.flatnonzero(s['nrem'])
        print(f'   {p}/{v:10s} {t[idx[0]]:7.0f}s -> {t[idx[-1]]:7.0f}s   nrem frac={s["nrem"].mean():.3f}')

work = [k for k, (s, _) in res.items() if s['rem'].any()]
print(f'\ncases with REM: {len(work)}/12 -> {[name(k) for k in work]}')
print('\npairwise REM Dice (temporal overlap):')
for a, b in itertools.combinations(work, 2):
    ra, rb = res[a][0]['rem'], res[b][0]['rem']
    n = min(len(ra), len(rb))
    dice = 2 * (ra[:n] & rb[:n]).sum() / max(ra[:n].sum() + rb[:n].sum(), 1)
    print(f'   {name(a):20s} vs {name(b):20s} dice={dice:.3f}')

# Does ProbeB REM fall inside ProbeA's NREM-flanked sleep block?
sA, tA = res[('ProbeA', 'lfp_cmr', 'watson')]
iA = np.flatnonzero(sA['nrem']); loA, hiA = tA[iA[0]], tA[iA[-1]]
for k in work:
    s, t = res[k]; rt = t[s['rem']]
    print(f'   {name(k):20s} REM within ProbeA sleep block [{loA:.0f},{hiA:.0f}]: '
          f'{float(((rt >= loA) & (rt <= hiA)).mean()):.3f}')
