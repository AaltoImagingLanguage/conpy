"""
Power mapping using DICS.
"""
from __future__ import print_function
import argparse

import mne
from mne.time_frequency import read_csd
from mne.beamformer import make_dics, apply_dics_csd

from config import (fname, freq_bands, conditions, reg)

# Be verbose
mne.set_log_level('INFO')

# Handle command line arguments
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('subject', metavar='sub###', help='The subject to process')
args = parser.parse_args()
subject = args.subject
print('Processing subject:', subject)

# Read the forward model
fwd = mne.read_forward_solution(fname.fwd(subject=subject))

# Read the info structure
info = mne.io.read_info(fname.epo(subject=subject))

# Compute source power for all frequency bands and all conditions
fmin = [f[0] for f in freq_bands]
fmax = [f[1] for f in freq_bands]

csds = dict()
for condition in conditions + ['baseline']:
    print('Reading CSD matrix for condition:', condition)
    # Read the CSD matrix
    csds[condition] = read_csd(fname.csd(condition=condition, subject=subject))

# Average the CSDs of both conditions
csd = csds['face'].copy()
csd._data += csds['scrambled']._data
csd._data /= 2

# Compute the DICS beamformer using the CSDs from both conditions, for all
# frequency bands.
filters = make_dics(info=info, forward=fwd, csd=csd.mean(fmin, fmax), reg=reg,
                    pick_ori='max-power')

stcs = dict()
for condition in conditions + ['baseline']:
    print('Computing power for condition:', condition)
    stcs[condition], _ = apply_dics_csd(csds[condition].mean(fmin, fmax),
                                        filters)
    stcs[condition].save(fname.power(condition=condition, subject=subject))
