"""
Compute average of the power maps.
"""
from __future__ import print_function
import argparse

import numpy as np
import mne

from config import fname, conditions, subjects

# Be verbose
mne.set_log_level('INFO')

# Handle command line arguments (only --help)
parser = argparse.ArgumentParser(description=__doc__)
args = parser.parse_args()

ga_stcs = list()
for condition in conditions + ['baseline']:
    print('Processing condition:', condition)
    stcs = list()

    for subject in subjects:
        print('Processing subject:', subject)
        stc_fname = fname.power(condition=condition, subject=subject)
        stc_subject = mne.read_source_estimate(stc_fname)

        # Morph the STC to the fsaverage brain.
        stc_subject.subject = subject
        stc_fsaverage = stc_subject.morph('fsaverage',
                                          subjects_dir=fname.subjects_dir)
        stcs.append(stc_fsaverage)

    # Average the source estimates
    data = np.mean([stc.data for stc in stcs], axis=0)
    ga_stc = mne.SourceEstimate(data, vertices=stcs[0].vertices,
                                tmin=stcs[0].tmin, tstep=stcs[0].tstep)
    ga_stc.save(fname.ga_power(condition=condition))
    ga_stcs.append(ga_stc)

# Compute contrast between face and scrambled. Compare against the baseline
# power.
ga_contrast = (ga_stcs[0] - ga_stcs[1]) / ga_stcs[-1]
ga_contrast.save(fname.ga_power(condition='contrast'))
