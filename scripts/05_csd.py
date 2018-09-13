"""
Compute cross-spectral density (CSD) matrices
"""
from __future__ import print_function
import warnings
import argparse

import numpy as np
import mne
from mne.time_frequency import csd_morlet

from config import (fname, n_jobs, csd_tmin, csd_tmax, freq_bands, conditions,
                    get_report, save_report)

# Be verbose
mne.set_log_level('INFO')

# Handle command line arguments
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('subject', metavar='sub###', help='The subject to process')
args = parser.parse_args()
subject = args.subject
print('Processing subject:', subject)

# Read the epochs
print('Reading epochs...')
epochs = mne.read_epochs(fname.epo(subject=subject))

report = get_report(subject)

# Suppress warning about wavelet length.
warnings.simplefilter('ignore')

# Individual frequencies to estimate the CSD for
fmin = freq_bands[0][0]
fmax = freq_bands[-1][1]
frequencies = np.arange(fmin, fmax + 1, 2)

# Compute CSD matrices for each frequency and each condition.
for condition in conditions:
    print('Condition:', condition)
    # Remove the mean during the time interval for which we compute the CSD
    epochs_baselined = epochs[condition].apply_baseline((csd_tmin, csd_tmax))

    # Compute CSD for the desired time interval
    csd = csd_morlet(epochs_baselined, frequencies=frequencies, tmin=csd_tmin,
                     tmax=csd_tmax, decim=20, n_jobs=n_jobs, verbose=True)

    # Save the CSD matrices
    csd.save(fname.csd(condition=condition, subject=subject))
    report.add_figs_to_section(csd.plot(show=False),
                               ['CSD for %s' % condition],
                               section='Sensor-level')

# Also compute the CSD for the baseline period (use all epochs for this,
# regardless of condition). This way, we can compare the change in power caused
# by the presentation of the stimulus.
epochs = epochs.apply_baseline((-0.2, 0))  # Make sure data is zero-mean
csd_baseline = csd_morlet(epochs, frequencies=frequencies, tmin=-0.2, tmax=0,
                          decim=20, n_jobs=n_jobs, verbose=True)
csd_baseline.save(fname.csd(condition='baseline', subject=subject))
report.add_figs_to_section(csd_baseline.plot(show=False), ['CSD for baseline'],
                           section='Sensor-level')

save_report(report)
