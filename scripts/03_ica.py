"""
Use ICA to remove ECG and EOG artifacts from the data.
"""
import argparse
import numpy as np
import mne
from mne.preprocessing import ICA, create_ecg_epochs, create_eog_epochs
from config import (fname, bandpass_fmin, bandpass_fmax, n_ecg_components,
                    n_eog_components)

# Be verbose
mne.set_log_level('INFO')

# Handle command line arguments
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('subject', metavar='sub###', help='The subject to process')
args = parser.parse_args()
subject = args.subject
print('Processing subject:', subject)

# Construct a raw object that will load the highpass-filtered data.
raw = mne.io.read_raw_fif(
    fname.filt(subject=subject, run=1, fmin=bandpass_fmin, fmax=bandpass_fmax),
    preload=False)
for run in range(2, 7):
    raw.append(mne.io.read_raw_fif(
        fname.filt(subject=subject, run=run,
                   fmin=bandpass_fmin, fmax=bandpass_fmax),
        preload=False))

# SSS reduces the data rank and the noise levels, so let's include
# components based on a higher proportion of variance explained (0.999)
# than we would otherwise do for non-Maxwell-filtered raw data (0.98)
n_components = 0.999

# Define the parameters for the ICA. There is a random component to this.
# We set a specific seed for the random state, so this script produces exactly
# the same results every time.
print('Fitting ICA')
ica = ICA(method='fastica', random_state=42, n_components=n_components)

# To compute the ICA, we don't need all data. We just need enough data to
# perform the statistics to a reasonable degree. Here, we use every 11th
# sample. We also apply some rejection limits to discard epochs with overly
# large signals that are likely due to the subject moving about.
ica.fit(raw, reject=dict(grad=4000e-13, mag=4e-12), decim=11)
print('Fit %d components (explaining at least %0.1f%% of the variance)'
      % (ica.n_components_, 100 * n_components))

# Find onsets of heart beats and blinks. Create epochs around them
ecg_epochs = create_ecg_epochs(raw, tmin=-.3, tmax=.3, preload=False)
eog_epochs = create_eog_epochs(raw, tmin=-.5, tmax=.5, preload=False)

# Find ICA components that correlate with heart beats.
ecg_epochs.decimate(5)
ecg_epochs.load_data()
ecg_epochs.apply_baseline((None, None))
ecg_inds, ecg_scores = ica.find_bads_ecg(ecg_epochs, method='ctps')
ecg_scores = np.abs(ecg_scores)
rank = np.argsort(ecg_scores)[::-1]
rank = [r for r in rank if ecg_scores[r] > 0.05]
ica.exclude = rank[:n_ecg_components]
print('    Found %d ECG indices' % (len(ecg_inds),))

# Find ICA components that correlate with eye blinks
eog_epochs.decimate(5)
eog_epochs.load_data()
eog_epochs.apply_baseline((None, None))
eog_inds, eog_scores = ica.find_bads_eog(eog_epochs)
eog_scores = np.max(np.abs(eog_scores), axis=0)
# Remove all components with a correlation > 0.1 to the EOG channels and that
# have not already been flagged as ECG components
rank = np.argsort(eog_scores)[::-1]
rank = [r for r in rank if eog_scores[r] > 0.1 and r not in ecg_inds]
ica.exclude += rank[:n_eog_components]
print('    Found %d EOG indices' % (len(eog_inds),))

# Save the ICA decomposition
ica.save(fname.ica(subject=subject))
