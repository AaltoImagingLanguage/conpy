"""
Perform bandpass filtering.
"""
import argparse
import mne

from config import fname, bandpass_fmin, bandpass_fmax, n_jobs

# Be verbose
mne.set_log_level('INFO')

# Handle command line arguments
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('subject', metavar='sub###', help='The subject to process')
args = parser.parse_args()
subject = args.subject
print('Processing subject:', subject)

for run in range(1, 7):
    # Load the SSS transformed data.
    raw = mne.io.read_raw_fif(fname.sss(subject=subject, run=run),
                              preload=True)

    # The EOG and ECG channels have been erroneously flagged as EEG channels.
    raw.set_channel_types({'EEG061': 'eog',
                           'EEG062': 'eog',
                           'EEG063': 'ecg',
                           'EEG064': 'misc'})  # EEG064 free-floating el.
    raw.rename_channels({'EEG061': 'EOG061',
                         'EEG062': 'EOG062',
                         'EEG063': 'ECG063'})

    # Drop EEG channels (they are very noisy in this dataset)
    raw.pick_types(meg=True, eeg=False, eog=True, ecg=True, stim=True)

    # Bandpass the data.
    raw_filt = raw.copy().filter(
        bandpass_fmin, bandpass_fmax, l_trans_bandwidth='auto',
        h_trans_bandwidth='auto', filter_length='auto', phase='zero',
        fir_window='hamming', fir_design='firwin', n_jobs=n_jobs)

    # Highpass the EOG channels to > 1Hz, regardless of the bandpass-filter
    # applied to the other channels
    picks_eog = mne.pick_types(raw_filt.info, meg=False, eog=True)
    raw_filt.filter(
        1., None, picks=picks_eog, l_trans_bandwidth='auto',
        filter_length='auto', phase='zero', fir_window='hann',
        fir_design='firwin', n_jobs=n_jobs)

    f = fname.filt(subject=subject, run=run,
                   fmin=bandpass_fmin, fmax=bandpass_fmax)
    raw_filt.save(f, overwrite=True)
