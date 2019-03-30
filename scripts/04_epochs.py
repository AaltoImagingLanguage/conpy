"""
Cut signal into epochs.
"""
import argparse
import mne
from mne.preprocessing import read_ica

from config import (fname, events_id, epoch_tmin, epoch_tmax, baseline,
                    bandpass_fmin, bandpass_fmax, reject)

# Be verbose
mne.set_log_level('INFO')

# Handle command line arguments
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('subject', metavar='sub###', help='The subject to process')
args = parser.parse_args()
subject = args.subject
print('Processing subject:', subject)

# Construct a raw object that will load the bandpass-filtered data.
raw = mne.io.read_raw_fif(
    fname.filt(subject=subject, run=1, fmin=bandpass_fmin, fmax=bandpass_fmax),
    preload=False)
for run in range(2, 7):
    raw.append(mne.io.read_raw_fif(
        fname.filt(subject=subject, run=run,
                   fmin=bandpass_fmin, fmax=bandpass_fmax),
        preload=False))

# Read events from the stim channel
mask = 4096 + 256  # mask for excluding high order bits
events = mne.find_events(raw, stim_channel='STI101', consecutive='increasing',
                         mask=mask, mask_type='not_and', min_duration=0.003)

# Compensate for projector delay
events[:, 0] += int(round(0.0345 * raw.info['sfreq']))

# Load the ICA object
print('  Using ICA')
ica = read_ica(fname.ica(subject=subject))

# Make epochs. Because the original 1000Hz sampling rate is a bit excessive
# for what we're going for, we only read every 5th sample. This gives us a
# sampling rate of ~200Hz.
epochs = mne.Epochs(raw, events, events_id, epoch_tmin, epoch_tmax,
                    baseline=baseline, decim=5, preload=True)

# Save evoked plot to the report
with mne.open_report(fname.report(subject=subject)) as report:
    report.add_figs_to_section(
        [epochs.average().plot(show=False)],
        ['Evoked without ICA'],
        section='Sensor-level'
    )
    report.save_html(fname.report_html(subject=subject), overwrite=True)

# Apply ICA to the epochs, dropping components that correlate with ECG and EOG
ica.apply(epochs)

# Drop epochs that have too large signals (most likely due to the subject
# moving or muscle artifacts)
epochs.drop_bad(reject)
print('  Dropped %0.1f%% of epochs' % (epochs.drop_log_stats(),))

print('  Writing to disk')
epochs.save(fname.epo(subject=subject))

# Save evoked plot to report
with mne.open_report(fname.report(subject=subject)) as report:
    report.add_figs_to_section(
        [epochs.average().plot(show=False)],
        ['Evoked with ICA'],
        section='Sensor-level',
        replace=True
    )
    report.save_html(fname.report_html(subject=subject), overwrite=True,
                     open_browser=False)
