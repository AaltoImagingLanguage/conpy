"""
Power mapping using DICS.
"""
from __future__ import print_function
import argparse

import mne
from mne.time_frequency import read_csd
from mne.beamformer import make_dics, apply_dics_csd
from mayavi import mlab
mlab.options.offscreen = True  # Don't open a window when rendering figure

from config import (fname, freq_bands, conditions, reg)

# Be verbose
mne.set_log_level('INFO')

# Handle command line arguments
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('subject', metavar='sub###', help='The subject to process')
args = parser.parse_args()
subject = args.subject
print('Processing subject:', subject)

# Open the HTML report
report = mne.open_report(fname.report(subject=subject))

# Read the forward model
fwd = mne.read_forward_solution(fname.fwd(subject=subject))

# Read the info structure
info = mne.io.read_info(fname.epo(subject=subject))
info = mne.pick_info(info, mne.pick_types(info, meg='grad'))

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

# Plot difference in power between the two experimental conditions, relative to
# the baseline power, and save them to the HTML report
stc_contrast = (stcs[conditions[0]] - stcs[conditions[1]]) / stcs['baseline']
figs = []
for i, freq in enumerate(freq_bands):
    fig = mlab.figure(size=(300, 300))
    mlab.clf()
    brain3 = stc_contrast.plot(subject='sub002', hemi='both',
                               background='white', foreground='black',
                               time_label='', colormap='mne', initial_time=i,
                               title='%s-%s Hz' % freq, figure=fig)
    mlab.view(-90, 110, 420, [0, 0, 0], figure=fig)
    figs.append(fig)

with mne.open_report(fname.report(subject=subject)) as report:
    report.add_slider_to_section(
        figs,
        ['%s-%s Hz' % freq for freq in freq_bands],
        title='Power contrast',
        section='Source-level',
        replace=True
    )
    report.save(fname.report_html(subject=subject), overwrite=True,
                open_browser=False)
