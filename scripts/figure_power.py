# encoding: utf-8
"""
This script makes two plots:
1. Makes a plot of the grand-average DICS source power estimate for the alpha
   band.
2. Makes a plot of the grand-average DICS source power estimate, contrasted
   between conditions, for each frequency band.

Author: Marijn van Vliet <w.m.vanvliet@gmail.com>
"""
import mne
from mayavi import mlab

from config import fname, freq_bands

# Load the grand average power maps
stc_face = mne.read_source_estimate(fname.ga_power(condition='face'))
stc_scrambled = mne.read_source_estimate(fname.ga_power(condition='scrambled'))
stc_contrast = mne.read_source_estimate(fname.ga_power(condition='contrast'))

# Show absolute power for both conditions
fig1 = mlab.figure(size=(300, 300))
fig2 = mlab.figure(size=(300, 300))
brain1 = stc_face.plot(
    subject='sub002',
    hemi='split',
    views='med',
    background='white',
    foreground='black',
    time_label='',
    initial_time=1,
    figure=[fig1, fig2],
)
brain1.scale_data_colormap(1e-23, 1.5e-23, 2.4e-23, True)

mlab.savefig('../paper/figures/power_face_lh.png', figure=fig1, magnification=4)
mlab.savefig('../paper/figures/power_face_rh.png', figure=fig2, magnification=4)

fig3 = mlab.figure(size=(300, 300))
fig4 = mlab.figure(size=(300, 300))
brain2 = stc_scrambled.plot(
    subject='fsaverage',
    hemi='split',
    views='med',
    background='white',
    foreground='black',
    time_label='',
    initial_time=1,
    figure=[fig3, fig4],
)
brain2.scale_data_colormap(1e-23, 1.5e-23, 2.5e-23, True)

mlab.savefig('../paper/figures/power_scrambled_lh.png', figure=fig3, magnification=4)
mlab.savefig('../paper/figures/power_scrambled_rh.png', figure=fig4, magnification=4)

# Show difference in power between the two experimental conditions, relative to the baseline power
figs = []
for i, freq in enumerate(freq_bands):
    fig5 = mlab.figure(size=(300, 300))
    mlab.clf()
    #brain3 = stc_contrast.copy().crop(i, i).plot(
    brain3 = stc_contrast.plot(
        subject='sub002',
        hemi='both',
        background='white',
        foreground='black',
        time_label='',
        colormap='mne',
        initial_time=i,
        figure=fig5,
    )
    mlab.view(-90, 110, 420, [0, 0, 0], figure=fig5)
    figs.append(fig5)

    mlab.savefig('../paper/figures/power_contrast_%s-%s-occ.png' % (freq[0], freq[1]), figure=fig5, magnification=4)
