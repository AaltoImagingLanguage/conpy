# encoding: utf-8
"""
Makes a plot of the CSD matrix.

Author: Marijn van Vliet <w.m.vanvliet@gmail.com>
"""
from matplotlib import pyplot as plt
import mne
from mne.time_frequency import read_csd, pick_channels_csd

from config import fname, subjects, freq_bands

info = mne.io.read_info(fname.epo(subject=subjects[0]))
grads = [info['ch_names'][ch] for ch in mne.pick_types(info, meg='grad')]
csd = read_csd(fname.csd(subject=subjects[0], condition='face'))
csd = pick_channels_csd(csd, grads)
csd = csd.mean([f[0] for f in freq_bands], [f[1] for f in freq_bands])

# Plot theta, alpha, low beta
csd[:3].plot(info, n_cols=3, show=False)
plt.savefig('../paper/figures/csd1.pdf', bbox_inches='tight')

# Plot high beta 1, high beta 2 and low gamma
csd[3:].plot(info, n_cols=3, show=False)
plt.savefig('../paper/figures/csd2.pdf', bbox_inches='tight')
