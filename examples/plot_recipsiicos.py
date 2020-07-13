# -*- coding: utf-8 -*-
"""
ReciPSIICOS test
================================================

Testing the RciPSIICOS enhancement to LCMV beamformers [1]_. Supposedly we can use
it to detect correlated sources in the brain.

"""
# Author: Marijn van Vliet <w.m.vanvliet@gmail.com>
#
# License: BSD (3-clause)

###############################################################################
# Setup
# -----
# We first import the required packages to run this and define a list of
# filenames for various things we'll be using.
import os.path as op
import numpy as np
from scipy.stats import pearsonr
from mayavi import mlab
from matplotlib import pyplot as plt

import mne
from mne.datasets import sample
from mne.minimum_norm import make_inverse_operator, apply_inverse
from mne.beamformer import make_lcmv, apply_lcmv_cov

# Suppress irrelevant output
mne.set_log_level('ERROR')

# We use the MEG and MRI setup from the MNE-sample dataset
data_path = sample.data_path(download=False)
subjects_dir = op.join(data_path, 'subjects')
mri_path = op.join(subjects_dir, 'sample')

# Filenames for various files we'll be using
meg_path = op.join(data_path, 'MEG', 'sample')
trans_fname = op.join(meg_path, 'sample_audvis_raw-trans.fif')
fwd_fname = op.join(meg_path, 'sample_audvis-meg-eeg-oct-6-fwd.fif')

# Seed for the random number generator
np.random.seed(42)

###############################################################################
# Data simulation
# ---------------
#
# The following function generates a timeseries that contains an oscillator,
# whose frequency fluctuates a little over time, but stays close to 10 Hz.
# We'll use this function to generate our two signals.

sfreq = 50  # Sampling frequency of the generated signal
times = np.arange(10. * sfreq) / sfreq  # 10 seconds of signal


def coh_signal_gen():
    """Generate an oscillating signal.

    Returns
    -------
    signal : ndarray
        The generated signal.
    """
    t_rand = 0.001  # Variation in the instantaneous frequency of the signal
    std = 0.1  # Std-dev of the random fluctuations added to the signal
    base_freq = 10.  # Base frequency of the oscillators in Hertz
    n_times = len(times)

    # Generate an oscillator with varying frequency and phase lag.
    iflaw = base_freq / sfreq + t_rand * np.random.randn(n_times)
    signal = np.exp(1j * 2.0 * np.pi * np.cumsum(iflaw))
    signal *= np.conj(signal[0])
    signal = signal.real

    # Add some random fluctuations to the signal.
    signal += std * np.random.randn(n_times)
    signal *= 1e-7

    return signal


###############################################################################
# Let's simulate two timeseries and plot some basic information about them.
signal1 = coh_signal_gen()
signal2 = coh_signal_gen()

plt.figure(figsize=(8, 4))

# # Plot the timeseries
# plt.subplot(211)
# plt.plot(times, signal1)
# plt.xlabel('Time (s)')
# plt.title('Signal 1')
# plt.subplot(212)
# plt.plot(times, signal2)
# plt.xlabel('Time (s)')
# plt.title('Signal 2')
# plt.tight_layout()

# Compute the correlation between the two timeseries
corr, _ = pearsonr(signal1, signal2)
print('Correlation between signal1 and signal2:', corr)

###############################################################################
# Now we put the signals at two locations on the cortex. We construct a
# :class:`mne.SourceEstimate` object to store them in.

# The locations on the cortex where the signal will originate from. These
# locations are indicated as vertex numbers.
source_vert1 = 146374
source_vert2 = 33830

# Construct a SourceEstimate object that describes the signal at the cortical
# level.
stc = mne.SourceEstimate(
    np.vstack((signal1, signal2)),  # The two signals
    vertices=[[source_vert1], [source_vert2]],  # Their locations
    tmin=0,
    tstep=1. / sfreq,
    subject='sample',  # We use the brain model of the MNE-Sample dataset
)

###############################################################################
# Before we simulate the sensor-level data, let's define a signal-to-noise
# ratio. You are encouraged to play with this parameter and see the effect of
# noise on our results.
SNR = 0.1  # Signal-to-noise ratio. Decrease to add more noise.

###############################################################################
# Now we run the signal through the forward model to obtain simulated sensor
# data. To save computation time, we'll only simulate gradiometer data. You can
# try simulating other types of sensors as well.

# Load the forward model
fwd = mne.read_forward_solution(fwd_fname)
fwd = mne.convert_forward_solution(fwd, surf_ori=True)

# Use only magneto
fwd = mne.pick_types_forward(fwd, meg='mag', eeg=False, exclude='bads')

# Create an info object that holds information about the sensors (their
# location, etc.).
info = mne.create_info(fwd['info']['ch_names'], sfreq, ch_types='grad')
info.update(fwd['info'])

# To simulate the data, we need a version of the forward solution where each
# source has a "fixed" orientation, i.e. pointing orthogonally to the surface
# of the cortex.
fwd_fixed = mne.convert_forward_solution(fwd, force_fixed=True)

# Now we can run our simulated signal through the forward model, obtaining
# simulated sensor data.
sensor_data = mne.apply_forward_raw(fwd_fixed, stc, info).get_data()

# We're going to add some noise to the sensor data
noise = np.random.randn(*sensor_data.shape)

# Scale the noise to be in the ballpark of MEG data
noise_scaling = np.linalg.norm(sensor_data) / np.linalg.norm(noise)
noise *= noise_scaling

# Mix noise and signal with the given signal-to-noise ratio.
sensor_data = SNR * sensor_data + noise

###############################################################################
# We create an :class:`mne.EpochsArray` object containing two trials: one with
# just noise and one with both noise and signal. The techniques we'll be
# using in this tutorial depend on being able to contrast data that contains
# the signal of interest versus data that does not.
epochs = mne.EpochsArray(
    data=np.concatenate(
        (noise[np.newaxis, :, :],
         sensor_data[np.newaxis, :, :]),
        axis=0),
    info=info,
    events=np.array([[0, 0, 1], [10, 0, 2]]),
    event_id=dict(noise=1, signal=2),
)

# Plot the simulated data
epochs.plot()

###############################################################################
# Power mapping
# -------------
# With our simulated dataset ready, we can now pretend to be researchers that
# have just recorded this from a real subject and are going to study what parts
# of the brain communicate with each other.
#
# First, we'll create a source estimate of the MEG data. We'll use both a
# straightforward MNE-dSPM inverse solution for this, and the DICS beamformer
# which is specifically designed to work with oscillatory data.

###############################################################################
# Computing the inverse using MNE-dSPM:

# Estimating the noise covariance on the trial that only contains noise.
cov = mne.compute_covariance(epochs['noise'])
inv = make_inverse_operator(epochs.info, fwd, cov)

# Apply the inverse model to the trial that also contains the signal.
s = apply_inverse(epochs['signal'].average(), inv)

# # Take the root-mean square along the time dimension and plot the result.
# s_rms = (s ** 2).mean()
# brain = s_rms.plot('sample', subjects_dir=subjects_dir, hemi='both', figure=1,
#                    size=400)
# 
# # Indicate the true locations of the source activity on the plot.
# brain.add_foci(source_vert1, coords_as_verts=True, hemi='lh')
# brain.add_foci(source_vert2, coords_as_verts=True, hemi='rh')
# 
# # Rotate the view and add a title.
# mlab.view(0, 0, 550, [0, 0, 0])
# mlab.title('MNE-dSPM inverse (RMS)', height=0.9)

###############################################################################
# Computing a cortical power map using an LCMV beamformer:

# Estimate the data covariance on the trial containing the signal.
cov_signal = mne.compute_covariance(epochs['signal'])

# Compute the LCMV powermap. For this simulated dataset, we need a lot of
# regularization for the beamformer to behave properly. For real recordings,
# this amount of regularization is probably too much.
filters = make_lcmv(epochs.info, fwd, cov_signal, reg=1, pick_ori='normal', recipsiicos=100)
power = apply_lcmv_cov(cov_signal, filters)

# Plot the LCMV power map.
brain = power.plot('sample', subjects_dir=subjects_dir, hemi='both', figure=2,
                   size=400)

# Indicate the true locations of the source activity on the plot.
brain.add_foci(source_vert1, coords_as_verts=True, hemi='lh')
brain.add_foci(source_vert2, coords_as_verts=True, hemi='rh')

# Rotate the view and add a title.
mlab.view(0, 0, 550, [0, 0, 0])
mlab.title('LCMV power map', height=0.9)

###############################################################################
# Excellent! Both methods found our two simulated sources. Of course, with a
# signal-to-noise ratio (SNR) of 1, it isn't very hard to find them. You can
# try playing with the SNR and see how the MNE-dSPM and LCMV results hold up in
# the presense of increasing noise.

###############################################################################
# References
# ----------
# .. [1] Kuznetsova A., Nurislamova J. and Ossadtchi A., Modified covariance
#        beamformer for solving MEG inverse problem in the environment with
#        correlated sources. BioRXiV, 668814. https://doi.org/10.1101/668814 
