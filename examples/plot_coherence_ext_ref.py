# -*- coding: utf-8 -*-
"""
Coherence analysis with external signal
=======================================

In this tutorial, we're going to simulate one signal originating from a
location on the cortex, and one signal originating from an external sensor.
These signals will also be coherent_, which means they are correlated in the
frequency domain.  We will then use dynamic imaging of coherent sources (DICS)
[1]_ to map out the cortical coherence between all areas of the cortex and the
external sensor and see if we can find the location of the simulated coherent
source.

.. _coherent: https://en.wikipedia.org/wiki/Coherence_(signal_processing)
"""
# Authors: Marijn van Vliet <w.m.vanvliet@gmail.com>
#          Maria Hakonen <maria.hakonen@gmail.com>
#
# License: BSD (3-clause)

###############################################################################
# Setup
# -----
# We first import the required packages to run this tutorial and define a list
# of filenames for various things we'll be using.
import os.path as op

import mne
import numpy as np
from conpy import dics_coherence_external
from matplotlib import pyplot as plt
from mne.beamformer import apply_dics_csd, make_dics
from mne.datasets import sample
from mne.time_frequency import csd_morlet
from scipy.signal import coherence, welch

# We use the MEG and MRI setup from the MNE-sample dataset
data_path = sample.data_path(download=False)
subjects_dir = op.join(data_path, "subjects")
mri_path = op.join(subjects_dir, "sample")

# Filenames for various files we'll be using
meg_path = op.join(data_path, "MEG", "sample")
fwd_fname = op.join(meg_path, "sample_audvis-meg-eeg-oct-6-fwd.fif")

# Setup the random number generator
rng = np.random.RandomState(42)

###############################################################################
# Data simulation
# ---------------
#
# The following function generates a timeseries that contains an oscillator,
# whose frequency fluctuates a little over time, but stays close to 10 Hz.
# We'll use this function to generate our two signals.

sfreq = 50  # Sampling frequency of the generated signal
times = np.arange(10.0 * sfreq) / sfreq  # 10 seconds of signal


def coh_signal_gen():
    """Generate an oscillating signal.

    Returns
    -------
    signal : ndarray, shape (n_times,)
        The generated signal.
    """
    t_rand = 0.001  # Variation in the instantaneous frequency of the signal
    std = 0.1  # Std-dev of the random fluctuations added to the signal
    base_freq = 10.0  # Base frequency of the oscillators in Hertz
    n_times = len(times)

    # Generate an oscillator with varying frequency and phase lag.
    iflaw = base_freq / sfreq + t_rand * rng.standard_normal(n_times)
    signal = np.exp(1j * 2.0 * np.pi * np.cumsum(iflaw))
    signal *= np.conj(signal[0])
    signal = signal.real

    # Add some random fluctuations to the signal.
    signal += std * rng.standard_normal(n_times)
    signal *= 1e-7

    return signal


###############################################################################
# Let's simulate two timeseries and plot some basic information about them.
signal1 = coh_signal_gen()
signal2 = coh_signal_gen()

plt.figure(figsize=(8, 4))

# Plot the timeseries
plt.subplot(221)
plt.plot(times, signal1)
plt.xlabel("Time (s)")
plt.title("Signal 1")
plt.subplot(222)
plt.plot(times, signal2)
plt.xlabel("Time (s)")
plt.title("Signal 2")

# Power spectrum of the first timeseries
f, p = welch(signal1, fs=sfreq, nperseg=128, nfft=256)
plt.subplot(223)
plt.plot(f[:100], p[:100])  # Only plot the first 30 frequencies
plt.xlabel("Frequency (Hz)")
plt.ylabel("Power")
plt.title("Power spectrum of signal 1")

# Compute the coherence between the two timeseries
f, coh = coherence(signal1, signal2, fs=sfreq, nperseg=100, noverlap=64)
plt.subplot(224)
plt.plot(f[:50], coh[:50])
plt.xlabel("Frequency (Hz)")
plt.ylabel("Coherence")
plt.title("Coherence between the timeseries")

plt.tight_layout()

###############################################################################
# Now we put one of the signals at a location on the cortex. We construct a
# :class:`mne.SourceEstimate` object to store it in.

# The location on the cortex where the signal will originate from. This
# location is indicated as a vertex number.
source_vert = 146374

# Construct a SourceEstimate object that describes the signal at the cortical
# level.
stc = mne.SourceEstimate(
    np.atleast_2d(signal1),  # The signal
    vertices=[[source_vert], []],  # Its location
    tmin=0,
    tstep=1.0 / sfreq,
    subject="sample",  # We use the brain model of the MNE-Sample dataset
)

###############################################################################
# Before we simulate the sensor-level data, let's define a signal-to-noise
# ratio. You are encouraged to play with this parameter and see the effect of
# noise on our results.
SNR = 1  # Signal-to-noise ratio. Decrease to add more noise.

###############################################################################
# Now we run the signal through the forward model to obtain simulated sensor
# data. To save computation time, we'll only simulate gradiometer data. You can
# try simulating other types of sensors as well.

# Load the forward model
fwd = mne.read_forward_solution(fwd_fname)

# Use only gradiometers
fwd = mne.pick_types_forward(fwd, meg="grad", eeg=False)

# Create an info object that holds information about the sensors (their
# location, etc.). Make sure to include the external sensor!
info = mne.create_info(
    fwd["info"]["ch_names"] + ["external"],
    sfreq,
    ch_types=["grad"] * fwd["info"]["nchan"] + ["misc"],
)
# Copy grad positions from the forward solution
for info_ch, fwd_ch in zip(info["chs"], fwd["info"]["chs"]):
    info_ch.update(fwd_ch)

# To simulate the data, we need a version of the forward solution where each
# source has a "fixed" orientation, i.e. pointing orthogonally to the surface
# of the cortex.
fwd_fixed = mne.convert_forward_solution(fwd, force_fixed=True)

# Now we can run our simulated signal through the forward model, obtaining
# simulated sensor data.
sensor_data = mne.apply_forward_raw(fwd_fixed, stc, info).get_data()

# We're going to add some noise to the sensor data
noise = rng.standard_normal(sensor_data.shape)

# Scale the noise to be in the ballpark of MEG data
noise_scaling = np.linalg.norm(sensor_data) / np.linalg.norm(noise)
noise *= noise_scaling

# Mix noise and signal with the given signal-to-noise ratio.
sensor_data = SNR * sensor_data + noise

# Add the simulated external sensor data
noise = np.vstack((noise, np.zeros((1, len(times)))))
sensor_data = np.vstack((sensor_data, np.atleast_2d(signal2)))

###############################################################################
# We create an :class:`mne.EpochsArray` object containing two trials: one with
# just noise and one with both noise and signal. The techniques we'll be
# using in this tutorial depend on being able to contrast data that contains
# the signal of interest versus data that does not.
#
epochs = mne.EpochsArray(
    data=np.concatenate(
        (noise[np.newaxis, :, :], sensor_data[np.newaxis, :, :]), axis=0
    ),
    info=info,
    events=np.array([[0, 0, 1], [10, 0, 2]]),
    event_id=dict(noise=1, signal=2),
)

###############################################################################
# Let's take a look at the stimulated data. Scroll all the way down to find the
# simulated external sensor.
epochs.plot(["grad", "misc"])  # By default, misc channels are not plotted

###############################################################################
# Power mapping
# -------------
# With our simulated dataset ready, we can now pretend to be researchers that
# have just recorded this from a real subject and are going to study what parts
# of the brain are coherent with the external sensor.

# Estimate the cross-spectral density (CSD) matrix on the trial containing the
# signal.
csd_signal = csd_morlet(epochs["signal"], frequencies=[10])

# Compute the DICS power map. For this simulated dataset, we need to use
# ``inversion="single"`` for the beamformer to behave properly. For real recordings,
# setting ``inversion="matrix"`` should be possible as well.
dics = make_dics(epochs.info, fwd, csd_signal, inversion="single")
power, f = apply_dics_csd(csd_signal, dics)

# Plot the DICS power map.
brain = power.plot("sample", subjects_dir=subjects_dir, hemi="both", figure=2, size=400)

# Indicate the true location of the source activity on the plot.
brain.add_foci(source_vert, coords_as_verts=True, hemi="lh")

# Rotate the view and add a title.
brain.show_view("frontal")
brain.add_text(0.5, 0.9, f"DICS power map at {f[0]:.1f} Hz", justification="center")

###############################################################################
# You can clearly see where our simulated source is in the brain. Now, let's
# move on to computing coherence between our brain source and the external
# sensor.

###############################################################################
# Sensor-level coherence with external source
# -------------------------------------------
# Let's do something simple first: estimate coherence between the gradiometers
# and the external sensor. The equation for coherence is:
#
#                                      (MEG-EXTERNAL CSD)^2
#          MEG-EXTERNAL coherence = --------------------------
#                                   MEG POWER * EXTERNAL POWER
#
# We can use the :func:`csd_morlet` function compute the full sensor-to-sensor
# CSD, which contains what we need to compute the numerator of the equation.
# The diagonal of this CSD matrix contains the power for each sensor, which we
# can use to compute the denominator of the equation.

csd_signal = csd_morlet(epochs["signal"], frequencies=[10], picks=["grad", "misc"])
csd_signal.plot(mode="coh")
csd_data = csd_signal.get_data(10)
diag_data = np.diag(csd_data)

# plot coherence
psd = np.diag(csd_data).real
coh = np.abs(csd_data) ** 2 / psd[np.newaxis, :] / psd[:, np.newaxis]
plt.imshow(coh)

# plot topomap of coherence
info_grads = mne.pick_info(info, mne.pick_types(info, meg="grad"))
mne.viz.plot_topomap(coh[:-1, -1], info_grads)

###############################################################################
# Source-level coherence with external source
# -------------------------------------------
# To estimate source-level coherence with the external sensor, we can use the
# DICS beamformer to::
#  1. Project the CSD between each gradiometer and the external source to the
#     cortical surface to compute the numerator of the equation.
#  2. Project the gradiometer CSD to the cortical surface to compute the
#     denominator of the equation.
#
dics = make_dics(epochs.info, fwd, csd_signal, reg=1, depth=1, pick_ori=None,
                 inversion='single')
stc_coh = dics_coherence_external(csd_signal, dics, info, fwd,
                                  external="external", pick_ori='max-coherence')
brain = stc_coh.plot("sample", subjects_dir=subjects_dir, hemi="both")

# Indicate the true location of the source activity on the plot.
brain.add_foci(source_vert, coords_as_verts=True, hemi="lh")

# Rotate the view and add a title.
brain.show_view("frontal")
brain.add_text(0.5, 0.9, "Coherence with external sensor", justification="center")

###############################################################################
# References
# ----------
# .. [1] Gross, J., Kujala, J., Hamalainen, M., Timmermann, L., Schnitzler, A.,
#    & Salmelin, R. (2001). Dynamic imaging of coherent sources: Studying
#    neural interactions in the human brain. Proceedings of the National
#    Academy of Sciences, 98(2), 694–699. https://doi.org/10.1073/pnas.98.2.694
# .. [2] van Vliet, M., Liljeström, M., Aro, S., Salmelin, R. and Kujala, J.
#    (2018). "Functional connectivity analysis using DICS: from raw MEG data to
#    group-level statistics in Python". bioRxiv 245530.
#    https://doi.org/10.1101/245530
