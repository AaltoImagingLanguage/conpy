import numpy as np, mne  # Import required Python modules

epochs = mne.read_epochs("sub002-epo.fif")  # Read epochs from FIFF file
epochs = epochs["face"]  # Select the experimental condition
frequencies = np.linspace(7, 17, num=11)  # Specify frequencies to use
csd = mne.time_frequency.csd_morlet(
    epochs, frequencies, tmin=0, tmax=0.4, n_cycles=7, decim=20
)
csd_alpha = csd.mean(7, 13)  # CSD for alpha band: 7-13 Hz
csd_beta = csd.mean(13, 17)  # CSD for beta band: 13-17 Hz
