import mne  # Import required Python modules

info = mne.io.read_info("sub002-epo.h5")  # Read info structure
fwd = mne.read_forward_solution("sub002-fwd.fif")  # Read forward model
csd = mne.time_frequency.read_csd("sub002-csd-face.h5")  # Read CSD
csd = csd.mean(fmin=7, fmax=13)  # Obtain CSD for frequency band 7-13 Hz.
# Compute DICS beamformer filters
filters = mne.beamformer.make_dics(info, fwd, csd, reg=0.05, pick_ori="max-power")
# Compute the power map
stc = mne.beamformer.apply_dics_csd(csd, filters)
