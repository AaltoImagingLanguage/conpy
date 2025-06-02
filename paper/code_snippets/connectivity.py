import conpy, mne  # Import required Python modules

# Read and convert a forward model to one that defines two orthogonal dipoles
# at each source, that are tangential to a sphere.
fwd = mne.read_forward_solution("sub002-fwd.fif")  # Read forward model
fwd_tan = conpy.forward_to_tangential(fwd_r)  # Convert to tangential model

# Pairs for which to compute connectivity. Use a distance threshold of 4 cm.
pairs = conpy.all_to_all_connectivity_pairs(fwd_tan, min_dist=0.04)

# Load CSD matrix
csd = conpy.read_csd("sub002-csd-face.h5")  # Read CSD for 'face' condition
csd = csd.mean(fmin=31, fmax=40)  # Obtain CSD for frequency band 31-40 Hz.

# Compute source connectivity using DICS. Try 50 orientations for each source
# point to find the orientation that maximizes coherence.
con = conpy.dics_connectivity(pairs, fwd_tan, csd, reg=0.05, n_angles=50)
