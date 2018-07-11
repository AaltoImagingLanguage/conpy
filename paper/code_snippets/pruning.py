import conpy, mne  # Import required Python modules
import operator  # For the 'add' operator
from functools import reduce

# Connectivity objects are morphed back to the fsaverage brain
fsaverage = mne.read_source_spaces('fsaverage-src.fif')

# For each of the subjects, read connectivity for different conditions.
# Re-order the vertices to be in the order of the fsaverage brain.
face = []
scrambled = []
contrast = []
subjects = ['sub002', 'sub003', 'sub004', 'sub006', 'sub007', 'sub008',
            'sub009', 'sub010', 'sub011', 'sub012', 'sub013', 'sub014',
            'sub015', 'sub017', 'sub018', 'sub019']
for subject in subjects:
    con_face = conpy.read_connectivity('%s-face-con.h5' % subject)
    con_face = con_face.to_original_src(fsaverage)
    con_scrambled = conpy.read_connectivity('%s-scrambled-con.h5' % subject)
    con_scrambled = con_scrambled.to_original_src(fsaverage)
    face.append(con_face)
    scrambled.append(con_scrambled)
    contrast.append(con_face - con_scrambled)  # Create contrast

# Compute the grand-average contrast
contrast = reduce(operator.add, contrast) / len(subjects)

# Perform a permutation test to only retain connections that are part of a
# significant bundle.
connection_indices = conpy.cluster_permutation_test(
    face, scrambled,      # The two conditions
    cluster_threshold=5,  # The initial t-value threshold to form bundles
    max_spread=0.01,      # Maximum distance (in m) between connections
                          #   that are assigned to the same bundle.
    src=fsaverage,        # The source space for distance computations
    n_permutations=1000,  # The number of permutations for estimating
                          #   the distribution of t-values.
    alpha=0.05            # The p-value at which to reject the null-hypothesis
)

# Prune the contrast connectivity to only contain connections that are part of
# significant bundles.
contrast = contrast[connection_indices]
