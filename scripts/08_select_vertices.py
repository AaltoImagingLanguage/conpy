"""
Figure out which vertices to use in the power mapping and connectivity
analysis.

Vertices are selected from the original source space. Since in the forward
operator, some vertices are dropped due to being too close to the skull, the
selected indices wouldn't match up if we repeat this selection on the forward
operator. Therefore we will use the vertex numbers to align the selections.
"""
from __future__ import print_function

import numpy as np
import mne
import conpy

from config import fname, subjects, max_sensor_dist, min_pair_dist

# Be verbose
mne.set_log_level('INFO')

print('Restricting source spaces...')
# Restrict the forward operator of the first subject to vertices that are close
# to the sensors.
fwd1 = mne.read_forward_solution(fname.fwd(subject=subjects[0]))
fwd1 = conpy.restrict_forward_to_sensor_range(fwd1, max_sensor_dist)

# Load the rest of the forward operators
fwds = [fwd1]
for subject in subjects[1:]:
    fwds.append(mne.read_forward_solution(fname.fwd(subject=subject)))

# Compute the vertices that are shared across the forward operators for all
# subjects. The first one we restricted ourselves, the other ones may have
# dropped vertices which were too close to the inner skull surface. We use the
# fsaverage brain as a reference to determine corresponding vertices across
# subjects.
fsaverage = mne.read_source_spaces(fname.fsaverage_src)
vert_inds = conpy.select_shared_vertices(fwds, ref_src=fsaverage,
                                         subjects_dir=fname.subjects_dir)

# Restrict all forward operators to the same vertices and save them.
for fwd, vert_ind, subject in zip(fwds, vert_inds, subjects):
    fwd_r = conpy.restrict_forward_to_vertices(fwd, vert_ind)
    mne.write_forward_solution(fname.fwd_r(subject=subject), fwd_r,
                               overwrite=True)

    # Update the forward operator of the first subject
    if subject == subjects[0]:
        fwd1 = fwd_r

# Compute vertex pairs for which to compute connectivity
# (distances are based on the MRI of the first subject).
print('Computing connectivity pairs for all subjects...')
pairs = conpy.all_to_all_connectivity_pairs(fwd1, min_dist=min_pair_dist)

# Store the pairs in fsaverage space
subj1_to_fsaverage = conpy.utils.get_morph_src_mapping(
    fsaverage, fwd1['src'], indices=True, subjects_dir=fname.subjects_dir
)[1]
pairs = [[subj1_to_fsaverage[v] for v in pairs[0]],
         [subj1_to_fsaverage[v] for v in pairs[1]]]
np.save(fname.pairs, pairs)
