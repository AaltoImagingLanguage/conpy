"""
Perform a cluster permutation test to only retain bundles of connections that
show a significant difference between the experimental conditions.
"""

from __future__ import print_function
import argparse

import mne
import conpy
from mne.externals.h5io import write_hdf5

from config import fname, conditions, subjects

# Be verbose
mne.set_log_level("INFO")

# Handle command line arguments (only --help)
parser = argparse.ArgumentParser(description=__doc__)
args = parser.parse_args()

# Connectivity will be morphed back to the fsaverage brain
fsaverage = mne.read_source_spaces(fname.fsaverage_src)

cons = dict()
for condition in conditions:
    print("Reading connectivity for condition:", condition)
    cons[condition] = list()

    for subject in subjects:
        con_subject = conpy.read_connectivity(
            fname.con(condition=condition, subject=subject)
        )

        # Morph the Connectivity to the fsaverage brain. This is possible,
        # since the original source space was fsaverage morphed to the current
        # subject.
        con_fsaverage = con_subject.to_original_src(
            fsaverage, subjects_dir=fname.subjects_dir
        )
        # By now, the connection objects should define the same connection
        # pairs between the same vertices.

        cons[condition].append(con_fsaverage)

# Average the connection objects. To save memory, we add the data in-place.
print("Averaging connectivity objects...")
ga_con = dict()
for cond in conditions:
    con = cons[cond][0].copy()
    for other_con in cons[cond][1:]:
        con += other_con
    con /= len(cons[cond])  # compute the mean
    ga_con[cond] = con
    con.save(fname.ga_con(condition=cond))

# Compute contrast between faces and scrambled pictures
contrast = ga_con[conditions[0]] - ga_con[conditions[1]]
contrast.save(fname.ga_con(condition="contrast"))

# Perform a permutation test to only retain connections that are part of a
# significant bundle.
stats = conpy.cluster_permutation_test(
    cons["face"],
    cons["scrambled"],
    cluster_threshold=5,
    src=fsaverage,
    n_permutations=1000,
    verbose=True,
    alpha=0.05,
    n_jobs=2,
    seed=10,
    return_details=True,
    max_spread=0.01,
)
connection_indices, bundles, bundle_ts, bundle_ps, H0 = stats
con_clust = contrast[connection_indices]

# Save some details about the permutation stats to disk
write_hdf5(
    fname.stats,
    dict(
        connection_indices=connection_indices,
        bundles=bundles,
        bundle_ts=bundle_ts,
        bundle_ps=bundle_ps,
        H0=H0,
    ),
    overwrite=True,
)

# Save the pruned grand average connection object
con_clust.save(fname.ga_con(condition="pruned"))

# Summarize the connectivity in parcels
labels = mne.read_labels_from_annot("fsaverage", "aparc")
del labels[-1]  # drop 'unknown-lh' label
con_parc = con_clust.parcellate(labels, summary="degree", weight_by_degree=False)
con_parc.save(fname.ga_con(condition="parcelled"))

print("[done]")
