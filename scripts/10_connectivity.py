"""
Connectivity analysis using DICS.
"""
from __future__ import print_function
import argparse

import numpy as np
import mne
from mne.time_frequency import read_csd, pick_channels_csd
import conpy
from matplotlib import pyplot as plt

from config import fname, con_fmin, con_fmax, conditions, reg, n_jobs

# Be verbose
mne.set_log_level('INFO')

# Handle command line arguments
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('subject', metavar='sub###', help='The subject to process')
args = parser.parse_args()
subject = args.subject
print('Processing subject:', subject)

# Read the forward model
fwd_r = mne.read_forward_solution(fname.fwd_r(subject=subject))

# Convert the forward model to one that defines two orthogonal dipoles at each
# source, that are tangential to a sphere.
fwd_tan = conpy.forward_to_tangential(fwd_r)

# Pairs for which to compute connectivity
pairs = np.load(fname.pairs)

# Pairs are defined in fsaverage space, map them to the source space of the
# current subject.
fsaverage = mne.read_source_spaces(fname.fsaverage_src)
fsaverage_to_subj = conpy.utils.get_morph_src_mapping(
    fsaverage, fwd_tan['src'], indices=True, subjects_dir=fname.subjects_dir
)[0]
pairs = [[fsaverage_to_subj[v] for v in pairs[0]],
         [fsaverage_to_subj[v] for v in pairs[1]]]

# Compute connectivity for one frequency band across all conditions
cons = dict()
for condition in conditions:
    print('Computing connectivity for condition:', condition)
    # Read the CSD matrix
    csd = read_csd(fname.csd(condition=condition, subject=subject))

    # Use the beta band for connectivity analysis
    csd = csd.mean(con_fmin, con_fmax)

    # Pick channels actually present in the forward model (only MEG)
    csd = pick_channels_csd(csd, fwd_tan['info']['ch_names'])

    # Compute connectivity for all frequency bands
    con = conpy.dics_connectivity(
        vertex_pairs=pairs,
        fwd=fwd_tan,
        data_csd=csd,
        reg=reg,
        n_jobs=n_jobs,
    )
    cons[condition] = con

    con.save(fname.con(condition=condition, subject=subject))

# Save a plot of the adjacency matrix to the HTML report
with mne.open_report(fname.report(subject=subject)) as report:
    adj = (cons[conditions[0]] - cons[conditions[1]]).get_adjacency()
    fig = plt.figure()
    plt.imshow(adj.toarray(), interpolation='nearest')
    report.add_figs_to_section(fig, ['Adjacency matrix'],
                               section='Source-level', replace=True)
    report.save(fname.report_html(subject=subject), overwrite=True,
                open_browser=False)
