"""
Create a source space by for the fsaverage brain.

Author: Marijn van Vliet <w.m.vanvliet@gmail.com>
"""
from __future__ import print_function
import os
import os.path as op
import shutil
import mne

from config import fname, spacing, n_jobs

# Be verbose
mne.set_log_level('INFO')

print('Copying fsaverage into subjects directory')
fsaverage_src_dir = op.join(os.environ['FREESURFER_HOME'], 'subjects',
                            'fsaverage')
fsaverage_dst_dir = op.join(fname.anatomy(subject='fsaverage'))

if not op.isdir(fname.anatomy(subject='fsaverage')):
    # Remove symlink if present
    os.unlink(fname.anatomy(subject='fsaverage'))

if not op.exists(fname.anatomy(subject='fsaverage')):
    shutil.copytree(fsaverage_src_dir, fname.anatomy(subject='fsaverage'))

if not op.isdir(fname.bem_dir(subject='fsaverage')):
    os.mkdir(fname.bem_dir(subject='fsaverage'))

# Create source space on the fsaverage brain
fsaverage = mne.setup_source_space('fsaverage', spacing=spacing,
                                   subjects_dir=fname.subjects_dir,
                                   n_jobs=n_jobs, add_dist=False)
mne.write_source_spaces(fname.fsaverage_src, fsaverage, overwrite=True)
