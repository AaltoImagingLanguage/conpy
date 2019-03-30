"""
===========
Config file
===========

Configuration parameters for the study.
"""

import os
from socket import getfqdn
from fnames import FileNames
import pickle
from mne import Report


###############################################################################
# Determine which user is running the scripts on which machine and set the path
# where the data is stored and how many CPU cores to use. This is probably the
# only section you need to modify to replicate the pipeline as presented in van
# Vliet et al. 2018.

user = os.environ['USER']  # Username of the user running the scripts
host = getfqdn()  # Hostname of the machine running the scripts

if user == 'rodin':
    # My laptop
    study_path = '/Volumes/scratch/nbe/conpy'
    n_jobs = 4
elif host == 'nbe-024.org.aalto.fi' and user == 'vanvlm1':
    # My workstation
    study_path = '/m/nbe/scratch/conpy'
    n_jobs = 8
elif 'triton' in host and user == 'vanvlm1':
    # The big computational cluster at Aalto University
    study_path = '/m/nbe/scratch/conpy'
    n_jobs = 1
elif ((host == 'iris.org.aalto.fi' or host == 'thor.org.aalto.fi')
      and user == 'vanvlm1'):
    # Some servers at Aalto University with a few more cores and more memory
    # than my workstation
    study_path = '/m/nbe/scratch/conpy'
    n_jobs = 16
else:
    raise RuntimeError('Please edit scripts/config.py and set the study_path '
                       'variable to point to the location where the data '
                       'should be stored and the n_jobs variable to the '
                       'number of CPU cores the analysis is allowed to use.')


###############################################################################
# These are all the relevant parameters for the analysis. You can experiment
# with changing these.

# Band-pass filter limits. Since we are performing ICA on the continuous data,
# it is important that the lower bound is at least 1Hz.
bandpass_fmin = 1  # Hz
bandpass_fmax = 40  # Hz

# Maximum number of ICA components to reject
n_ecg_components = 2  # ICA components that correlate with heart beats
n_eog_components = 2  # ICA components that correlate with eye blinks

# Experimental conditions we're interested in: faces versus scrambled
conditions = ['face', 'scrambled']

# Time window (relative to stimulus onset) to use for extracting epochs
epoch_tmin, epoch_tmax = -0.2, 2.9

# Time window to use for computing the baseline of the epoch
baseline = (-0.2, 0)

# Thresholds to use for rejecting epochs that have a too large signal amplitude
reject = dict(grad=3E-10, mag=4E-12)

# Time window (relative to stimulus onset) to use for computing the CSD
csd_tmin, csd_tmax = 0, 0.4

# Spacing of sources to use
spacing = 'ico4'

# Maximum distance between sources and a sensor (in meters)
max_sensor_dist = 0.07

# Minimum distance between sources and the skull (in mm)
min_skull_dist = 0

# Regularization parameter to use when computing the DICS beamformer
reg = 0.05

# Frequency bands to perform powermapping for
freq_bands = [
    (3, 7),     # theta
    (7, 13),    # alpha
    (13, 17),   # low beta
    (17, 25),   # high beta 1
    (25, 31),   # high beta 2
    (31, 40),   # low gamma
]

# Frequency band to use when computing connectivity (low gamma)
con_fmin = 31
con_fmax = 40

# Minimum distance between sources to compute connectivity for (in meters)
min_pair_dist = 0.04


###############################################################################
# Here are some other configuration variables that are used by the script.
# These are specific to the openfmri ds117 dataset and unless you apply this
# pipeline to a new dataset, you probably have no reason to change these.

# Some mapping between filenames and subjects
map_subjects = {
    'sub001': 'subject_01',
    'sub002': 'subject_02',
    'sub003': 'subject_03',
    'sub004': 'subject_05',
    'sub005': 'subject_06',
    'sub006': 'subject_08',
    'sub007': 'subject_09',
    'sub008': 'subject_10',
    'sub009': 'subject_11',
    'sub010': 'subject_12',
    'sub011': 'subject_14',
    'sub012': 'subject_15',
    'sub013': 'subject_16',
    'sub014': 'subject_17',
    'sub015': 'subject_18',
    'sub016': 'subject_19',
    'sub017': 'subject_23',
    'sub018': 'subject_24',
    'sub019': 'subject_25',
}

# For these, no forward operator was computed in the MNE-Python pipeline
bad_subjects = ['sub001', 'sub005', 'sub016']

# All "good" subjects
subjects = [s for s in sorted(map_subjects.keys()) if s not in bad_subjects]

# The event codes used in the experiment
events_id = {
    'face/famous/first': 5,
    'face/famous/immediate': 6,
    'face/famous/long': 7,
    'face/unfamiliar/first': 13,
    'face/unfamiliar/immediate': 14,
    'face/unfamiliar/long': 15,
    'scrambled/first': 17,
    'scrambled/immediate': 18,
    'scrambled/long': 19,
}


###############################################################################
# Templates for filenames
#
# This part of the config file uses the FileNames class. It provides a small
# wrapper around string.format() to keep track of a list of filenames.
# See fnames.py for details on how this class works.
fname = FileNames()

# Some directories
fname.add('study_path', study_path)
fname.add('archive_dir', '{study_path}/archive')
fname.add('meg_dir', '{study_path}/MEG')
fname.add('subjects_dir', '{study_path}/subjects')
fname.add('subject_dir', '{meg_dir}/{subject}')

# URLs and filenames for the original openfmri ds117 files
fname.add('ds117_url', 'http://openfmri.s3.amazonaws.com/tarballs')
fname.add('metadata_url', '{ds117_url}/ds117_R0.1.1_metadata.tgz')
fname.add('metadata_tarball', '{archive_dir}/ds117_R0.1.1_metadata.tgz')
fname.add('metadata_dir', '{study_path}/metadata')
fname.add('subject_url', '{ds117_url}/ds117_R0.1.1_{subject}_raw.tgz')
fname.add('subject_tarball', '{archive_dir}/ds117_R0.1.1_{subject}_raw.tgz')
fname.add('ds117_dir', '{study_path}/ds117/{subject}')

# Original MEG files provided in the ds117 dataset
fname.add('raw', '{ds117_dir}/MEG/run_{run:02d}_raw.fif')
fname.add('sss', '{ds117_dir}/MEG/run_{run:02d}_sss.fif')
fname.add('trans', '{ds117_dir}/MEG/{subject}-trans.fif')

# Directories related to FreeSurfer files
fname.add('t1', '{ds117_dir}/anatomy/highres001.nii.gz')
fname.add('freesurfer_log', '{ds117_dir}/my-recon-all.txt')
fname.add('flash_glob', '{ds117_dir}/anatomy/FLASH/meflash*')
fname.add('anatomy', '{subjects_dir}/{subject}')
fname.add('flash_dir', '{anatomy}/mri/flash')
fname.add('flash5', '{anatomy}/mri/flash/parameter_maps/flash5.mgz')
fname.add('bem_dir', '{anatomy}/bem')
fname.add('bem', '{bem_dir}/{subject}-5120-bem-sol.fif')
fname.add('surface', '{bem_dir}/flash/{surf}.surf')

# Filenames for all the files that will be generated during the analysis
fname.add('filt', '{subject_dir}/run_{run:02d}-filt-{fmin}-{fmax}-raw_sss.fif')
fname.add('ica', '{subject_dir}/{subject}-ica.fif')
fname.add('epo', '{subject_dir}/{subject}-epo.fif')
fname.add('epo_dirty', '{subject_dir}/{subject}-dirty-epo.fif')
fname.add('reject', '{subject_dir}/{subject}-reject-thresholds.h5')
fname.add('csd', '{subject_dir}/{subject}-{condition}-csd.h5')
fname.add('power', '{subject_dir}/{subject}-{condition}-dics-power')
fname.add('power_hemi',
          '{subject_dir}/{subject}-{condition}-dics-power-{hemi}.stc')
fname.add('con', '{subject_dir}/{subject}-{condition}-connectivity.h5')
fname.add('ga_power', '{meg_dir}/{condition}-average-dics')
fname.add('ga_power_hemi', '{meg_dir}/{condition}-average-dics-{hemi}.stc')
fname.add('ga_con', '{meg_dir}/{condition}-average-connectivity.h5')
fname.add('stats', '{meg_dir}/stats.h5')
fname.add('sp', spacing)  # Add this so we can use it in the filenames below
fname.add('src', '{anatomy}/fsaverage_to_{subject}-{sp}-src.fif')
fname.add('fsaverage_src', '{subjects_dir}/fsaverage/fsaverage-{sp}-src.fif')
fname.add('fwd', '{subject_dir}/fsaverage_to_{subject}-meg-{sp}-fwd.fif')
fname.add('fwd_r', '{subject_dir}/{subject}-restricted-meg-{sp}-fwd.fif')
fname.add('pairs', '{meg_dir}/pairs.npy')

# Filenames for MNE reports
fname.add('reports_dir', '{study_path}/reports/')
fname.add('report', '{reports_dir}/{subject}-report.h5')
fname.add('report_html', '{reports_dir}/{subject}-report.html')

# For FreeSurfer and MNE-Python to find the MRI data
os.environ["SUBJECTS_DIR"] = fname.subjects_dir

# For BLAS to use the right amount of cores
os.environ['OMP_NUM_THREADS'] = str(n_jobs)


def get_report(subject):
    """Get a Report object for a subject.

    If the Report had been saved (pickle'd) before, load it. Otherwise,
    construct a new one.
    """
    report_fname = fname.report(subject=subject)
    if os.path.exists(report_fname):
        with open(report_fname, 'rb') as f:
            return pickle.load(f)
    else:
        return Report(subjects_dir=fname.subjects_dir, subject=subject,
                      title='Analysis for %s' % subject)


def save_report(report):
    """Save a Report object using pickle and render it to HTML."""
    report_fname = fname.report(subject=report.subject)
    with open(report_fname, 'wb') as f:
        pickle.dump(report, f)
    report.save(fname.report_html(subject=report.subject), open_browser=False,
                overwrite=True)

