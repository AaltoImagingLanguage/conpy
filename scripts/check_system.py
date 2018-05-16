import os
import warnings

# Scientific stack message
stack_msg = ('Make sure the basic Python scientific stack (Numpy/Scipy/Matplotlib) is installed.'
             'We recommend using the Anaconda Python distribution for this: http://docs.continuum.io/anaconda/install')

# Check if dependencies are present
try:
    import numpy
except:
    raise ValueError('numpy is not installed. ' + stack_msg)

try:
    import scipy
except:
    raise ValueError('scipy is not installed. ' + stack_msg)

figures = True
try:
    from matplotlib import pyplot
except:
    warnings.warn('matplotlib is not installed. Making the figures will not work. ' + stack_msg)
    figures = False

try:
    import tqdm
except:
    raise ValueError('tqdm is not installed. Please run `conda install tqdm` to install it.')

try:
    import mne
except:
    raise ValueError('mne is not installed. Please run `pip install mne` to install it.')

try:
    from distutils.version import LooseVersion
    assert LooseVersion(mne.__version__) >= LooseVersion('0.16')
except:
    raise ValueError('your mne version is too old. Version %s is current installed, while version >= 0.16 is required. Please run `pip install --update https://github.com/wmvanvliet/mayavi/zipball/master` to install the lastest development version.' % mne.__version__)

try:
    from mayavi import mlab
except:
    warnings.warn('mayavi is not installed. Making the figures will not work. '
                  'See the MNE-Python installation instructions on how to best get mayavi running: '
                  'https://martinos.org/mne/stable/install_mne_python.html#install-dependencies-and-mne')
    figures = False

try:
    import surfer
except:
    warning.warn('pysurfer is not installed. Making the figures will not work.')
    figures = False

try:
    import doit
except:
    raise ValueError('doit is not installed. Please run `pip install doit` to install it.')

try:
    import numba
except:
    warning.warn('numba is not installed. You can speed up the connectivity analysis by install it with: `conda install numba`.')

try:
    import conpy
except:
    raise ValueError('conpy is not installed. Please run `python setup.py install` to install it.')

mne.sys_info()

OMP_NUM_THREADS = os.environ.get("OMP_NUM_THREADS", None)
if OMP_NUM_THREADS is None:
    warnings.warn('OMP_NUM_THREADS is not set. We recommend you set it to '
                  '2 or 4 depending on your system.')
else:
    print("OMP_NUM_THREADS: %s" % OMP_NUM_THREADS)

# Check that the example dataset is installed
from config import fname, subjects
if not os.path.exists(fname.study_path):
    raise ValueError('The `study_path` points to a directory that does not exist: ' + fname.study_path())

if figures:
    print("\nAll seems to be in order.\nYou can now run the entire pipeline with: python -m doit")
else:
    print("\nYou can run the pipeline without producing the figures using: python -m doit grand_average_power connectivity_stats")
