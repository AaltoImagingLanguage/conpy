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
    raise ValueError('matplotlib is not installed. ' + stack_msg)

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
    raise ValueError('your mne version is too old. Version %s is current installed, while version >= 0.16 is required. Please run `pip install --update mne` to install the lastest version.' % mne.__version__)

try:
    from mayavi import mlab
except:
    raise ValueError('mayavi is not installed. Please run `pip install mayavi` to install it.')

try:
    import surfer
except:
    raise ValueError('pysurfer is not installed. Please run `pip install pysurfer` to install it.')

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

print("\nAll seems to be in order.\nYou can now run the entire pipeline with: python -m doit")
