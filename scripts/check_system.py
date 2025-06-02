"""
This script performs a series of checks on the system to see if everything is
ready to run the analysis pipeline.
"""

import os
import warnings
import pkg_resources

# Check to see if the python dependencies are fullfilled.
dependencies = []
with open("../requirements.txt") as f:
    for line in f:
        line = line.strip()
        if len(line) == 0 or line.startswith("#"):
            continue
        dependencies.append(line)

# This raises errors of dependencies are not met
pkg_resources.working_set.require(dependencies)

try:
    import mne
    from distutils.version import LooseVersion

    assert LooseVersion(mne.__version__) >= LooseVersion("0.16")
except:
    raise ValueError(
        "your mne version is too old. Version %s is current installed, while version >= 0.16 is required. Please run `pip install --update mne` to install the lastest version."
        % mne.__version__
    )

try:
    import numba
except:
    warnings.warn(
        "numba is not installed. You can speed up the connectivity analysis by install it with: `conda install numba`."
    )

try:
    import conpy
except:
    raise ValueError(
        "conpy is not installed. Please run `python setup.py install` to install it."
    )

mne.sys_info()

OMP_NUM_THREADS = os.getenv("OMP_NUM_THREADS")
if OMP_NUM_THREADS is None:
    warnings.warn(
        "OMP_NUM_THREADS is not set. We recommend you set it to "
        "2 or 4 depending on your system."
    )
else:
    print("OMP_NUM_THREADS: %s" % OMP_NUM_THREADS)

# Check that the example dataset is installed
from config import fname, subjects

if not os.path.exists(fname.study_path):
    raise ValueError(
        "The `study_path` points to a directory that does not exist: "
        + fname.study_path()
    )

print(
    "\nAll seems to be in order.\nYou can now run the entire pipeline with: python -m doit"
)
