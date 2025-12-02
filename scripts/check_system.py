"""
This script performs a series of checks on the system to see if everything is
ready to run the analysis pipeline.
"""

import importlib
import os
import re
import warnings

# Check to see if the python dependencies are fullfilled.
dependencies = []
with open("../requirements.txt") as f:
    for line in f:
        line = line.strip()
        if matches := re.match(r"^[a-z0-9\-_]+", line):
            dependencies.append(matches.group(0))

# This raises errors of dependencies are not met
missing_deps = list()
print("Dependecies")
for dep in dependencies:
    if importlib.util.find_spec(dep.replace("-", "_")) is None:
        print("├☒ ", dep)
        missing_deps.append(dep)
    else:
        print("├☑ ", dep)
if len(missing_deps) > 0:
    raise ValueError(
        f"Not all packages in requirements.txt are installed. Missing: {missing_deps}."
    )

try:
    import mne
    from packaging.version import Version

    assert Version(mne.__version__) >= Version("1.0")
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
