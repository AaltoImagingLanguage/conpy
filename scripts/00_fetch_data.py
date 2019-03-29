"""
Download the openfmri ds117 dataset and extract it into the designated
"study_path" folder.
"""
import argparse
import tarfile
from urllib.request import urlretrieve
from os import makedirs
from tqdm import tqdm

from config import fname

# Handle command line arguments
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('subject', metavar='sub###',
                    help='The subject to download the data for')
args = parser.parse_args()
subject = args.subject
print('Downloading data for subject:', subject)


def download(url, filename):
    """Downloads a file with pretty tqdm progress bar.

    Parameters
    ----------
    url : str
        The URL where we can find the file.
    filename : str
        The full path and filename where to store the file on the local system.
    """
    print('%s -> %s' % (url, filename))

    with tqdm(unit='b', unit_scale=True, unit_divisor=1024, miniters=1) as t:
        prev_n_blocks = [0]

        def progress(n_blocks, block_size, total_n_blocks):
            t.total = total_n_blocks
            t.update((n_blocks - prev_n_blocks[0]) * block_size)
            prev_n_blocks[0] = n_blocks

        urlretrieve(url, filename=filename, reporthook=progress)


# Make the "archive" directory for storing the downloaded files
makedirs(fname.archive_dir, exist_ok=True)

# Download the subject file
download(fname.subject_url(subject=subject),
         fname.subject_tarball(subject=subject))

# Extract file
makedirs(fname.ds117_dir(subject=subject), exist_ok=True)
print('Unzipping %s -> %s' % (fname.subject_tarball(subject=subject),
                              fname.ds117_dir(subject=subject)))
with tarfile.open(fname.subject_tarball(subject=subject), 'r:gz') as tar:
    if subject == 'sub019':
        # This subject's tar.gz file does not give an ds117/sub019 subfolder structure
        tar.extractall(fname.ds117_dir(subject=subject))
    else:
        tar.extractall(fname.study_path)

makedirs(fname.subject_dir(subject=subject), exist_ok=True)
makedirs(fname.reports_dir, exist_ok=True)
