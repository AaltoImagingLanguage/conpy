"""
Run FreeSurfer to process the anatomical MRI data.

Adapted from the MNE-Python preprocessing pipeline:
https://github.com/mne-tools/mne-biomag-group-demo/blob/master/scripts/processing/01-anatomy.py
"""  # noqa E402

import argparse
import os
import os.path as op
import subprocess
import glob
import shutil
import mne
import nibabel

from config import fname, subjects, n_jobs

# Handle command line arguments
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("subject", metavar="sub###", help="The subject to process")
args = parser.parse_args()
subject = args.subject
print("Processing subject:", subject, "(usually takes hours)")


def tee_output(command, log_file):
    """Write the output of a command to a logfile as well as stdout."""
    print("Writing the output of the command below to", log_file)
    print(" ".join(command))
    with open(log_file, "wb") as fid:
        proc = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
        )
        for line in proc.stdout:
            fid.write(line)
    if proc.wait() != 0:
        raise RuntimeError("Command failed")


tee_output(
    [
        "recon-all",
        "-all",
        "-s",
        subject,
        "-sd",
        fname.subjects_dir,
        "-i",
        fname.t1(subject=subject),
    ],
    fname.freesurfer_log(subject=subject),
)

print("Copying FLASH files")
os.makedirs(fname.flash_dir(subject=subject), exist_ok=True)
for f_src in glob.glob(fname.flash_glob(subject=subject)):
    f_dst = op.basename(f_src).replace("meflash_", "mef")
    f_dst = op.join(fname.flash_dir(subject=subject), f_dst)
    shutil.copy(f_src, f_dst)

# Fix the headers for subject 19
if subject == "sub019":
    print("Fixing FLASH files for %s" % (subject,))
    flash_files = ["mef05_%d.mgz" % x for x in range(7)] + [
        "mef30_%d.mgz" % x for x in range(7)
    ]

    for flash_file in flash_files:
        dest_fname = op.join(fname.flash_dir(subject=subject), flash_file)
        dest_img = nibabel.load(op.splitext(dest_fname)[0] + ".nii.gz")

        # Copy the headers from the first subject
        src_img = nibabel.load(
            op.join(fname.flash_dir(subject=subjects[0]), flash_file)
        )
        hdr = src_img.header
        fixed = nibabel.MGHImage(dest_img.get_data(), dest_img.affine, hdr)
        nibabel.save(fixed, dest_fname)

print("Converting flash MRIs")
mne.bem.convert_flash_mris(subject, convert=False, subjects_dir=fname.subjects_dir)

print("Making BEM surfaces")
mne.bem.make_flash_bem(subject, subjects_dir=fname.subjects_dir, show=False)

# Save BEM figure to report
with mne.open_report(fname.report(subject=subject)) as report:
    report.add_bem_to_section(
        subject, "BEM surfaces", section="Anatomy", n_jobs=n_jobs, replace=True
    )
    report.save(fname.report_html(subject=subject), overwrite=True, open_browser=False)
