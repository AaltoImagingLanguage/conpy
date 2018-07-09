"""
Do-it script to execute the entire pipeline using the doit tool:
http://pydoit.org

All the filenames are defined in config.py

Author: Marijn van Vliet <w.m.vanvliet@gmail.com>
"""
from config import (fname, subjects, conditions, freq_bands, bandpass_fmin,
                    bandpass_fmax)

DOIT_CONFIG = dict(
    default_tasks=['grand_average_power', 'connectivity_stats', 'figures'],
    verbosity=2,
)


def task_check():
    """Check the system dependencies and run conpy test suite."""
    return dict(
        actions=['python check_system.py',
                 'python -m pytest ../conpy/tests/test_*.py'],
    )


def task_fetch_data():
    """Step 00: Download the openfmri ds117 dataset files."""
    for subject in subjects:
        t1_fname = fname.t1(subject=subject)
        sss_fnames = [fname.sss(subject=subject, run=run) for run in range(1, 7)]
        flash5_fname = fname.flash5(subject=subject)

        yield dict(
            name=subject,
            file_dep=['00_fetch_data.py'],
            targets=[t1_fname, flash5_fname] + sss_fnames,
            actions=['python 00_fetch_data.py %s' % subject],
        )


def task_anatomy():
    """Step 01: Run the FreeSurfer anatomical-MRI segmentation program."""
    for subject in subjects:
        t1_fname = fname.t1(subject=subject)
        surf_fnames = [fname.surface(subject=subject, surf=surf)
                       for surf in ['inner_skull', 'outer_skull', 'outer_skin']]
        bem_fname = fname.bem(subject=subject)

        yield dict(
            name=subject,
            task_dep=['fetch_data'],
            file_dep=[t1_fname, '01_anatomy.py'],
            targets=surf_fnames + [bem_fname],
            actions=['python 01_anatomy.py %s' % subject],
        )


def task_filter():
    """Step 02: Bandpass-filter the data"""
    for subject in subjects:
        sss_fnames = [fname.sss(subject=subject, run=run) for run in range(1, 7)]
        filt_fnames = [fname.filt(subject=subject, run=run,
                                  fmin=bandpass_fmin, fmax=bandpass_fmax)
                       for run in range(1, 7)]

        yield dict(
            name=subject,
            task_dep=['anatomy'],
            file_dep=sss_fnames + ['02_filter.py'],
            targets=filt_fnames,
            actions=['python 02_filter.py %s' % subject],
        )


def task_ica():
    """Step 03: Use ICA to clean up ECG and EOG artifacts"""
    for subject in subjects:
        filt_fnames = [fname.filt(subject=subject, run=run,
                                  fmin=bandpass_fmin, fmax=bandpass_fmax)
                       for run in range(1, 7)]
        ica_fname = fname.ica(subject=subject)

        yield dict(
            name=subject,
            task_dep=['filter'],
            file_dep=filt_fnames + ['03_ica.py'],
            targets=[ica_fname],
            actions=['python 03_ica.py %s' % subject],
        )


def task_epochs():
    """Step 04: Cut the data into epochs"""
    for subject in subjects:
        filt_fnames = [fname.filt(subject=subject, run=run,
                                  fmin=bandpass_fmin, fmax=bandpass_fmax)
                       for run in range(1, 7)]
        ica_fname = fname.ica(subject=subject)
        epo_fname = fname.epo(subject=subject)

        yield dict(
            name=subject,
            task_dep=['ica'],
            file_dep=filt_fnames + [ica_fname, '04_epochs.py'],
            targets=[epo_fname],
            actions=['python 04_epochs.py %s' % subject],
        )


def task_csd():
    """Step 05: Compute cross-spectral density (CSD) matrices"""
    for subject in subjects:
        epo_fname = fname.epo(subject=subject)
        csd_fnames = [fname.csd(subject=subject, condition=cond)
                      for cond in conditions + ['baseline']]

        yield dict(
            name=subject,
            task_dep=['epochs'],
            file_dep=[epo_fname, '05_csd.py'],
            targets=csd_fnames,
            actions=['python 05_csd.py %s' % subject],
        )


def task_fsaverage_src():
    """Step 06: Create a source space for the fsaverage brain."""
    return dict(
        file_dep=['06_fsaverage_src.py'],
        task_dep=['csd'],
        targets=[fname.fsaverage_src],
        actions=['python 06_fsaverage_src.py'],
    )


def task_forward():
    """Step 07: Compute forward operators for each subject."""
    for subject in subjects:
        epo_fname = fname.epo(subject=subject)
        fwd_fname = fname.fwd(subject=subject)
        src_fname = fname.src(subject=subject)

        yield dict(
            name=subject,
            task_dep=['fsaverage_src'],
            file_dep=[fname.fsaverage_src, epo_fname, '07_forward.py'],
            targets=[fwd_fname, src_fname],
            actions=['python 07_forward.py %s' % subject],
        )


def task_select_vertices():
    """Step 08: Select the vertices for which to do the analyses."""
    fwd_fnames = [fname.fwd(subject=subject) for subject in subjects]
    fwd_r_fnames = [fname.fwd_r(subject=subject) for subject in subjects]
    src_fnames = [fname.src(subject=subject) for subject in subjects]
    pairs_fname = fname.pairs

    return dict(
        task_dep=['forward'],
        file_dep=['08_select_vertices.py'] + fwd_fnames + src_fnames,
        targets=fwd_r_fnames + [pairs_fname],
        actions=['python 08_select_vertices.py']
    )


def task_power():
    """Step 09: Compute DICS power maps."""
    for subject in subjects:
        fwd_r_fname = fname.fwd_r(subject=subject)

        csd_fnames = []
        stc_fnames = []
        for cond in conditions + ['baseline']:
            csd_fnames.append(fname.csd(subject=subject, condition=cond))
            stc_fnames.append(fname.power_hemi(subject=subject, condition=cond, hemi='lh'))
            stc_fnames.append(fname.power_hemi(subject=subject, condition=cond, hemi='rh'))

        yield dict(
            name=subject,
            task_dep=['select_vertices'],
            file_dep=[fwd_r_fname, '09_power.py'] + csd_fnames,
            targets=stc_fnames,
            actions=['python 09_power.py %s' % subject],
        )


def task_connectivity():
    """Step 10: Compute DICS connectivity."""
    for subject in subjects:
        fwd_r_fname = fname.fwd_r(subject=subject)

        csd_fnames = []
        con_fnames = []
        for cond in conditions:
            csd_fnames.append(fname.csd(subject=subject, condition=cond))
            con_fnames.append(fname.con(subject=subject, condition=cond))

        yield dict(
            name=subject,
            task_dep=['power'],
            file_dep=[fwd_r_fname, fname.pairs, '10_connectivity.py'] + csd_fnames,
            targets=con_fnames,
            actions=['python 10_connectivity.py %s' % subject],
        )


def task_grand_average_power():
    """Step 11: Compute grand average DICS power maps."""
    stc_fnames = []
    for subject in subjects:
        for cond in conditions + ['baseline']:
            stc_fnames.append(fname.power_hemi(subject=subject, condition=cond, hemi='lh'))
            stc_fnames.append(fname.power_hemi(subject=subject, condition=cond, hemi='rh'))
    ga_filenames = [fname.ga_power_hemi(condition=cond, hemi='lh') for cond in conditions + ['contrast']]
    ga_filenames += [fname.ga_power_hemi(condition=cond, hemi='rh') for cond in conditions + ['contrast']]

    return dict(
        task_dep=['connectivity'],
        file_dep=['11_grand_average_power.py'] + stc_fnames,
        targets=ga_filenames,
        actions=['python 11_grand_average_power.py']
    )


def task_connectivity_stats():
    """Step 12: Compute statistics on connectivity."""
    con_fnames = []
    for subject in subjects:
        for cond in conditions:
            con_fnames.append(fname.con(subject=subject, condition=cond))
    ga_con_fnames = [fname.ga_con(condition=cond) for cond in conditions + ['contrast', 'parcelled']]

    return dict(
        task_dep=['grand_average_power'],
        file_dep=['12_connectivity_stats.py'] + con_fnames,
        targets=ga_con_fnames + [fname.stats],
        actions=['python 12_connectivity_stats.py']
    )


def task_figures():
    """Make all figures. Each figure is a sub-task."""
    # Make figure 1: plot of the CSD matrices.
    yield dict(
        name='csd',
        task_dep=['connectivity_stats'],
        file_dep=[fname.epo(subject=subjects[0]),
                  fname.csd(subject=subjects[0], condition='face')],
        targets=['../paper/figures/csd.pdf'],
        actions=['python figure_csd.py'],
    )

    # Make figure 2: plot of the source space and forward model.
    yield dict(
        name='forward',
        file_dep=[fname.fwd(subject=subjects[0]),
                  fname.fwd_r(subject=subjects[0]),
                  fname.trans(subject=subjects[0])],
        targets=['../paper/figures/forward1.png',
                 '../paper/figures/forward2.png'],
        actions=['python figure_forward.py'],
    )

    # Make figure 3: grand average power maps.
    file_dep = [fname.ga_power_hemi(condition=cond, hemi='lh') for cond in conditions]
    file_dep += [fname.ga_power_hemi(condition=cond, hemi='rh') for cond in conditions]
    targets = ['../paper/figures/power_face_lh.png',
               '../paper/figures/power_face_rh.png',
               '../paper/figures/power_scrambled_lh.png',
               '../paper/figures/power_scrambled_rh.png']
    targets += ['../paper/figures/power_contrast_%s-%s-lh.png' % (freq[0], freq[1]) for freq in freq_bands]

    yield dict(
        name='power',
        file_dep=file_dep,
        targets=targets,
        actions=['python figure_power.py'],
    )

    # Make figure 4: plot of the functional connectivity.
    yield dict(
        name='connectivity',
        file_dep=[fname.ga_con(condition='pruned'),
                  fname.ga_con(condition='parcelled')],
        targets=['../paper/figures/degree_lh.png',
                 '../paper/figures/degree_rh.png',
                 '../paper/figures/squircle.pdf'],
        actions=['python figure_connectivity.py'],
    )
