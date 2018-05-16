""" Some utility functions.

Author: Marijn van Vliet <w.m.vanvliet@gmail.com>
"""
import numpy as np
from mne.source_space import (_ensure_src, _get_morph_src_reordering,
                              _ensure_src_subject, SourceSpaces)
from mne.utils import estimate_rank, warn, get_subjects_dir


def _make_diagonal_noise_matrix(csd, reg):
    """Make a diagonal matrix suitable for using as a noise CSD.

    Starts with an identity matrix and scales it to the smallest singular value
    of the CSD matrix that is comfortably larger than 0.

    Parameters
    ----------
    csd : ndarray, shape (n_series, n_series)
        The data cross-spectral density (CSD) matrix.
    reg : float
        The regulariation parameter used when inverting the CSD matrix.

    Returns
    -------
    noise : ndarray, shape (n_series, n_series)
        A suitable noise cross-spectral density (CSD) matrix.
    """
    rank, s = estimate_rank(csd, tol='auto', norm=False,
                            return_singular=True)
    last_singular_value = abs(s[rank - 1])
    ratio = max(abs(reg), last_singular_value)
    return np.eye(len(csd)) * ratio


def _find_indices_1d(haystack, needles, check_needles=True):
    """Find the indices of multiple values in an 1D array.

    Parameters
    ----------
    haystack : ndarray, shape (n,)
        The array in which to find the indices of the needles.
    needles : ndarray, shape (m,)
        The values to find the index of in the haystack.
    check_needles : bool
        Whether to check that all needles are present in the haystack and raise
        an IndexError if this is not the case. Defaults to True. If all needles
        are guaranteed to be present in the haystack, you can disable this
        check for avoid unnecessary computation.

    Returns
    -------
    needle_inds : ndarray, shape (m,)
        The indices of the needles in the haystack.

    Raises
    ------
    IndexError
        If one or more needles could not be found in the haystack.
    """
    haystack = np.asarray(haystack)
    needles = np.asarray(needles)
    if haystack.ndim != 1 or needles.ndim != 1:
        raise ValueError('Both the haystack and the needles arrays should be '
                         '1D.')

    if check_needles and len(np.setdiff1d(needles, haystack)) > 0:
        raise IndexError('One or more values where not present in the given '
                         'haystack array.')

    sorted_ind = np.argsort(haystack)
    return sorted_ind[np.searchsorted(haystack[sorted_ind], needles)]


def get_morph_src_mapping(src_from, src_to, subject_from=None,
                           subject_to=None, subjects_dir=None, indices=False):
    """Get a mapping between an original source space and its morphed version.

    It is assumed that the number of vertices and their positions match between
    the source spaces, only the ordering is different. This is commonly the
    case when using :func:`morph_source_spaces`.

    Parameters
    ----------
    src_from : instance of SourceSpaces
        The original source space that was morphed to the target subject.
    src_to : instance of SourceSpaces | list of two arrays
        Either the source space to which ``src_from`` was morphed, or the
        vertex numbers of this source space.
    subject_from : str | None
        The name of the Freesurfer subject to which ``src_from`` belongs. By
        default, the value stored in the SourceSpaces object is used.
    subject_to : str | None
        The name of the Freesurfer subject to which ``src_to`` belongs. By
        default, the value stored in the SourceSpaces object is used.
    subjects_dir : str | None
        Path to SUBJECTS_DIR if it is not set in the environment.
    indices : bool
        Whether to return mapping between vertex numbers (``False``, the
        default) or vertex indices (``True``).

    Returns
    -------
    from_to : dict | pair of dicts
        If ``indices=True``, a dictionary mapping vertex indices from
        src_from -> src_to. If ``indices=False``, for each hemisphere, a
        dictionary mapping vertex numbers from src_from -> src_to.
    to_from : dict | pair of dicts
        If ``indices=True``, a dictionary mapping vertex indices from
        src_to -> src_from. If ``indices=False``, for each hemisphere, a
        dictionary mapping vertex numbers from src_to -> src_from.

    See also
    --------
    _get_morph_src_reordering
    """
    if subject_from is None:
        subject_from = src_from[0]['subject_his_id']
    if subject_to is None:
        subject_to = src_to[0]['subject_his_id']
    subjects_dir = get_subjects_dir(subjects_dir, raise_error=True)

    src_from = _ensure_src(src_from, kind='surf')
    subject_from = _ensure_src_subject(src_from, subject_from)

    if isinstance(src_to, SourceSpaces):
        to_vert_lh = src_to[0]['vertno']
        to_vert_rh = src_to[1]['vertno']
    else:
        if subject_to is None:
            ValueError('When supplying vertex numbers as `src_to`, the '
                       '`subject_to` parameter must be set.')
        to_vert_lh, to_vert_rh = src_to

    order, from_vert = _get_morph_src_reordering(
        [to_vert_lh, to_vert_rh], src_from, subject_from, subject_to,
        subjects_dir=subjects_dir
    )
    from_vert_lh, from_vert_rh = from_vert

    if indices:
        # Find vertex indices corresponding to the vertex numbers for src_from
        from_n_lh = src_from[0]['nuse']
        from_ind_lh = _find_indices_1d(src_from[0]['vertno'], from_vert_lh)
        from_ind_rh = _find_indices_1d(src_from[1]['vertno'], from_vert_rh)
        from_ind = np.hstack((from_ind_lh, from_ind_rh + from_n_lh))

        # The indices for src_to are easy
        to_ind = order

        # Create the mappings
        from_to = dict(zip(from_ind, to_ind))
        to_from = dict(zip(to_ind, from_ind))
    else:
        # Re-order the vertices of src_to to match the ordering of src_from
        to_n_lh = len(to_vert_lh)
        to_vert_lh = to_vert_lh[order[:to_n_lh]]
        to_vert_rh = to_vert_rh[order[to_n_lh:] - to_n_lh]

        # Create the mappings
        from_to = [dict(zip(from_vert_lh, to_vert_lh)),
                   dict(zip(from_vert_rh, to_vert_rh))]
        to_from = [dict(zip(to_vert_lh, from_vert_lh)),
                   dict(zip(to_vert_rh, from_vert_rh))]

    return from_to, to_from
