# -*- coding: utf-8 -*-
"""Extensions for MNE-Python's Forward operator.

Authors: Susanna Aro <susanna.aro@aalto.fi>
         Marijn van Vliet <w.m.vanvliet@gmail.com>
"""

from copy import deepcopy

import numpy as np
from mne import Forward, SourceSpaces, channel_type
from mne._fiff.pick import _picks_to_idx
from mne.bem import _fit_sphere
from mne.forward import convert_forward_solution
from mne.io.constants import FIFF
from mne.transforms import (
    Transform,
    _cart_to_sph,
    _ensure_trans,
    apply_trans,
    invert_transform,
    read_trans,
)
from mne.utils import logger, verbose
from scipy.spatial import cKDTree

# from mne.externals.six import string_types
from six import string_types

from .utils import _find_indices_1d, get_morph_src_mapping


@verbose
def select_vertices_in_sensor_range(
    inst, dist, info=None, picks=None, trans=None, indices=False, verbose=None
):
    """Find vertices within given distance to a sensor.

    Parameters
    ----------
    inst : instance of Forward | instance of SourceSpaces
        The object to select vertices from.
    dist : float
        The minimum distance between a vertex and the nearest sensor. All
        vertices for which the distance to the nearest sensor exceeds this
        limit are discarded.
    info : instance of Info | None
        The info structure that contains information about the channels. Only
        needs to be specified if the object to select vertices from does is
        an instance of SourceSpaces.
    picks : array-like of int | None
        Indices of sensors to include in the search for the nearest sensor. If
        ``None``, the default, only MEG channels are used.
    trans : str | instance of Transform | None
        Either the full path to the head<->MRI transform ``*-trans.fif`` file
        produced during coregistration, or the Transformation itself. If trans
        is None, an identity matrix is assumed. Only needed when ``inst`` is a
        source space in MRI coordinates.
    indices: False | True
        If ``True``, return vertex indices instead of vertex numbers. Defaults
        to ``False``.
    verbose : bool | str | int | None
        If not None, override default verbose level (see :func:`mne.verbose`
        and :ref:`Logging documentation <tut_logging>` for more).

    Returns
    -------
    vertices : pair of lists | list of int
        Either a list of vertex numbers for the left and right hemisphere (if
        ``indices==False``) or a single list with vertex indices.

    See Also
    --------
    restrict_forward_to_vertices : restrict Forward to the given vertices
    restrict_src_to_vertices : restrict SourceSpaces to the given vertices
    """
    if isinstance(inst, Forward):
        info = inst["info"]
        src = inst["src"]
    elif isinstance(inst, SourceSpaces):
        src = inst
        if info is None:
            raise ValueError(
                "You need to specify an Info object with "
                "information about the channels."
            )

    # Load the head<->MRI transform if necessary
    if src[0]["coord_frame"] == FIFF.FIFFV_COORD_MRI:
        if trans is None:
            raise ValueError(
                "Source space is in MRI coordinates, but no "
                "head<->MRI transform was given. Please specify "
                "the full path to the appropriate *-trans.fif "
                'file as the "trans" parameter.'
            )
        if isinstance(trans, string_types):
            trans = read_trans(trans, return_all=True)
            for trans in trans:  # we got at least 1
                try:
                    trans = _ensure_trans(trans, "head", "mri")
                except Exception:
                    pass
                else:
                    break
            else:
                raise ValueError("No head->MRI transform.")

        src_trans = invert_transform(_ensure_trans(trans, "head", "mri"))
        print("Transform!")
    else:
        src_trans = Transform("head", "head")  # Identity transform

    dev_to_head = _ensure_trans(info["dev_head_t"], "meg", "head")

    if picks is None:
        try:
            picks = _picks_to_idx(info, "meg")
        except ValueError:
            picks = []
        if len(picks) > 0:
            logger.info("Using MEG channels")
        else:
            logger.info("Using EEG channels")
            picks = _picks_to_idx(info, "eeg")

    src_pos = np.vstack(
        [apply_trans(src_trans, s["rr"][s["inuse"].astype("bool")]) for s in src]
    )

    sensor_pos = []
    for ch in picks:
        # MEG channels are in device coordinates, translate them to head
        if channel_type(info, ch) in ["mag", "grad"]:
            sensor_pos.append(apply_trans(dev_to_head, info["chs"][ch]["loc"][:3]))
        else:
            sensor_pos.append(info["chs"][ch]["loc"][:3])
    sensor_pos = np.array(sensor_pos)

    # Find vertices that are within range of a sensor. We use a KD-tree for
    # speed.
    logger.info("Finding vertices within sensor range...")
    tree = cKDTree(sensor_pos)
    distances, _ = tree.query(src_pos, distance_upper_bound=dist)

    # Vertices out of range are flagged as np.inf
    src_sel = np.isfinite(distances)
    logger.info("[done]")

    if indices:
        return np.flatnonzero(src_sel)
    else:
        n_lh_verts = src[0]["nuse"]
        lh_sel, rh_sel = src_sel[:n_lh_verts], src_sel[n_lh_verts:]
        vert_lh = src[0]["vertno"][lh_sel]
        vert_rh = src[1]["vertno"][rh_sel]
        return [vert_lh, vert_rh]


@verbose
def restrict_forward_to_vertices(
    fwd, vertno_or_idx, check_vertno=True, copy=True, verbose=None
):
    """Restrict the forward model to the given vertices.

    .. note :: The order of the vertices in ``vertno_or_idx`` does not matter.
               Forward objects will always have the vertices ordered by vertex
               number. This also means this function cannot be used to re-order
               the rows of the leadfield matrix.

    Parameters
    ----------
    fwd : instance of Forward
        The forward operator to restrict the vertices of.
    vertno_or_idx : tuple of lists (vertno_lh, vertno_rh) | list of int
        Either, for each hemisphere, the vertex numbers to keep. Or a single
        list of vertex indices to keep. All other vertices are discarded.
    check_vertno : bool
        Whether to check that all requested vertices are present in the forward
        solution and raise an IndexError if this is not the case. Defaults to
        True. If all vertices are guaranteed to be present, you can disable
        this check for avoid unnecessary computation.
    copy : bool
        Whether to operate in place (``False``) to on a copy (``True``, the
        default).
    verbose : bool | str | int | None
        If not None, override default verbose level (see :func:`mne.verbose`
        and :ref:`Logging documentation <tut_logging>` for more).

    Returns
    -------
    fwd_out : instance of Forward
        The restricted forward operator.

    See Also
    --------
    select_vertices_sens_distance : Find the vertices within the given sensor
                                    distance.
    """
    if copy:
        fwd_out = deepcopy(fwd)
    else:
        fwd_out = fwd

    lh_vertno, rh_vertno = [src["vertno"] for src in fwd["src"]]

    if isinstance(vertno_or_idx[0], int):
        logger.info("Interpreting given vertno_or_idx as vertex indices.")
        vertno_or_idx = np.asarray(vertno_or_idx)

        # Make sure the vertices are in sequential order
        fwd_idx = np.sort(vertno_or_idx)

        n_vert_lh = len(lh_vertno)
        sel_lh_idx = vertno_or_idx[fwd_idx < n_vert_lh]
        sel_rh_idx = vertno_or_idx[fwd_idx >= n_vert_lh] - n_vert_lh
        sel_lh_vertno = lh_vertno[sel_lh_idx]
        sel_rh_vertno = rh_vertno[sel_rh_idx]
    else:
        logger.info("Interpreting given vertno_or_idx as vertex numbers.")

        # Make sure vertno_or_idx is sorted
        vertno_or_idx = [np.sort(v) for v in vertno_or_idx]

        sel_lh_vertno, sel_rh_vertno = vertno_or_idx
        src_lh_idx = _find_indices_1d(lh_vertno, sel_lh_vertno, check_vertno)
        src_rh_idx = _find_indices_1d(rh_vertno, sel_rh_vertno, check_vertno)
        fwd_idx = np.hstack((src_lh_idx, src_rh_idx + len(lh_vertno)))

    logger.info(
        "Restricting forward solution to %d out of %d vertices."
        % (len(fwd_idx), len(lh_vertno) + len(rh_vertno))
    )

    n_orient = fwd["sol"]["ncol"] // fwd["nsource"]
    n_orig_orient = fwd["_orig_sol"].shape[1] // fwd["nsource"]

    fwd_out["source_rr"] = fwd["source_rr"][fwd_idx]
    fwd_out["nsource"] = len(fwd_idx)

    def _reshape_select(X, dim3, sel):
        """Make matrix X 3D and select along the second dimension."""
        dim1 = X.shape[0]
        X = X.reshape(dim1, -1, dim3)
        X = X[:, sel, :]
        return X.reshape(dim1, -1)

    fwd_out["source_nn"] = _reshape_select(fwd["source_nn"].T, n_orient, fwd_idx).T
    fwd_out["sol"]["data"] = _reshape_select(fwd["sol"]["data"], n_orient, fwd_idx)
    fwd_out["sol"]["ncol"] = fwd_out["sol"]["data"].shape[1]

    if "sol_grad" in fwd and fwd["sol_grad"] is not None:
        fwd_out["sol_grad"] = _reshape_select(fwd["sol_grad"], n_orient, fwd_idx)
    if "_orig_sol" in fwd:
        fwd_out["_orig_sol"] = _reshape_select(fwd["_orig_sol"], n_orig_orient, fwd_idx)
    if "_orig_sol_grad" in fwd and fwd["_orig_sol_grad"] is not None:
        fwd_out["_orig_sol_grad"] = _reshape_select(
            fwd["_orig_sol_grad"], n_orig_orient, fwd_idx
        )

    # Restrict the SourceSpaces inside the forward operator
    fwd_out["src"] = restrict_src_to_vertices(
        fwd_out["src"],
        [sel_lh_vertno, sel_rh_vertno],
        check_vertno=False,
        verbose=False,
    )

    return fwd_out


@verbose
def restrict_src_to_vertices(
    src, vertno_or_idx, check_vertno=True, copy=True, verbose=None
):
    """Restrict a source space to the given vertices.

    .. note :: The order of the vertices in ``vertno_or_idx`` does not matter.
               SourceSpaces objects will always have the vertices ordered by
               vertex number.

    Parameters
    ----------
    src: instance of SourceSpaces
        The source space to be restricted.
    vertno_or_idx : tuple of lists (vertno_lh, vertno_rh) | list of int
        Either, for each hemisphere, the vertex numbers to keep. Or a single
        list of vertex indices to keep. All other vertices are discarded.
    check_vertno : bool
        Whether to check that all requested vertices are present in the
        SourceSpaces and raise an IndexError if this is not the case. Defaults
        to True. If all vertices are guaranteed to be present, you can disable
        this check for avoid unnecessary computation.
    copy : bool
        Whether to operate in place (``False``) to on a copy (``True``, the
        default).
    verbose : bool | str | int | None
        If not None, override default verbose level (see :func:`mne.verbose`
        and :ref:`Logging documentation <tut_logging>` for more).

    Returns
    -------
    src_out : instance of SourceSpaces
        The restricted source space.
    """
    if copy:
        src_out = deepcopy(src)
    else:
        src_out = src

    if vertno_or_idx:
        if isinstance(vertno_or_idx[0], int):
            logger.info("Interpreting given vertno_or_idx as vertex indices.")
            vertno_or_idx = np.asarray(vertno_or_idx)
            n_vert_lh = src[0]["nuse"]
            ind_lh = vertno_or_idx[vertno_or_idx < n_vert_lh]
            ind_rh = vertno_or_idx[vertno_or_idx >= n_vert_lh] - n_vert_lh
            vert_no_lh = src[0]["vertno"][ind_lh]
            vert_no_rh = src[1]["vertno"][ind_rh]
        else:
            logger.info("Interpreting given vertno_or_idx as vertex numbers.")
            vert_no_lh, vert_no_rh = vertno_or_idx
            if check_vertno:
                if not (
                    np.all(np.isin(vert_no_lh, src[0]["vertno"]))
                    and np.all(np.isin(vert_no_rh, src[1]["vertno"]))
                ):
                    raise ValueError(
                        "One or more vertices were not present in SourceSpaces."
                    )

    else:
        # Empty list
        vert_no_lh, vert_no_rh = [], []

    logger.info(
        "Restricting source space to %d out of %d vertices."
        % (len(vert_no_lh) + len(vert_no_rh), src[0]["nuse"] + src[1]["nuse"])
    )

    for hemi, verts in zip(src_out, (vert_no_lh, vert_no_rh)):
        # Ensure vertices are in sequential order
        verts = np.sort(verts)

        # Restrict the source space
        hemi["vertno"] = verts
        hemi["nuse"] = len(verts)
        hemi["inuse"] = hemi["inuse"].copy()
        hemi["inuse"].fill(0)
        if hemi["nuse"] > 0:  # Don't use empty array as index
            hemi["inuse"][verts] = 1
        hemi["use_tris"] = np.array([[]], int)
        hemi["nuse_tri"] = np.array([0])

    return src_out


@verbose
def restrict_forward_to_sensor_range(fwd, dist, picks=None, verbose=None):
    """Restrict forward operator to sources within given distance to a sensor.

    For each vertex defined in the source space, finds the nearest sensor and
    discards the vertex if the distance to this sensor the given
    distance.

    Parameters
    ----------
    fwd : instance of Forward
        The forward operator to restrict the vertices of.
    dist : float
        The minimum distance between a vertex and the nearest sensor (in
        meters). All vertices for which the distance to the nearest sensor
        exceeds this limit are discarded.
    picks : array-like of int | None
        Indices of sensors to include in the search for the nearest sensor. If
        None, the default, meg channels are used.
    verbose : bool | str | int | None
        If not None, override default verbose level (see :func:`mne.verbose`
        and :ref:`Logging documentation <tut_logging>` for more).

    Returns
    -------
    fwd_out : instance of Forward
        A copy of the forward operator, restricted to the given sensor range

    See Also
    --------
    restrict_fwd_to_stc : Restrict the forward operator to the vertices defined
                          in a source estimate object.
    restrict_fwd_to_label : Restrict the forward operator to specific labels.
    """
    vertno = select_vertices_in_sensor_range(fwd, dist, picks, verbose=verbose)
    return restrict_forward_to_vertices(fwd, vertno, verbose=verbose)


def _make_radial_coord_system(points, origin):
    """Compute a radial coordinate system at the given points.

    For each point X, a set of three unit vectors is computed that point along
    the axes of a radial coordinate system. The first axis of the coordinate
    system is in the direction of the line between X and the origin point. The
    second and third axes are perpendicular to the first axis.

    Parameters
    ----------
    points : ndarray, shape (n_points, 3)
        For each point, the XYZ Cartesian coordinates.
    origin : (x, y, z)
        A tuple (or other array-like) containing the XYZ Cartesian coordinates
        of the point of origin. This can for example be the center of a sphere
        fitted through the points.

    Returns
    -------
    radial : ndarray, shape (n_points, 3)
        For each point X, a unit vector pointing in the radial direction, i.e.,
        the direction of the line between X and the origin point. This is the
        first axis of the coordinate system.
    tan1 : ndarray, shape (n_points, 3)
        For each point, a unit vector perpendicular to both ``radial`` and
        ``tan2``. This is the second axis of the coordinate system.
    tan2 : ndarray, shape (n_points, 3)
        For each point, a unit vector perpendicular to both ``radial`` and
        ``tan1``. This is the third axis of the coordinate system.
    """
    radial = points - origin
    radial /= np.linalg.norm(radial, axis=1)[:, np.newaxis]
    theta = _cart_to_sph(radial)[:, 1]

    # Compute tangential directions
    tan1 = np.vstack((-np.sin(theta), np.cos(theta), np.zeros(len(points)))).T
    tan2 = np.cross(radial, tan1)

    return radial, tan1, tan2


def _plot_coord_system(points, dim1, dim2, dim3, scale=0.001, n_ori=3):
    """Plot the results of _make_radial_coord_system.

    Usage:
    >>> _, origin = _fit_sphere(fwd['source_rr'])
    ... rad, tan1, tan2 = _make_radial_coord_system(fwd['source_rr'], origin)
    ... _plot_coord_system(fwd['source_rr'], rad, tan1, tan2)

    Use ``scale`` to control the size of the arrows.
    """
    from mayavi import mlab

    f = mlab.figure(size=(600, 600))
    red, blue, black = (1, 0, 0), (0, 0, 1), (0, 0, 0)
    if n_ori == 3:
        mlab.quiver3d(
            points[:, 0],
            points[:, 1],
            points[:, 2],
            dim1[:, 0],
            dim1[:, 1],
            dim1[:, 2],
            scale_factor=scale,
            color=red,
        )

    if n_ori > 1:
        mlab.quiver3d(
            points[:, 0],
            points[:, 1],
            points[:, 2],
            dim2[:, 0],
            dim2[:, 1],
            dim2[:, 2],
            scale_factor=scale,
            color=blue,
        )

    mlab.quiver3d(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        dim3[:, 0],
        dim3[:, 1],
        dim3[:, 2],
        scale_factor=scale,
        color=black,
    )
    return f


def forward_to_tangential(fwd, center=None):
    """Convert a free orientation forward solution to a tangential one.

    Places two source dipoles at each vertex that are oriented tangentially to
    a sphere with its origin at the center of the brain. Recomputes the forward
    model according to the new dipoles.

    Parameters
    ----------
    fwd : instance of Forward
        The forward solution to convert.
    center : tuple of float (x, y, z) | None
        The Cartesian coordinates of the center of the brain. By default, a
        sphere is fitted through all the points in the source space.

    Returns
    -------
    fwd_out : instance of Forward
        The tangential forward solution.
    """
    if fwd["source_ori"] != FIFF.FIFFV_MNE_FREE_ORI:
        raise ValueError("Forward solution needs to have free orientation.")

    n_sources, n_channels = fwd["nsource"], fwd["nchan"]

    if fwd["sol"]["ncol"] // n_sources == 2:
        raise ValueError(
            "Forward solution already seems to be in tangential orientation."
        )

    # Compute two dipole directions tangential to a sphere that has its origin
    # in the center of the brain.
    if center is None:
        _, center = _fit_sphere(fwd["source_rr"])
        _, tan1, tan2 = _make_radial_coord_system(fwd["source_rr"], center)

    # Make sure the forward solution is in head orientation for this
    fwd_out = convert_forward_solution(fwd, surf_ori=False, copy=True)
    G = fwd_out["sol"]["data"].reshape(n_channels, n_sources, 3)

    # Compute the forward solution for the new dipoles
    Phi = np.einsum("ijk,ljk->ijl", G, [tan1, tan2])
    fwd_out["sol"]["data"] = Phi.reshape(n_channels, 2 * n_sources)
    fwd_out["sol"]["ncol"] = 2 * n_sources

    # Store the source orientations
    fwd_out["source_nn"] = np.stack((tan1, tan2), axis=1).reshape(-1, 3)

    # Mark the orientation as free for now. In the future we should add a
    # new constant to indicate "tangential" orientations.
    fwd_out["source_ori"] = FIFF.FIFFV_MNE_FREE_ORI

    return fwd_out


def select_shared_vertices(insts, ref_src=None, subjects_dir=None):
    """Select the vertices that are present in each of the given objects.

    Produces a list of vertices which are present in each of the given objects.
    Objects can either be instances of SourceSpaces or Forward.

    If the given source spaces are from different subjects, each vertex number
    will not necessarily refer to the same vertex in each source space. In this
    case, supply the source space that will be use as a reference point as the
    ``ref_src`` parameter. All source spaces will be morphed to the reference
    source space to determine corresponding vertices between subjects.

    Parameters
    ----------
    insts : list of instance of (SourceSpaces | Forward)
        The objects to select the vertices from. Each object can have a
        different number of vertices defined.

    ref_src : instance of SourceSpaces | None
        The source space to use as reference point to determine corresponding
        vertices between subjects. If ``None`` (the default), vertex numbers
        are assumed to correspond to the same vertex in all source spaces.

    subjects_dir : str | None
        Path to SUBJECTS_DIR if it is not set in the environment. Only needed
        if ``ref_src`` is specified.

    Returns
    -------
    vertices : two lists | list of tuple of lists
        Two lists with the selected vertex numbers in each hemisphere. If
        ``ref_subject`` is specified, for each object, two lists with the
        selected vertex numbers in each hemisphere.
    """
    src_spaces = []
    for inst in insts:
        if isinstance(inst, SourceSpaces):
            src_spaces.append(inst)
        elif isinstance(inst, Forward):
            src_spaces.append(inst["src"])
        else:
            raise ValueError(
                "Given instances must either be of type "
                "SourceSpaces or Forward, not %s." % type(inst)
            )

    if ref_src is not None:
        # Map the vertex numbers to the reference source space and vice-versa
        ref_to_subj = list()
        subj_to_ref = list()
        for src in src_spaces:
            mappings = get_morph_src_mapping(ref_src, src, subjects_dir=subjects_dir)
            ref_to_subj.append(mappings[0])
            subj_to_ref.append(mappings[1])

        vert_lh = ref_src[0]["vertno"]
        vert_rh = ref_src[1]["vertno"]
    else:
        vert_lh = src_spaces[0][0]["vertno"]
        vert_rh = src_spaces[0][1]["vertno"]

    # Drop any vertices missing from one of the source spaces from the list
    for i, src in enumerate(src_spaces):
        subj_vert_lh = src[0]["vertno"]
        subj_vert_rh = src[1]["vertno"]

        if ref_src is not None:
            # Map vertex numbers to reference source space
            subj_vert_lh = [subj_to_ref[i][0][v] for v in subj_vert_lh]
            subj_vert_rh = [subj_to_ref[i][1][v] for v in subj_vert_rh]

        vert_lh = np.intersect1d(vert_lh, subj_vert_lh)
        vert_rh = np.intersect1d(vert_rh, subj_vert_rh)

    if ref_src is not None:
        # Map vertex numbers from reference source space to each source space
        verts_lh = [
            np.array([ref_to_subj[i][0][v] for v in vert_lh])
            for i in range(len(src_spaces))
        ]
        verts_rh = [
            np.array([ref_to_subj[i][1][v] for v in vert_rh])
            for i in range(len(src_spaces))
        ]
        return list(zip(verts_lh, verts_rh))
    else:
        return [vert_lh, vert_rh]
