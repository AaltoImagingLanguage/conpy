import os.path as op

import mne
import numpy as np
import pytest
from conpy import (
    forward_to_tangential,
    restrict_forward_to_sensor_range,
    restrict_forward_to_vertices,
    restrict_src_to_vertices,
    select_shared_vertices,
    select_vertices_in_sensor_range,
)
from conpy.forward import _find_indices_1d, _make_radial_coord_system
from numpy.testing import assert_array_equal


@pytest.fixture
def fwd():
    """Make a forward solution."""
    path = mne.datasets.sample.data_path()
    return mne.read_forward_solution(
        op.join(path, "MEG", "sample", "sample_audvis-meg-oct-6-fwd.fif")
    )


@pytest.fixture
def src():
    """Make a source space."""
    path = mne.datasets.sample.data_path()
    return mne.read_source_spaces(
        op.join(path, "subjects", "sample", "bem", "sample-oct-6-src.fif")
    )


def _trans():
    path = mne.datasets.sample.data_path()
    return op.join(path, "MEG", "sample", "sample_audvis_raw-trans.fif")


def _info():
    path = mne.datasets.sample.data_path()
    return mne.io.read_info(op.join(path, "MEG", "sample", "sample_audvis_raw.fif"))


def test_find_indices_1d():
    """Test finding indices."""
    assert_array_equal(_find_indices_1d([], []), [])
    assert_array_equal(_find_indices_1d([0], [0]), [0])
    assert_array_equal(_find_indices_1d([0, 1, 2], [2]), [2])
    assert_array_equal(_find_indices_1d([2, 1, 0], [2]), [0])
    assert_array_equal(_find_indices_1d([2, 1, 0], [2, 1, 0]), [0, 1, 2])

    with pytest.raises(IndexError):
        _find_indices_1d([0, 1, 2], [3])


def test_restrict_src_to_vertices(src):
    """Test restricting a source space to the given vertices."""
    # Vertex numbers
    src_r = restrict_src_to_vertices(src, ([1170, 1609], [2159]))
    assert_array_equal(src_r[0]["vertno"], [1170, 1609])
    assert_array_equal(src_r[1]["vertno"], [2159])
    # No vertices selected
    src_r = restrict_src_to_vertices(src, ([], []))
    assert_array_equal(src_r[0]["vertno"], [])
    assert_array_equal(src_r[1]["vertno"], [])
    # Input contains vertices not in use
    with pytest.raises(ValueError):
        src_r = restrict_src_to_vertices(src, (np.arange(25), [2159]))
    with pytest.raises(ValueError):
        src_r = restrict_src_to_vertices(src, ([1170, 1609], [29, 30]))
    with pytest.raises(ValueError):
        src_r = restrict_src_to_vertices(src, ([13, 1609], [29, 30]))
    # Indices
    src_r = restrict_src_to_vertices(src, [0, 1, 3, src[0]["nuse"], src[0]["nuse"] + 1])
    assert_array_equal(src_r[0]["vertno"], src[0]["vertno"][np.array([0, 1, 3])])
    assert_array_equal(src_r[1]["vertno"], src[1]["vertno"][np.array([0, 1])])
    src_r = restrict_src_to_vertices(src, [])
    assert_array_equal(src_r[0]["vertno"], [])
    assert_array_equal(src_r[1]["vertno"], [])

    # Test in place operation
    restrict_src_to_vertices(src, ([1170, 1609], [2159]), copy=False)
    assert_array_equal(src[0]["vertno"], [1170, 1609])
    assert_array_equal(src[1]["vertno"], [2159])


def test_restrict_forward_to_vertices(fwd):
    """Test restricting a forward solution to the given vertices."""
    fwd_r = restrict_forward_to_vertices(fwd, ([1170, 1609], [2159]))
    assert_array_equal(fwd_r["src"][0]["vertno"], [1170, 1609])
    assert_array_equal(fwd_r["src"][1]["vertno"], [2159])
    assert fwd_r["sol"]["ncol"] == 3 * 3  # Free orientation
    assert fwd_r["sol"]["nrow"] == fwd["sol"]["nrow"]
    assert fwd_r["sol"]["data"].shape == (fwd["sol"]["nrow"], 3 * 3)

    fwd_r = restrict_forward_to_vertices(fwd, ([], []))
    assert_array_equal(fwd_r["src"][0]["vertno"], [])
    assert_array_equal(fwd_r["src"][1]["vertno"], [])

    # Test fixed orientation forward solution
    fwd_fixed = mne.forward.convert_forward_solution(
        fwd, force_fixed=True, use_cps=True
    )
    fwd_r = restrict_forward_to_vertices(fwd_fixed, ([1170, 1609], [2159]))
    assert fwd_r["sol"]["ncol"] == 3
    assert fwd_r["sol"]["data"].shape == (fwd["sol"]["nrow"], 3)

    # Test tangential forward solution
    fwd_tan = forward_to_tangential(fwd)
    fwd_r = restrict_forward_to_vertices(fwd_tan, ([1170, 1609], [2159]))
    assert fwd_r["sol"]["ncol"] == 3 * 2
    assert fwd_r["sol"]["data"].shape == (fwd["sol"]["nrow"], 3 * 2)

    # Vertices not present in src
    with pytest.raises(IndexError):
        restrict_forward_to_vertices(fwd, ([30, 1609], [2159]))

    # Use indices
    fwd_r = restrict_forward_to_vertices(fwd, [0, 1, 3732])
    assert fwd_r["sol"]["ncol"] == 3 * 3
    assert_array_equal(fwd_r["src"][0]["vertno"], fwd["src"][0]["vertno"][:2])
    assert_array_equal(fwd_r["src"][1]["vertno"], fwd["src"][1]["vertno"][:1])

    # Test in place operation
    restrict_forward_to_vertices(fwd, ([1170, 1609], [2159]), copy=False)
    assert_array_equal(fwd["src"][0]["vertno"], [1170, 1609])
    assert_array_equal(fwd["src"][1]["vertno"], [2159])


def test_select_shared_vertices(src, fwd):
    """Test selecting shared vertices between a source space and forward solution."""
    # SourceSpaces
    src2 = src.copy()
    src1_r = restrict_src_to_vertices(
        src, ([14, 54, 59, 108, 228], [30, 98, 180, 187, 230])
    )
    vert_inds = select_shared_vertices([src1_r, src2])
    assert_array_equal(vert_inds[0], src1_r[0]["vertno"])
    assert_array_equal(vert_inds[1], src1_r[1]["vertno"])

    subjects_dir = op.join(mne.datasets.sample.data_path(), "subjects")
    vert_inds = select_shared_vertices(
        [src1_r, src2], ref_src=src2, subjects_dir=subjects_dir
    )
    assert_array_equal(vert_inds[0][0], src1_r[0]["vertno"])
    assert_array_equal(vert_inds[0][1], src1_r[1]["vertno"])

    src2_r = restrict_src_to_vertices(
        src2, [[108, 228, 364, 582, 627], [187, 230, 263, 271, 313]]
    )
    vert_inds = select_shared_vertices([src1_r, src2_r, src, src2])
    assert_array_equal(vert_inds[1], np.array([187, 230]))
    assert_array_equal(vert_inds[0], np.array([108, 228]))

    # Forward
    fwd2 = fwd.copy()
    fwd1_r = restrict_forward_to_vertices(fwd, ([1170, 1609], [2159]))
    vert_inds = select_shared_vertices([fwd1_r, fwd2])
    assert_array_equal(vert_inds[0], fwd1_r["src"][0]["vertno"])
    assert_array_equal(vert_inds[1], fwd1_r["src"][1]["vertno"])

    # Incorrect input
    with pytest.raises(ValueError):
        select_shared_vertices("wrong", fwd)


def test_select_vertices_in_sensor_range(fwd, src):
    """Test selecting and restricting vertices in the sensor range."""
    fwd_r = restrict_forward_to_vertices(fwd, ([1170, 1609], [2159]))

    verts = select_vertices_in_sensor_range(fwd_r, 0.05)
    assert_array_equal(verts[0], np.array([1170]))
    assert_array_equal(verts[1], np.array([]))
    # Test indices
    verts = select_vertices_in_sensor_range(fwd_r, 0.07, indices=True)
    assert_array_equal(verts, np.array([0, 1, 2]))

    # Test restricting
    fwd_rs = restrict_forward_to_sensor_range(fwd_r, 0.05)
    assert_array_equal(fwd_rs["src"][0]["vertno"], np.array([1170]))
    assert_array_equal(fwd_rs["src"][1]["vertno"], np.array([]))

    verts = select_vertices_in_sensor_range(fwd_r, 0.07)
    assert_array_equal(verts[0], np.array([1170, 1609]))
    assert_array_equal(verts[1], np.array([2159]))

    src_r = restrict_src_to_vertices(src, ([1170, 1609], [2159]))

    with pytest.raises(ValueError):  # info missing
        select_vertices_in_sensor_range(src_r, 0.07)
    info = _info()
    with pytest.raises(ValueError):  # trans missing
        select_vertices_in_sensor_range(src_r, 0.07, info=info)

    # Correct input
    trans = _trans()
    verts2 = select_vertices_in_sensor_range(src_r, 0.05, info=info, trans=trans)
    assert_array_equal(verts2[0], np.array([1170]))
    verts2 = select_vertices_in_sensor_range(src_r, 0.07, info=info, trans=trans)
    assert_array_equal(verts[0], verts2[0])
    assert_array_equal(verts[1], verts2[1])
    # Indices
    verts2 = select_vertices_in_sensor_range(
        src_r, 0.07, info=info, trans=trans, indices=True
    )
    assert_array_equal(verts2, np.array([0, 1, 2]))
    # Try with only EEG
    info = mne.pick_info(info, sel=mne.pick_types(info, meg=False, eeg=True))
    verts2 = select_vertices_in_sensor_range(src_r, 0.05, info=info, trans=trans)
    assert_array_equal(verts2[0], np.array([1170, 1609]))
    assert_array_equal(verts2[1], np.array([2159]))


# FIXME: disabled until we can make a proper test
# def test_radial_coord_system():
#     """Test making a radial coordinate system."""
#     r = np.ones((4, 1))
#     theta = np.ones((4, 1)) * np.pi
#     phi = np.ones((4, 1)) * np.pi
#     sph = np.hstack((r, theta, phi))
#     cart = mne.transforms._sph_to_cart(sph)
#
#     rad, tan1, tan2 = _make_radial_coord_system(cart, (0, 0, 0))
#     assert_array_equal(rad, cart)  # Norm will be 1
#     assert_array_equal(tan1[:, :2], np.hstack((-np.sin(theta), np.cos(theta))))


def test_forward_to_tangential(fwd):
    """Test tangential forward solution."""
    fwd_tan = forward_to_tangential(fwd)
    # Check shapes
    assert fwd_tan["sol"]["ncol"] == fwd["nsource"] * 2
    assert fwd_tan["sol"]["data"].shape == (fwd["sol"]["nrow"], fwd["nsource"] * 2)
    assert fwd_tan["source_nn"].shape == (fwd["nsource"] * 2, 3)

    # Forward operator already tangential
    with pytest.raises(ValueError):
        forward_to_tangential(fwd_tan)
    # Fixed forward operator
    fwd_fixed = mne.convert_forward_solution(fwd, force_fixed=True, use_cps=True)
    with pytest.raises(ValueError):
        forward_to_tangential(fwd_fixed)
