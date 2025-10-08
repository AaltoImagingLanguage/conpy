from __future__ import division

import os.path as op
import warnings

import mne
import numpy as np
import pytest
from conpy import (
    LabelConnectivity,
    VertexConnectivity,
    all_to_all_connectivity_pairs,
    dics_connectivity,
    forward_to_tangential,
    one_to_all_connectivity_pairs,
    read_connectivity,
    restrict_forward_to_vertices,
    dics_coherence_external
)
from conpy.connectivity import _BaseConnectivity, _get_vert_ind_from_label
from mne import BiHemiLabel, Label, SourceEstimate
from mne.beamformer import make_dics
from mne.datasets import testing
from mne.time_frequency import csd_morlet
from mne.utils import _TempDir
from numpy.testing import assert_array_equal

# Silence these warnings
warnings.simplefilter("ignore", category=DeprecationWarning)

data_path = testing.data_path(download=False)
fname_raw = op.join(data_path, "MEG", "sample", "sample_audvis_trunc_raw.fif")
fname_event = op.join(data_path, "MEG", "sample", "sample_audvis_trunc_raw-eve.fif")
fname_fwd = op.join(
    data_path, "MEG", "sample", "sample_audvis_trunc-meg-eeg-oct-4-fwd.fif"
)
subjects_dir = op.join(data_path, "subjects")
random = np.random.RandomState(42)


def _load_forward():
    """Load forward model."""
    return mne.read_forward_solution(fname_fwd)


def _load_restricted_forward(source_vertno1, source_vertno2):
    """Load forward models and restrict them to the given source vertices."""
    fwd_free = mne.read_forward_solution(fname_fwd)
    fwd_free = mne.pick_types_forward(fwd_free, meg="grad", eeg=False)

    # Restrict forward
    vertno_lh = np.random.choice(fwd_free["src"][0]["vertno"], 40, replace=False)
    if source_vertno1 not in vertno_lh:
        vertno_lh = np.append(vertno_lh, source_vertno1)
    vertno_rh = np.random.choice(fwd_free["src"][1]["vertno"], 40, replace=False)
    if source_vertno2 not in vertno_rh:
        vertno_rh = np.append(vertno_rh, source_vertno2)

    fwd_free = restrict_forward_to_vertices(fwd_free, [vertno_lh, vertno_rh])
    fwd_fixed = mne.convert_forward_solution(fwd_free, force_fixed=True, use_cps=False)

    return fwd_free, fwd_fixed


def _simulate_data(fwd_fixed, source_vertno1, source_vertno2, external=False):
    """Simulate two oscillators on the cortex."""
    sfreq = 50.0  # Hz.
    base_freq = 10
    t_rand = 0.001
    std = 0.1
    times = np.arange(10.0 * sfreq) / sfreq  # 10 seconds of data
    n_times = len(times)
    # Generate an oscillator with varying frequency and phase lag.
    iflaw = base_freq / sfreq + t_rand * np.random.randn(n_times)
    signal1 = np.exp(1j * 2.0 * np.pi * np.cumsum(iflaw))
    signal1 *= np.conj(signal1[0])
    signal1 = signal1.real

    # Add some random fluctuations to the signal.
    signal1 += std * np.random.randn(n_times)
    signal1 *= 1e-7

    # Make identical signal
    signal2 = signal1.copy()

    # Add random fluctuations
    signal1 += 1e-8 * np.random.randn(len(times))
    signal2 += 1e-8 * np.random.randn(len(times))

    # Construct a SourceEstimate object
    if external:
        source_data = signal1[np.newaxis, :]
        vertices = [np.array([source_vertno1]), np.array([])]
        # Create an info object that holds information about the sensors
        info = mne.create_info(
            fwd_fixed["info"]["ch_names"] + ["external"],
            sfreq,
            ch_types=["grad"] * fwd_fixed["info"]["nchan"] + ["misc"],
        )
    else:
        source_data = np.vstack((signal1[np.newaxis, :], signal2[np.newaxis, :]))
        vertices = [np.array([source_vertno1]), np.array([source_vertno2])]
        # Create an info object that holds information about the sensors
        info = mne.create_info(fwd_fixed["info"]["ch_names"], sfreq,
                               ch_types="grad")

    stc = mne.SourceEstimate(
        source_data,
        vertices=vertices,
        tmin=0,
        tstep=1 / sfreq,
        subject="sample",
    )
    # Merge in sensor position information
    for info_ch, fwd_ch in zip(info["chs"], fwd_fixed["info"]["chs"]):
        info_ch.update(fwd_ch)

    # Simulated sensor data.
    raw = mne.apply_forward_raw(fwd_fixed, stc, info)

    # Add noise
    noise = random.randn(*raw._data.shape) * 1e-14
    raw._data += noise

    if external:
        sensor_data = raw.get_data()
        sensor_data = np.vstack((sensor_data, np.atleast_2d(signal2)))
        raw = mne.io.RawArray(sensor_data, info)

    # Define a single epoch
    epochs = mne.Epochs(
        raw,
        np.array([[0, 0, 1]]),
        event_id=1,
        tmin=0,
        tmax=raw.times[-1],
        preload=True,
        baseline=(0, 0),
    )

    # Compute the cross-spectral density matrix
    csd = csd_morlet(epochs, picks=['meg', 'misc'], frequencies=[10, 20])

    return csd


def _make_base_connectivity():
    pairs = ([1, 1, 2, 2, 3], [2, 3, 3, 4, 4])
    n_sources = 10
    data = np.arange(1, len(pairs[1]) + 1)

    return _BaseConnectivity(data, pairs, n_sources)


def _make_alltoall_connectivity():
    vertices = [[1, 2, 3], [1, 2]]
    pairs = [[0, 1, 2], [2, 3, 4]]
    data = np.arange(1, len(pairs[1]) + 1)

    return VertexConnectivity(data, pairs, vertices)


def _make_label_connectivity():
    labels = [
        Label(vertices=np.arange(3), hemi="lh", name="Label1"),
        Label(vertices=np.arange(3, 6), hemi="lh", name="Label2"),
        Label(vertices=np.arange(6, 9), hemi="lh", name="Label3"),
    ]

    pairs = [[0, 0, 1], [1, 2, 2]]
    data = np.arange(len(pairs[1]))

    return LabelConnectivity(data, pairs, labels)


def _generate_labels(vertices, n_labels):
    vert_lh, vert_rh = vertices
    n_lh_chunck = len(vert_lh) // (n_labels // 2)
    n_rh_chunck = len(vert_rh) // (n_labels - n_labels // 2)

    labels_lh = [
        Label(vertices=vert_lh[x : x + n_lh_chunck], hemi="lh", name="Label" + str(x))
        for x in range(0, len(vert_lh), n_lh_chunck)
    ]

    labels_rh = [
        Label(vertices=vert_rh[x : x + n_rh_chunck], hemi="rh", name="Label" + str(x))
        for x in range(0, len(vert_rh), n_lh_chunck)
    ]

    return labels_lh + labels_rh


def test_base_connectivity():
    """Test construction of BaseConnectivity."""
    # Pairs and data shape don't match
    base_con = _BaseConnectivity([0.5, 0.5, 0.5], [[1, 1, 2], [2, 3, 3]], 4)
    assert_array_equal(base_con.data, [0.5, 0.5, 0.5])

    with pytest.raises(ValueError):
        base_con = _BaseConnectivity([0.5, 0.5, 0.5], [[1, 1], [2, 3]], 3)

    # Not enough sources
    with pytest.raises(ValueError):
        base_con = _BaseConnectivity([0.5, 0.5, 0.5], [[1, 1, 2], [2, 3, 3]], 2)

    #  Incorrecly shaped source degree
    with pytest.raises(ValueError):
        base_con = _BaseConnectivity(
            [0.5, 0.5, 0.5],
            [[1, 1, 2], [2, 3, 3]],
            4,
            source_degree=([0, 2, 1], [0, 0, 1]),
        )

    base_con = _make_base_connectivity()
    assert_array_equal(
        base_con.source_degree,
        np.array([[0, 2, 2, 1, 0, 0, 0, 0, 0, 0], [0, 0, 1, 2, 2, 0, 0, 0, 0, 0]]),
    )

    # Test properties
    assert base_con.n_connections == 5
    state = {
        "data": np.array([1, 1, 1]),
        "pairs": [[2, 2, 3], [3, 4, 4]],
        "n_sources": 5,
        "subject": None,
        "directed": False,
    }
    base_con.__setstate__(state)
    assert base_con.n_sources == 5
    assert_array_equal(base_con.source_degree[0], [0, 0, 2, 1, 0])


def test_alltoall_connectivity():
    """Test construction of VertexConnectivity."""
    all_con = VertexConnectivity(
        [1, 2, 3, 4], [[0, 0, 1, 2], [1, 2, 2, 3]], vertices=[[1, 2, 3], [4]]
    )
    assert all_con.n_sources == 4
    assert_array_equal(all_con.source_degree, ([2, 1, 1, 0], [0, 1, 2, 1]))

    # Vertices not a list of two
    with pytest.raises(ValueError):
        all_con = VertexConnectivity(
            [1, 2, 3],
            [
                [
                    0,
                    0,
                    1,
                ],
                [1, 2, 2],
            ],
            vertices=[1, 2, 3],
        )


def test_label_connnectivity():
    """Test construction of LabelConnectivity."""
    labels = _generate_labels([np.arange(12), np.arange(12)], 4)
    pairs = [[0, 0, 1, 2], [1, 2, 3, 3]]
    data = [0, 0.5, 0.75, 1]
    # Incorrect input
    with pytest.raises(ValueError):
        LabelConnectivity(data, pairs, labels[0])

    label_con = LabelConnectivity(data, pairs, labels)
    assert label_con.n_sources == 4
    assert_array_equal(label_con.source_degree, ([2, 1, 1, 0], [0, 1, 1, 2]))

    # Test set state
    state = {
        "data": np.array([1, 1, 1]),
        "pairs": [[0, 0, 1], [1, 2, 2]],
        "labels": [
            [np.arange(4), None, None, "lh", "", "Label1"],
            [np.arange(4, 8), None, None, "lh", "", "Label2"],
            [np.arange(8, 12), None, None, "lh", "", "Label3"],
        ],
        "n_sources": 3,
        "subject": None,
        "directed": False,
    }
    label_con.__setstate__(state)
    assert label_con.n_sources == 3
    assert_array_equal(label_con.labels[0].vertices, np.arange(4))


def test_connectivity_repr():
    """Test string representation of connectivity classes."""
    base_con = _make_base_connectivity()
    assert str(base_con) == (
        "<_BaseConnectivity  |  n_sources=10, n_conns=5," " subject=None>"
    )

    all_con = _make_alltoall_connectivity()
    assert str(all_con) == (
        "<VertexConnectivity  |  n_sources=5, n_conns=3," " subject=None>"
    )

    label_con = _make_label_connectivity()
    assert str(label_con) == (
        "<LabelConnectivity  |  n_sources=3, n_conns=3," " subject=None>"
    )


"""Test functions of BaseConnectivity object"""


def test_connectivity_save():
    """Test saving and loading connectivity objects."""
    all_con = _make_alltoall_connectivity()
    tempdir = _TempDir()
    fname = op.join(tempdir, "coh.h5")
    all_con.save(fname)
    all_con2 = read_connectivity(fname)

    assert_array_equal(all_con.data, all_con2.data)
    assert_array_equal(all_con.pairs, all_con2.pairs)
    assert_array_equal(all_con.source_degree, all_con2.source_degree)
    assert_array_equal(all_con.vertices[0], all_con2.vertices[0])
    assert_array_equal(all_con.vertices[1], all_con2.vertices[1])

    assert all_con.n_connections == all_con2.n_connections
    assert all_con.n_sources == all_con2.n_sources
    assert all_con.subject == all_con2.subject
    assert all_con.directed == all_con2.directed

    fname = op.join(tempdir, "coh")
    all_con.save(fname)
    all_con3 = read_connectivity(fname)
    assert all_con.n_connections == all_con3.n_connections
    assert all_con.n_sources == all_con3.n_sources
    assert all_con.subject == all_con3.subject
    assert all_con.directed == all_con3.directed

    label_con = _make_label_connectivity()
    tempdir = _TempDir()
    fname = op.join(tempdir, "coh.h5")
    label_con.save(fname)
    label_con2 = read_connectivity(fname)

    assert_array_equal(label_con.data, label_con2.data)
    assert_array_equal(label_con.pairs, label_con2.pairs)
    assert_array_equal(label_con.source_degree, label_con2.source_degree)
    assert label_con.n_connections == label_con2.n_connections
    assert label_con.n_sources == label_con2.n_sources
    assert label_con.subject == label_con2.subject
    assert label_con.directed == label_con2.directed
    assert len(label_con.labels) == len(label_con2.labels)
    assert isinstance(label_con2.labels[0], Label)


def test_adjacency():
    """Test adjacency matrix."""
    base_con = _make_base_connectivity()
    adjmat = base_con.get_adjacency()
    assert adjmat.nnz == 2 * base_con.n_connections
    assert adjmat.shape == (base_con.n_sources, base_con.n_sources)

    # Directed
    base_con.directed = True
    adjmat = base_con.get_adjacency()
    assert adjmat.nnz == base_con.n_connections
    assert adjmat.shape == (base_con.n_sources, base_con.n_sources)


def test_connectivity_threshold():
    """Test thresholding function of BaseConnectivity."""
    # Criterion = None
    base_con = _make_base_connectivity()
    threshCon = base_con.threshold(2, copy=True)
    assert threshCon.n_connections == 3
    assert base_con.n_connections == 5
    assert_array_equal(threshCon.data, np.array([3, 4, 5]))

    threshCon = base_con.copy()
    threshCon.threshold(2, direction="below", copy=False)
    assert threshCon.n_connections == 1
    assert_array_equal(threshCon.data, np.array([1]))

    # Incorrect direction
    with pytest.raises(ValueError):
        base_con.threshold(1, direction="wrong")

    # Use criterion
    with pytest.raises(ValueError):
        base_con.threshold(1, crit=np.array([0, 0, 1, 2]))

    pval = np.array([0.04, 0.7, 0.001, 0.1, 0.06])
    threshCon = base_con.threshold(0.05, crit=pval, copy=True)
    assert_array_equal(threshCon.data, np.array([2, 4, 5]))


def test_compatibility():
    """Test _iscombatible function."""
    base_con = _make_base_connectivity()
    all_con = _make_alltoall_connectivity()
    label_con = _make_label_connectivity()

    # Test BaseConnectivity
    assert not base_con.is_compatible(label_con)
    assert not base_con.is_compatible(all_con)
    base_con2 = _BaseConnectivity(
        np.array([6, 7, 8, 9, 10]), base_con.pairs, base_con.n_sources
    )
    assert base_con.is_compatible(base_con2)

    # Test VertexConnectivity
    assert not all_con.is_compatible(label_con)
    all_con2 = VertexConnectivity(np.array([4, 5, 6]), all_con.pairs, all_con.vertices)
    assert all_con.is_compatible(all_con2)

    # Test label_connectivity
    assert not label_con.is_compatible(all_con)
    label_con2 = LabelConnectivity(
        np.array([3, 4, 5]), pairs=label_con.pairs, labels=label_con.labels
    )
    assert label_con.is_compatible(label_con2)


def test_operations():
    """Test arithmetic operations of connectivity objects."""
    all_con = _make_alltoall_connectivity()
    all_con2 = VertexConnectivity(np.array([4, 5, 6]), all_con.pairs, all_con.vertices)

    # Test operations
    assert_array_equal((all_con + all_con2).data, [5, 7, 9])
    assert_array_equal((all_con - all_con2).data, [-3, -3, -3])
    assert_array_equal((all_con / all_con2).data, [0.25, 0.4, 0.5])
    assert_array_equal((all_con * all_con2).data, [4, 10, 18])
    assert_array_equal((all_con**all_con2).data, [1, 32, 729])
    assert_array_equal(-all_con.data, [-1, -2, -3])

    # Test in-place operations
    all_con3 = all_con.copy()
    all_con3 += all_con2
    assert_array_equal(all_con3.data, [5, 7, 9])

    all_con3 = all_con.copy()
    all_con3 -= all_con2
    assert_array_equal(all_con3.data, [-3, -3, -3])

    all_con3 = all_con.copy()
    all_con3.data = all_con3.data.astype(float)
    all_con3 /= all_con2
    assert_array_equal(all_con3.data, [0.25, 0.4, 0.5])

    all_con3 = all_con.copy()
    all_con3 *= all_con2
    assert_array_equal(all_con3.data, [4, 10, 18])

    all_con3 = all_con.copy()
    all_con3 **= all_con2
    assert_array_equal(all_con3.data, [1, 32, 729])


def test_make_stc():
    """Test VertexConnectivity.make_stc()."""
    all_con = _make_alltoall_connectivity()

    with pytest.raises(ValueError):
        all_con.make_stc(summary="False")

    all_con_broken = all_con.copy()
    all_con_broken.vertices = None
    with pytest.raises(ValueError):
        all_con_broken.make_stc()

    stc = all_con.make_stc(summary="degree", weight_by_degree=False)
    assert isinstance(stc, SourceEstimate)
    assert_array_equal(stc.data.flatten(), [1, 1, 2, 1, 1])
    assert_array_equal(all_con.vertices[0], stc.vertices[0])
    assert_array_equal(all_con.vertices[1], stc.vertices[1])

    stc = all_con.make_stc(summary="degree", weight_by_degree=True)
    assert_array_equal(stc.data.flatten(), [1, 1, 1, 1, 1])

    stc = all_con.make_stc(summary="sum", weight_by_degree=False)
    assert isinstance(stc, SourceEstimate)
    assert_array_equal(stc.data.flatten(), np.array([1, 2, 4, 2, 3]))
    assert_array_equal(all_con.vertices[0], stc.vertices[0])
    assert_array_equal(all_con.vertices[1], stc.vertices[1])

    stc = all_con.make_stc(summary="sum", weight_by_degree=True)
    assert_array_equal(stc.data.flatten(), np.array([1, 2, 2, 2, 3]))


def test_parcellate():
    """Test VertexConnectivity.parcellate()."""
    vertices = [np.arange(10), np.arange(10)]
    pairs = [
        np.array([0, 0, 0, 0, 1, 1, 1, 2, 2, 3, 4, 5, 5, 6, 6]),
        np.array([1, 2, 3, 4, 2, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]),
    ]

    all_con = VertexConnectivity(
        np.ones(
            15,
        ),
        pairs,
        vertices,
    )

    labels = _generate_labels(vertices, 4)

    # Test incorrect input: Labels not a list
    with pytest.raises(ValueError):
        label_con1 = all_con.parcellate(labels[0], weight_by_degree=False)

    # Subjects don't match
    all_con2 = _make_alltoall_connectivity()
    all_con2.subject = "Test1"
    labels2 = _generate_labels(vertices, 4)
    labels2[0].subject = "Test2"
    with pytest.raises(RuntimeError):
        all_con2.parcellate(labels2, weight_by_degree=False)

    # No weighting, degree
    label_con1 = all_con.parcellate(labels, weight_by_degree=False)
    assert_array_equal(label_con1.pairs[0], np.array([0, 0, 1, 1]))
    assert_array_equal(label_con1.data, np.array([3, 3, 2, 2]))

    # Weighting and degree
    label_con2 = all_con.parcellate(labels, weight_by_degree=True)
    assert_array_equal(
        label_con2.data, np.array([3.0 / 14, 3.0 / 16, 2.0 / 9, 2.0 / 6])
    )

    # Labels that doesn't have vertices in all_con
    labels2 = [
        Label(vertices=np.array([20, 21, 22, 23]), hemi="lh", name="Label1"),
        Label(vertices=np.array([11, 12, 13, 14]), hemi="lh", name="Label2"),
    ]
    label_con2 = all_con.parcellate(labels2, weight_by_degree=True)
    assert label_con2.data.size == 0
    # Sum
    label_con3 = all_con.parcellate(labels, summary="sum", weight_by_degree=False)
    assert_array_equal(label_con3.data, np.array([3, 3, 2, 2]))
    label_con4 = all_con.parcellate(labels, summary="sum")
    assert_array_equal(
        label_con4.data, np.array([3.0 / 14, 3.0 / 16, 2.0 / 9, 2.0 / 6])
    )

    # Absmax
    label_con5 = all_con.parcellate(labels, summary="absmax", weight_by_degree=False)
    assert_array_equal(label_con5.data, np.array([1.0, 1.0, 1.0, 1.0]))
    label_con5 = all_con.parcellate(labels2, summary="absmax", weight_by_degree=True)
    assert label_con5.data.size == 0

    # Other function
    def summary(c, f, t):
        return c[f, :][:, t].mean()

    label_con6 = all_con.parcellate(labels, summary=summary, weight_by_degree=False)
    assert_array_equal(
        label_con6.data, np.array([3.0 / 25, 3.0 / 25, 2.0 / 25, 2.0 / 25])
    )

    # Incorrect input
    with pytest.raises(ValueError):
        all_con.parcellate(labels, summary="incorrect", weight_by_degree=False)


def test_vert_ind_from_Label():
    """Test _get_vert_ind_from_label."""
    label_lh = Label(np.arange(10), hemi="lh", name="Label1-lh")
    label_rh = Label(np.arange(10), hemi="rh", name="Label1-rh")
    bilabel = BiHemiLabel(label_lh, label_rh, name="Label1")
    vertices = [np.arange(5), np.arange(5, 10)]

    inds_lh = _get_vert_ind_from_label(vertices, label_lh)
    inds_rh = _get_vert_ind_from_label(vertices, label_rh)
    assert_array_equal(inds_lh, np.arange(5))
    assert_array_equal(inds_rh, np.arange(5, 10))

    inds = _get_vert_ind_from_label(vertices, bilabel)
    assert_array_equal(inds, np.arange(10))

    # Incorrect input
    with pytest.raises(TypeError):
        _get_vert_ind_from_label(vertices, "label")


@testing.requires_testing_data
def test_all_to_all_connectivity_pairs():
    """Test creating all-to-all connectivity pairs."""
    fwd = _load_forward()
    # Test using forward
    total_vertices = len(fwd["source_rr"])
    # All should pass
    pairs = all_to_all_connectivity_pairs(fwd, min_dist=0)
    assert max(pairs[0].max(), pairs[1].max()) == total_vertices - 1
    # None should pass
    pairs = all_to_all_connectivity_pairs(fwd, min_dist=1)
    assert pairs[0].size == 0
    assert pairs[1].size == 0

    # Test using SourceSpaces
    # All should pass
    pairs = all_to_all_connectivity_pairs(fwd["src"], min_dist=0)
    assert max(pairs[0].max(), pairs[1].max()) == total_vertices - 1
    # None should pass
    pairs = all_to_all_connectivity_pairs(fwd["src"], min_dist=1)
    assert pairs[0].size == 0
    assert pairs[1].size == 0

    # Incorrect input
    with pytest.raises(ValueError):
        all_to_all_connectivity_pairs([], min_dist=1)


@testing.requires_testing_data
def test_one_to_all_connectivity_pairs():
    """Test creating one-to-all connectivity pairs."""
    fwd = _load_forward()
    # Test using forward
    total_vertices = len(fwd["source_rr"])
    # All should pass
    pairs = one_to_all_connectivity_pairs(fwd, 0, min_dist=0)
    assert max(pairs[0].max(), pairs[1].max()) == total_vertices - 1
    # None should pass
    pairs = one_to_all_connectivity_pairs(fwd, 0, min_dist=1)
    assert pairs[0].size == 0
    assert pairs[1].size == 0
    assert np.all(pairs[0] == 0)
    assert not np.any(pairs[1] == 0)

    # Test using SourceSpaces
    total_vertices = len(fwd["source_rr"])
    # All should pass
    pairs = one_to_all_connectivity_pairs(fwd["src"], 0, min_dist=0)
    assert max(pairs[0].max(), pairs[1].max()) == total_vertices - 1
    # None should pass
    pairs = one_to_all_connectivity_pairs(fwd["src"], 0, min_dist=1)
    assert pairs[0].size == 0
    assert pairs[1].size == 0

    # Incorrect input
    with pytest.raises(ValueError):
        one_to_all_connectivity_pairs([], 0, min_dist=1)


@testing.requires_testing_data
def test_dics_connectivity():
    """Test dics_connectivity function."""
    source_vertno1 = 146374  # Somewhere on the frontal lobe
    source_vertno2 = 33830
    # Load restricted forwards
    fwd, fwd_fixed = _load_restricted_forward(source_vertno1, source_vertno2)
    csd = _simulate_data(fwd_fixed, source_vertno1, source_vertno2)
    # Take subset of pairs to make calculations faster
    pairs = all_to_all_connectivity_pairs(fwd, 0.05)

    # CSD not averaged
    with pytest.raises(ValueError):
        con = dics_connectivity(pairs, fwd, csd)
    csd = csd.mean()
    # Fixed forward
    with pytest.raises(ValueError):
        dics_connectivity(pairs, fwd_fixed, csd)
    # Incorrect pairs
    with pytest.raises(ValueError):
        dics_connectivity([[0, 1, 3], [2, 4]], fwd, csd)

    con = dics_connectivity(pairs, fwd, csd, reg=1)
    max_ind = np.argmax(con.data)
    vertices = np.concatenate(con.vertices)
    assert len(con.data) == len(pairs[0])
    assert con.n_sources == fwd["nsource"]
    assert vertices[pairs[0][max_ind]] == source_vertno1
    assert vertices[pairs[1][max_ind]] == source_vertno2
    # Check result is the same with tangential
    fwd_tan = forward_to_tangential(fwd)
    con2 = dics_connectivity(pairs, fwd_tan, csd, reg=1)
    assert_array_equal(con2.data, con.data)


@testing.requires_testing_data
def test_dics_coherence_external():
    """Test dics_coherence_external function."""
    fwd = _load_forward()
    fwd = mne.pick_types_forward(fwd, meg="grad", eeg=False)
    fwd_fixed = mne.convert_forward_solution(fwd, force_fixed=True)
    source_vert = 146374
    sfreq = 50.0
    csd = _simulate_data(fwd_fixed, source_vert, None, external=True)
    print(csd.ch_names)
    # Create an info object that holds information about the sensors (their
    # location, etc.). Make sure to include the external sensor!
    info = mne.create_info(
        fwd["info"]["ch_names"] + ["external"],
        sfreq,
        ch_types=["grad"] * fwd["info"]["nchan"] + ["misc"],
    )
    # Copy grad positions from the forward solution
    for info_ch, fwd_ch in zip(info["chs"], fwd["info"]["chs"]):
        info_ch.update(fwd_ch)

    dics = make_dics(info.copy(), fwd, csd.copy(), reg=1,
                     inversion="single",
                     pick_ori=None)
    dics_matrix = make_dics(info.copy(), fwd, csd.copy(), reg=1,
                            inversion="matrix",
                            pick_ori=None)
    dics_fixed = make_dics(info.copy(), fwd, csd.copy(), reg=1,
                           inversion="single",
                           pick_ori='max-power')
    # Tangential source space
    with pytest.raises(ValueError):
        dics_tangential = dics.copy()
        dics_tangential["weights"] = np.delete(
            dics_tangential["weights"],
            np.arange(0, dics["weights"].shape[1], 3), axis=1)
        dics_coherence_external(csd, dics_tangential, info, fwd,
                                external='external',
                                pick_ori='max-coherence')
    # Max-power ori selected but dics has vector ori
    with pytest.raises(ValueError):
        dics_coherence_external(csd, dics, info, fwd, external='external',
                                pick_ori='max-power')
    # Max-coherence ori selected but dics has fixed ori
    with pytest.raises(ValueError):
        dics_coherence_external(csd, dics_fixed, info, fwd, external='external',
                                pick_ori='max-coherence')
    # Max-coherence ori selected but dics was computed using matrix inversion
    with pytest.raises(ValueError):
        dics_coherence_external(csd, dics_matrix, info, fwd,
                                external='external', pick_ori='max-coherence')
    # Incorrect pick_ori
    with pytest.raises(ValueError):
        dics_coherence_external(csd, dics, info, fwd, external='external',
                                pick_ori='normal')
    # Max power orientation
    coh_stc = dics_coherence_external(csd, dics_fixed, info, fwd,
                                      external='external', pick_ori='max-power')
    assert isinstance(coh_stc, SourceEstimate)
    assert coh_stc.data.shape == (fwd['nsource'], 2)  # 2 frequencies
    assert np.max(coh_stc.data) <= 1 and np.min(coh_stc.data) >= 0
    # Max coherence orientation
    coh_stc = dics_coherence_external(csd, dics, info, fwd,
                                      external='external',
                                      pick_ori='max-coherence')
    assert isinstance(coh_stc, SourceEstimate)
    assert coh_stc.data.shape == (fwd['nsource'], 2)  # 2 frequencies
    assert np.max(coh_stc.data) <= 1 and np.min(coh_stc.data) >= 0
