import os.path as op

import numpy as np
import pytest
from conpy.connectivity import VertexConnectivity
from conpy.stats import cluster_threshold, group_connectivity_ttest
from mne import read_forward_solution
from mne.datasets import testing
from numpy.testing import assert_array_equal
from scipy.spatial.distance import cdist, pdist

rand_gen = np.random.RandomState(42)
data_path = testing.data_path(download=False)
fname_raw = op.join(data_path, "MEG", "sample", "sample_audvis_trunc_raw.fif")
fname_event = op.join(data_path, "MEG", "sample", "sample_audvis_trunc_raw-eve.fif")
fname_fwd = op.join(
    data_path, "MEG", "sample", "sample_audvis_trunc-meg-eeg-oct-4-fwd.fif"
)


def _load_src():
    """Load forward model and generate pairs."""
    # Find distances for subset of vertices
    fwd = read_forward_solution(fname_fwd)
    return fwd["src"]


def _make_random_connectivity(pairs, mu=0, sigma=1, inds=None, val=None, vertices=None):
    """Generate random all-to-all Connectivity object.

    Set predefined indices to defined value.
    """
    pairs = np.asarray(pairs)
    n_sources = pairs.max() + 1
    if vertices is None:
        vertices = [np.arange(n_sources / 2), np.arange(n_sources / 2, n_sources)]
    data = mu + sigma * rand_gen.randn(len(pairs[0]))
    if inds is not None and isinstance(val, int):
        data[inds] = val

    return VertexConnectivity(data, pairs, vertices)


def test_group_connectivity_ttest():
    """Test group connectivity t-test."""
    n_subjects = 5
    mus = rand_gen.rand(n_subjects)
    sigmas = rand_gen.rand(n_subjects)
    # Generate random pairs
    n_sources = 10
    vert_from, vert_to = np.triu_indices(n_sources, k=1)
    vert_from, vert_to = np.triu_indices(n_sources, k=1)
    pairs = np.asarray([vert_from, vert_to])
    inds = rand_gen.choice(pairs.shape[1], 10)
    cond1 = [
        _make_random_connectivity(pairs, mu, sigma, inds, 0.98)
        for mu, sigma in zip(mus, sigmas)
    ]
    cond2 = [
        _make_random_connectivity(pairs, mu + 0.25, sigma, 0.5)
        for mu, sigma in zip(mus, sigmas)
    ]

    # Check that results make sense
    t, pval = group_connectivity_ttest(cond1, cond2)
    assert t.shape[0] == pairs.shape[1]
    assert pval.shape[0] == pairs.shape[1]
    assert_array_equal(np.where(pval < 0.00001), inds.sort())
    t_left, p_left = group_connectivity_ttest(cond1, cond2, tail="left")
    t_right, p_right = group_connectivity_ttest(cond1, cond2, tail="right")
    assert_array_equal(t, t_left)
    assert_array_equal(t, t_right)
    assert np.allclose(p_left + p_right, 1)

    # Different pairs
    pairs3 = pairs - 1
    cond3 = [
        _make_random_connectivity(pairs3, mu + 0.5, sigma)
        for mu, sigma in zip(mus, sigmas)
    ]
    with pytest.raises(ValueError):
        t, pval = group_connectivity_ttest(cond1, cond3)

    # Different number of subjects
    cond4 = [
        _make_random_connectivity(pairs, mu, sigma)
        for mu, sigma in zip(mus[:-1], sigmas[:-1])
    ]
    with pytest.raises(ValueError):
        t, pval = group_connectivity_ttest(cond1, cond4)

    # Incorrect tail
    with pytest.raises(ValueError):
        t, pval = group_connectivity_ttest(cond1, cond2, tail="wrong")


@testing.requires_testing_data
def test_clustering():
    """Test clustering functionality."""
    src = _load_src()
    n_verts = 40
    max_spread = 0.02
    vertices = [src[0]["vertno"][:n_verts], src[1]["vertno"][:n_verts]]
    grid_points = np.vstack([s["rr"][v] for s, v in zip(src, vertices)])

    # Create one-to-all pairs in left hemisphere with pairwise distance
    # below max_spread
    dist_lh = cdist(grid_points[:n_verts], grid_points[:n_verts])
    from_lh, to_lh = np.where(dist_lh < max_spread)
    counts = np.bincount(from_lh)
    min_size = counts.max()
    chosen_inds = from_lh == np.argmax(counts)
    pairs = np.array([from_lh[chosen_inds], to_lh[chosen_inds]])

    # Add some random connections with large enough distances
    # that they should not survive
    dist = pdist(grid_points)
    pairs_rand = np.array(np.triu_indices(2 * len(vertices[0]), k=1))
    indices = rand_gen.choice(np.where(dist > 2 * max_spread + 0.01)[0], 20)
    pairs = np.hstack((pairs, pairs_rand[:, indices]))

    # Make random contrast
    contrast = _make_random_connectivity(pairs, sigma=0.6, vertices=vertices)
    # Check that the resulting connectivity makes sense
    contrast_thresh = cluster_threshold(
        contrast, src, min_size=min_size, max_spread=max_spread
    )
    assert contrast_thresh.n_connections == min_size
    assert contrast_thresh.n_sources == contrast.n_sources
    assert_array_equal(contrast_thresh.pairs, contrast.pairs[:, :min_size])
