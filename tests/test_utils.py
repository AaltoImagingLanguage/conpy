import numpy as np
import pytest
from conpy.utils import reg_pinv
from numpy.testing import assert_array_equal


def testreg_pinv():
    """Test regularization and inversion of covariance matrix."""
    # create rank-deficient array
    a = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 1.0]])

    # Test if rank-deficient matrix without regularization throws
    # specific warning
    with pytest.warns(RuntimeWarning, match="deficient"):
        reg_pinv(a, reg=0.0)

    # Test inversion with explicit rank
    a_inv_np = np.linalg.pinv(a)
    a_inv_mne, loading_factor, rank = reg_pinv(a, rank=2)
    assert loading_factor == 0
    assert rank == 2
    assert_array_equal(a_inv_np, a_inv_mne)

    # Test inversion with automatic rank detection
    a_inv_mne, _, estimated_rank = reg_pinv(a, rank=None)
    assert_array_equal(a_inv_np, a_inv_mne)
    assert estimated_rank == 2

    # Test adding regularization
    a_inv_mne, loading_factor, estimated_rank = reg_pinv(a, reg=2)
    # Since A has a diagonal of all ones, loading_factor should equal the
    # regularization parameter
    assert loading_factor == 2
    # The estimated rank should be that of the non-regularized matrix
    assert estimated_rank == 2
    # Test result against the NumPy version
    a_inv_np = np.linalg.pinv(a + loading_factor * np.eye(3))
    assert_array_equal(a_inv_np, a_inv_mne)

    # Test setting rcond
    a_inv_np = np.linalg.pinv(a, rcond=0.5)
    a_inv_mne, _, estimated_rank = reg_pinv(a, rcond=0.5)
    assert_array_equal(a_inv_np, a_inv_mne)
    assert estimated_rank == 1

    # Test inverting an all zero cov
    a_inv, loading_factor, estimated_rank = reg_pinv(np.zeros((3, 3)), reg=2)
    assert_array_equal(a_inv, 0)
    assert loading_factor == 0
    assert estimated_rank == 0
