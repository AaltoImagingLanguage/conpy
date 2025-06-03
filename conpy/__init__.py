"""Python module for computing spectral connectivity using DICS."""

from . import utils
from .connectivity import (
    LabelConnectivity,
    VertexConnectivity,
    all_to_all_connectivity_pairs,
    dics_connectivity,
    one_to_all_connectivity_pairs,
    read_connectivity,
)
from .forward import (
    forward_to_tangential,
    restrict_forward_to_sensor_range,
    restrict_forward_to_vertices,
    restrict_src_to_vertices,
    select_shared_vertices,
    select_vertices_in_sensor_range,
)
from .stats import cluster_permutation_test, cluster_threshold, group_connectivity_ttest

__version__ = "1.4-dev"
