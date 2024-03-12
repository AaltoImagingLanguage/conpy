"""Python module for computing spectral connectivity using DICS."""

from .forward import (restrict_forward_to_sensor_range, forward_to_tangential,
                      select_vertices_in_sensor_range, select_shared_vertices,
                      restrict_forward_to_vertices, restrict_src_to_vertices)
from .connectivity import (all_to_all_connectivity_pairs,
                           one_to_all_connectivity_pairs, dics_connectivity,
                           VertexConnectivity, LabelConnectivity,
                           read_connectivity)
from .stats import (group_connectivity_ttest, cluster_threshold,
                    cluster_permutation_test)
from . import utils

__version__ = '1.3.1'
