.. _api_documentation:

=================
API Documentation
=================

.. currentmodule:: conpy


Forward models
==============

.. autosummary::
    :toctree: functions
    :template: function.rst

    forward_to_tangential
    restrict_forward_to_sensor_range
    restrict_forward_to_vertices
    restrict_src_to_vertices
    select_shared_vertices
    select_vertices_in_sensor_range


Connectivity
============

.. currentmodule:: conpy.connectivity

Classes:

.. autosummary::
    :toctree: functions/
    :template: class.rst

    LabelConnectivity
    VertexConnectivity


Functions:

.. autosummary::
    :toctree: functions
    :template: function.rst

    all_to_all_connectivity_pairs
    dics_connectivity
    one_to_all_connectivity_pairs
    read_connectivity
    dics_coherence_external


Statistics
==========

.. currentmodule:: conpy.stats

.. autosummary::
    :toctree: functions
    :template: function.rst

    cluster_threshold
    group_connectivity_ttest
    cluster_permutation_test


Utilities
=========

.. currentmodule:: conpy.utils

.. autosummary::
    :toctree: functions
    :template: function.rst

   get_morph_src_mapping
   reg_pinv
