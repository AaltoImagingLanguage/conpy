.. _api_documentation:

=================
API Documentation
=================

.. currentmodule:: conpy


Forward models
==============

.. autosummary::
   :toctree: generated/

   forward_to_tangential
   restrict_forward_to_sensor_range
   restrict_forward_to_vertices
   restrict_src_to_vertices
   select_shared_vertices
   select_vertices_in_sensor_range


Connectivity
============

Classes:

.. autosummary::
   :toctree: generated/
   :template: class.rst

   LabelConnectivity
   VertexConnectivity


Functions:

.. autosummary::
   :toctree: generated/

    all_to_all_connectivity_pairs
    dics_connectivity
    one_to_all_connectivity_pairs
    read_connectivity


Statistics
==========

.. autosummary::
   :toctree: generated/

    cluster_threshold
    group_connectivity_ttest
    cluster_permutation_test


Utilities
=========

.. currentmodule:: conpy.utils

.. autosummary::
   :toctree: generated/

   get_morph_src_mapping
