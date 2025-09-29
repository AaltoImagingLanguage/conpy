# -*- coding: utf-8 -*-
"""Connectivity analysis using Dynamic Imaging of Coherent Sources (DICS).

Authors: Susanna Aro <susanna.aro@aalto.fi>
         Marijn van Vliet <w.m.vanvliet@gmail.com>
"""

import copy
import types

import numpy as np
from h5io import read_hdf5, write_hdf5
from mne import BiHemiLabel, Forward, Label, SourceSpaces, pick_channels_forward
from mne.parallel import parallel_func
from mne.source_estimate import _make_stc
from mne.source_space._source_space import (
    _ensure_src,
    _ensure_src_subject,
    _get_morph_src_reordering,
)
from mne.time_frequency import pick_channels_csd
from mne.utils import copy_function_doc_to_method_doc, logger, verbose
from scipy import sparse
from scipy.spatial.distance import cdist, pdist

from .forward import forward_to_tangential
from .utils import reg_pinv
from .viz import plot_connectivity


class _BaseConnectivity(object):
    """Base class for connectivity objects.

    Contains implementation of methods that are defined for all connectivity
    objects.

    Parameters
    ----------
    data : ndarray, shape (n_pairs,)
        For each connectivity source pair, a value describing the connection.
        For example, this can be the strength of the connection between the
        sources.
    pairs : ndarray, shape (n_pairs, 2)
        The sources involved in the from-to connectivity pair. The sources
        are listed as indices of the list given as the ``sources`` parameter.
    n_sources : int
        The number of sources between which connectivity is defined.
    source_degree : tuple of lists (out_degree, in_degree) | None
        For each source, the total number of possible connections from and to
        the source. This information is needed to perform weighting on the
        number of connections during visualization and statistics. If ``None``,
        it is assumed that all possible connections are defined in the
        ``pairs`` parameter and the out- and in-degree of each source is
        computed.
    subject : str | None
        The subject-id. Defaults to ``None``.
    directed : bool
        Whether the connectivity is directed (from->to != to->from). Defaults
        to False.

    Attributes
    ----------
    n_connections : int
        The number of connections.
    """

    def __init__(
        self, data, pairs, n_sources, source_degree=None, subject=None, directed=False
    ):
        self.data = np.asarray(data)
        pairs = np.asarray(pairs)
        if pairs.shape[1] != len(data):
            raise ValueError(
                "The number of pairs does not match the number "
                "of items in the data list."
            )

        if pairs.shape[1] > 0 and n_sources < pairs.max():
            raise ValueError(
                "Pairs are defined between non-existent sources "
                "(n_sources=%d)." % n_sources
            )

        self.pairs = pairs
        self.n_sources = n_sources
        self.subject = subject
        self.directed = directed

        if source_degree is not None:
            source_degree = np.asarray(source_degree)
            if source_degree.shape[1] != n_sources:
                raise ValueError(
                    "The length of the source_degree list does "
                    "not match the number of sources."
                )
            self.source_degree = source_degree
        else:
            self.source_degree = np.asarray(_compute_degree(pairs, n_sources))

    def __repr__(self):
        return "<{}  |  n_sources={}, n_conns={}, subject={}>".format(
            self.__class__.__name__, self.n_sources, self.n_connections, self.subject
        )

    @property
    def n_connections(self):
        """The number of connections."""
        return len(self.data)

    def copy(self):
        """Return copy of the Connectivity object."""
        return copy.deepcopy(self)

    def __setstate__(self, state):  # noqa: D105
        self.data = state["data"]
        self.pairs = state["pairs"]
        self.n_sources = state["n_sources"]

        if "source_degree" in state:
            self.source_degree = state["source_degree"]
        else:
            self.source_degree = _compute_degree(self.pairs, self.n_sources)

        self.subject = state["subject"]
        self.directed = state["directed"]

    def __getstate__(self):  # noqa: D105
        return dict(
            data=self.data,
            pairs=self.pairs,
            subject=self.subject,
            n_sources=self.n_sources,
            source_degree=self.source_degree,
            directed=self.directed,
        )

    def save(self, fname):
        """Save the connectivity object to an HDF5 file.

        Parameters
        ----------
        fname : str
            The name of the file to save the connectivity to. The extension
            '.h5' will be appended if the given filename doesn't have it
            already.

        See Also
        --------
        read_connectivity : For reading connectivity objects from a file.
        """
        if not fname.endswith(".h5"):
            fname += ".h5"

        write_hdf5(fname, self.__getstate__(), overwrite=True, title="conpy")

    def get_adjacency(self):
        """Get a source-to-source adjacency matrix.

        Each non-zero element in the matrix indicates a connection exists
        between the sources. The value of the element is the strength of the
        connection.
        """
        A = sparse.csr_matrix(
            (self.data, self.pairs),
            shape=(self.n_sources, self.n_sources),
        )

        if self.directed:
            return A
        else:
            return A + A.T

    def threshold(self, thresh, crit=None, direction="above", copy=False):
        """Threshold the connectivity.

        Only retain the connections which exceed a given threshold.

        Parameters
        ----------
        thresh : float
            threshold limit
        crit : None | ndarray, shape (n_connections,)
            An array containing for each connection, a value which must pass
            the threshold for the connection to be retained. By default, this
            is the data value of the connection. Common uses for this parameter
            include thresholding connections based on t-values or p-values.
        direction: 'above' | 'below'
            Defines whether the `thres_data` must be above or below the given
            threshold in order for the vertex-pair to be retained. Defaults to
            'above'.
        copy : bool
            Whether to operate in place (``False``, the default) or on a copy
            (``True``).

        Returns
        -------
        thresholded_con : instance of Connectivity
            The thresholded version of the connectivity.
        """
        if crit is None:
            crit = self.data
        elif len(crit) != self.n_connections:
            raise ValueError(
                "The number of items in `crit` does not match "
                "the number of connections."
            )

        # Convert crit into a binary mask
        if direction == "above":
            mask = crit > thresh
        elif direction == "below":
            mask = crit < thresh
        else:
            raise ValueError(
                'The direction parameter must be either "above" ' 'or "below".'
            )

        if copy:
            thresholded_con = self.copy()
        else:
            thresholded_con = self

        thresholded_con.data = self.data[mask]
        thresholded_con.pairs = self.pairs[:, mask]

        return thresholded_con

    def __getitem__(self, index):
        """Select connections without making a deep copy."""
        # Create an "empty" connection object
        con = self.__class__.__new__(self.__class__)

        # Construct the fields for the newconnection object.
        state = self.__getstate__()
        state["data"] = self.data[index]
        state["pairs"] = self.pairs[:, index]

        # Set the fields of the new connection object
        con.__setstate__(state)

        return con

    def is_compatible(self, other):
        """Check compatibility with another connectivity object.

        Two connectivity objects are compatible if they define the same
        connectivity pairs.

        Returns
        -------
        is_compatible : bool
            Whether the given connectivity object is compatible with this one.
        """
        return (
            isinstance(other, _BaseConnectivity)
            and other.n_sources == self.n_sources
            and np.array_equal(other.pairs, self.pairs)
        )

    def __iadd__(self, other):  # noqa: D105
        if self.is_compatible(other):
            self.data += other.data
        return self

    def __add__(self, other):  # noqa: D105
        return self.copy().__iadd__(other)

    def __isub__(self, other):  # noqa: D105
        if self.is_compatible(other):
            self.data -= other.data
        return self

    def __sub__(self, other):  # noqa: D105
        return self.copy().__isub__(other)

    def __idiv__(self, other):  # noqa: D105
        if self.is_compatible(other):
            self.data /= other.data
        return self

    def __div__(self, other):  # noqa: D105
        con = self.copy()
        # Always use floating point for division
        con.data = con.data.astype("float")
        return con.__idiv__(other)

    def __truediv__(self, other):  # noqa: D105
        return self.__div__(other)

    def __itruediv__(self, other):  # noqa: D105
        con = self.copy()
        # Always use floating point for division
        con.data = con.data.astype("float")
        return con.__idiv__(other)

    def __imul__(self, other):  # noqa: D105
        if self.is_compatible(other):
            self.data *= other.data
        return self

    def __mul__(self, other):  # noqa: D105
        return self.copy().__imul__(other)

    def __ipow__(self, other):  # noqa: D105
        if self.is_compatible(other):
            self.data **= other.data
        return self

    def __pow__(self, other):  # noqa: D105
        return self.copy().__ipow__(other)

    def __neg__(self):  # noqa: D105
        self.data *= -1
        return self

    def __radd__(self, other):  # noqa: D105
        return self + other

    def __rsub__(self, other):  # noqa: D105
        return self - other

    def __rmul__(self, other):  # noqa: D105
        return self * other

    def __rdiv__(self, other):  # noqa: D105
        return self / other


def _compute_degree(pairs, n_sources):
    """Compute out- and in- degree of each source.

    Computes for each source, the number of connections from and to the source.

    Parameters
    ----------
    pairs : ndarray, shape (n_pairs, 2)
        The indices of the sources involved in the from-to connectivity pair.
    n_sources : int
        The total number of sources.

    Returns
    -------
    out_degree : ndarray, shape (n_sources,)
        The number of outgoing connections for each source.
    in_degree : ndarray, shape (n_sources,)
        The number of incoming connections for each source.
    """
    out_degree = np.zeros(n_sources, dtype=int)
    ind, degree = np.unique(pairs[0], return_counts=True)
    out_degree[ind] = degree

    in_degree = np.zeros(n_sources, dtype=int)
    ind, degree = np.unique(pairs[1], return_counts=True)
    in_degree[ind] = degree

    return out_degree, in_degree


class VertexConnectivity(_BaseConnectivity):
    """Estimation of connectivity between vertices.

    Parameters
    ----------
    data : ndarray, shape (n_pairs,)
        For each connectivity source pair, a value describing the connection.
        For example, this can be the strength of the connection between the
        sources.
    pairs : ndarray, shape (n_pairs, 2)
        The vertices involved in the from-to connectivity pair. The vertices
        are listed as "vertex indices" in the array:
            ``np.hstack((vertices[0], (vertices[1] + len(vertices[0]))))``
    vertices : list of two arrays of shape (n_vertices,)
        For each hemisphere, the vertex numbers of sources defined in the
        corresponding source space.
    vertex_degree : tuple of lists (out_degree, in_degree) | None
        For each vertex, the total number of possible connections from and to
        the vertex. This information is needed to perform weighting on the
        number of connections during visualization and statistics. If ``None``,
        it is assumed that all possible connections are defined in the
        ``pairs`` parameter and the out- and in-degree of each vertex is
        computed.
    subject : str | None
        The subject-id.
    directed : bool
        Whether the connectivity is directed (from->to != to->from). Defaults
        to False.

    Attributes
    ----------
    n_connections : int
        The number of connections.
    n_sources : int
        The number of sources between possible connections were computed.
    """

    def __init__(
        self, data, pairs, vertices, vertex_degree=None, subject=None, directed=False
    ):
        if len(vertices) != 2:
            raise ValueError(
                "The `vertices` parameter should be a list of " "two arrays."
            )

        self.vertices = [np.asarray(v) for v in vertices]
        n_vertices = len(self.vertices[0]) + len(self.vertices[1])
        super().__init__(
            data=data,
            pairs=pairs,
            n_sources=n_vertices,
            source_degree=vertex_degree,
            subject=subject,
            directed=directed,
        )

    def make_stc(self, summary="sum", weight_by_degree=True):
        """Obtain a summary of the connectivity as a SourceEstimate object.

        Parameters
        ----------
        summary : 'sum' | 'degree' | 'absmax'
            How to summarize the adjacency data:

            'sum' : sum the strenghts of both the incoming and outgoing connections
                    for each source.
            'degree': count the number of incoming and outgoing connections for each
                      source.
            'absmax' : show the strongest coherence across both incoming and outgoing
                       connections at each source. In this setting, the
                       ``weight_by_degree`` parameter is ignored.

            Defaults to ``'sum'``.

        weight_by_degree : bool
            Whether to weight the summary by the number of possible
            connections. Defaults to ``True``.

        Returns
        -------
        stc : instance of SourceEstimate
            The summary of the connectivity.
        """
        if self.vertices is None:
            raise ValueError("Stc needs vertices!")

        if summary == "degree":
            vert_inds, data = np.unique(self.pairs, return_counts=True)

            n_vert_lh = len(self.vertices[0])
            lh_inds = vert_inds < n_vert_lh
            vertices = [
                self.vertices[0][vert_inds[lh_inds]],
                self.vertices[1][vert_inds[~lh_inds] - n_vert_lh],
            ]

        elif summary == "sum":
            A = self.get_adjacency()
            data = A.sum(axis=0).T + A.sum(axis=1)
            vertices = self.vertices

            # These are needed later in order to weight by degree
            vert_inds = np.arange(len(self.vertices[0]) + len(self.vertices[1]))

            # For undirected connectivity objects, all connections have been
            # counted twice.
            if not self.directed:
                data = data / 2.0

        elif summary == "absmax":
            A = self.get_adjacency()
            in_max = A.max(axis=0).toarray().ravel()
            out_max = A.max(axis=1).toarray().ravel()
            data = np.maximum(in_max, out_max)
            vertices = self.vertices

        else:
            raise ValueError(
                'The summary parameter must be "degree", or ' '"sum", or "absmax".'
            )

        data = np.asarray(data, dtype="float").ravel()

        if weight_by_degree and summary != "absmax":
            degree = self.source_degree[:, vert_inds].sum(axis=0)
            # Prevent division by zero
            zero_mask = degree == 0
            data[~zero_mask] /= degree[~zero_mask]
            data[zero_mask] = 0

        return _make_stc(
            data[:, np.newaxis],
            vertices=vertices,
            tmin=0,
            tstep=1,
            subject=self.subject,
        )

    @verbose
    def parcellate(self, labels, summary="sum", weight_by_degree=True, verbose=None):
        """Get the connectivity parcellated according to the given labels.

        The coherence of all connections within a label are averaged.

        Parameters
        ----------
        labels : list of (Label |  BiHemiLabel)
            The labels to use to parcellate the connectivity.
        summary : 'sum' | 'degree' | 'absmax' | function
            How to summarize the connectivity within a label. Either the
            summation of the connection values ('sum'), the number of
            connections from and to the label is used ('degree'), the absolute
            maximum value of the connections ('absmax'), or a function can be
            specified, which is called for each label with the following
            signature:

            >>> def summary(adjacency, vert_from, vert_to):
            ...     '''Summarize the connections within a label.
            ...
            ...     Parameters
            ...     ----------
            ...     adjacency : sparse matrix, shape (n_sources, n_sources)
            ...         The adjacency matrix that defines the connection
            ...         between the sources.
            ...     src_from : list of int
            ...         Indices of sources that are outside of the label.
            ...     src_to : list of int
            ...         Indices of sources that are inside the label.
            ...
            ...     Returns
            ...     -------
            ...     coh : float
            ...         Summarized coherence of the parcel.

        weight_by_degree : bool
            Whether to weight the summary of each label by the number of
            possible connections from and to that label. Defaults to ``True``.
        verbose : bool | str | int | None
            If not None, override default verbose level (see
            :func:`mne.verbose` and :ref:`Logging documentation <tut_logging>`
            for more).

        Returns
        -------
        coh_parc : LabelConnectivity
            The parcellated connectivity.

        See Also
        --------
        mne.read_labels_from_annot : To read a list of labels from a FreeSurfer
                                     annotation.
        """
        if not isinstance(labels, list):
            raise ValueError("labels must be a list of labels")
        # Make sure labels and connectivity are compatible
        if (
            labels[0].subject is not None
            and self.subject is not None
            and labels[0].subject != self.subject
        ):
            raise RuntimeError(
                "label and connectivity must have same subject names, "
                'currently "%s" and "%s"' % (labels[0].subject, self.subject)
            )

        if summary == "degree":

            def summary(c, f, t):
                return float(c[f, :][:, t].nnz)
        elif summary == "sum":

            def summary(c, f, t):
                return c[f, :][:, t].sum()
        elif summary == "absmax":

            def summary(c, f, t):
                if len(f) == 0 or len(t) == 0:
                    return 0.0
                else:
                    return np.abs(c[f, :][:, t]).max()
        elif not isinstance(summary, types.FunctionType):
            raise ValueError(
                'The summary parameter must be "degree", "sum" '
                '"absmax" or a function.'
            )

        logger.info("Computing out- and in-degree for each label...")
        n_labels = len(labels)
        label_degree = np.zeros((2, n_labels), dtype=int)
        for i, label in enumerate(labels):
            vert_ind = _get_vert_ind_from_label(self.vertices, label)
            label_degree[:, i] = self.source_degree[:, vert_ind].sum(axis=1)

        logger.info("Summarizing connectivity...")

        adjacency = self.get_adjacency()
        pairs = np.triu_indices(n_labels, k=1)
        n_pairs = len(pairs[0])
        summary_parc = np.zeros(n_pairs)
        prev_from = -1
        for pair_i, (lab_from, lab_to) in enumerate(zip(*pairs)):
            if lab_from != prev_from:
                logger.info("    in %s" % labels[lab_from].name)
                prev_from = lab_from

            vert_from = _get_vert_ind_from_label(self.vertices, labels[lab_from])
            vert_to = _get_vert_ind_from_label(self.vertices, labels[lab_to])
            val = summary(adjacency, vert_from, vert_to)

            if weight_by_degree:
                degree = label_degree[0, lab_from] + label_degree[1, lab_to]
                if degree == 0:
                    # Prevent division by 0
                    val = 0
                else:
                    val /= degree

            summary_parc[pair_i] = val

        # Drop connections with a value of zero. We take this to mean that no
        # connection exists.
        nonzero_inds = np.flatnonzero(summary_parc)
        pairs = np.array(pairs)[:, nonzero_inds]
        summary_parc = summary_parc[nonzero_inds]

        logger.info("[done]")

        return LabelConnectivity(
            data=summary_parc,
            pairs=pairs,
            labels=labels,
            label_degree=label_degree,
            subject=self.subject,
        )

    def __setstate__(self, state):  # noqa: D105
        super().__setstate__(state)
        self.vertices = state["vertices"]

    def __getstate__(self):  # noqa: D105
        state = super().__getstate__()
        state.update(
            type="all-to-all",
            vertices=self.vertices,
        )
        return state

    def is_compatible(self, other):
        """Check compatibility with another connectivity object.

        Two connectivity objects are compatible if they define the same
        connectivity pairs.

        Returns
        -------
        is_compatible : bool
            Whether the given connectivity object is compatible with this one.
        """
        return (
            isinstance(other, VertexConnectivity)
            and np.array_equal(other.vertices[0], self.vertices[0])
            and np.array_equal(other.vertices[1], self.vertices[1])
            and np.array_equal(other.pairs, self.pairs)
        )

    def to_original_src(
        self, src_orig, subject_orig=None, subjects_dir=None, verbose=None
    ):
        """Get the connectivity from a morphed source to the original subject.

        Parameters
        ----------
        src_orig : instance of SourceSpaces
            The original source spaces that were morphed to the current
            subject.
        subject_orig : str | None
            The original subject. For most source spaces this shouldn't need
            to be provided, since it is stored in the source space itself.
        subjects_dir : string, or None
            Path to SUBJECTS_DIR if it is not set in the environment.
        verbose : bool | str | int | None
            If not None, override default verbose level (see
            :func:`mne.verbose` and :ref:`Logging documentation <tut_logging>`
            for more).

        Returns
        -------
        con : instance of VertexConnectivity
            The transformed connectivity.

        See Also
        --------
        mne.morph_source_spaces
        """
        if self.subject is None:
            raise ValueError("con.subject must be set")

        src_orig = _ensure_src(src_orig, kind="surface")
        subject_orig = _ensure_src_subject(src_orig, subject_orig)

        data_idx, vertices = _get_morph_src_reordering(
            vertices=self.vertices,
            src_from=src_orig,
            subject_from=subject_orig,
            subject_to=self.subject,
            subjects_dir=subjects_dir,
            verbose=verbose,
        )

        # Map the pairs to new vertices
        mapping = np.argsort(data_idx)
        pairs = [[mapping[p_] for p_ in p] for p in self.pairs]
        vertex_degree = self.source_degree[:, data_idx]
        return VertexConnectivity(
            data=self.data,
            pairs=pairs,
            vertices=vertices,
            vertex_degree=vertex_degree,
            subject=subject_orig,
        )


class LabelConnectivity(_BaseConnectivity):
    """Estimation of all-to-all connectivity, parcellated into labels.

    Parameters
    ----------
    data : ndarray, shape (n_pairs,)
        For each connectivity source pair, a value describing the connection.
        For example, this can be the strength of the connection between the
        sources.
    pairs : ndarray, shape (n_pairs, 2)
        The index of the labels involved in the from-to connectivity pair.
    labels : list of instance of Label
        The labels between which connectivity has been computed.
    label_degree : tuple of lists (out_degree, in_degree) | None
        For each label, the total number of possible connections from and to
        the label. This information is needed to perform weighting on the
        number of connections during visualization and statistics. If ``None``,
        it is assumed that all possible connections are defined in the
        ``pairs`` parameter and the out- and in-degree of each label is
        computed.
    subject : str | None
        The subject-id.

    Attributes
    ----------
    n_connections : int
        The number of connections.
    """

    def __init__(self, data, pairs, labels, label_degree=None, subject=None):
        if not isinstance(labels, list):
            raise ValueError("labels must be a list of labels")
        super().__init__(
            data=data,
            pairs=pairs,
            n_sources=len(labels),
            source_degree=label_degree,
            subject=subject,
        )
        self.labels = labels

    @copy_function_doc_to_method_doc(plot_connectivity)
    def plot(  # noqa
        self,
        n_lines=None,
        node_angles=None,
        node_width=None,
        node_colors=None,
        facecolor="black",
        textcolor="white",
        node_edgecolor="black",
        linewidth=1.5,
        colormap="hot",
        vmin=None,
        vmax=None,
        colorbar=True,
        title=None,
        colorbar_size=0.2,
        colorbar_pos=(-0.3, 0.1),
        fontsize_title=12,
        fontsize_names=8,
        fontsize_colorbar=8,
        padding=6.0,
        fig=None,
        subplot=111,
        interactive=True,
        node_linewidth=2.0,
        show=True,
    ):
        return plot_connectivity(
            self,
            n_lines=n_lines,
            node_angles=node_angles,
            node_width=node_width,
            node_colors=node_colors,
            facecolor=facecolor,
            textcolor=textcolor,
            node_edgecolor=node_edgecolor,
            linewidth=linewidth,
            colormap=colormap,
            vmin=vmin,
            vmax=vmax,
            colorbar=colorbar,
            title=title,
            colorbar_size=colorbar_size,
            colorbar_pos=colorbar_pos,
            fontsize_title=fontsize_title,
            fontsize_names=fontsize_names,
            fontsize_colorbar=fontsize_colorbar,
            padding=padding,
            fig=fig,
            subplot=subplot,
            interactive=interactive,
            node_linewidth=node_linewidth,
            show=show,
        )

    def is_compatible(self, other):
        """Check compatibility with another connectivity object.

        Two connectivity objects are compatible if they define the same
        connectivity pairs.

        Returns
        -------
        is_compatible : bool
            Whether the given connectivity object is compatible with this one.
        """
        return (
            isinstance(other, LabelConnectivity)
            and np.array_equal(other.pairs, self.pairs)
            and np.all(
                [
                    np.array_equal(l1.vertices, l2.vertices)
                    for l1, l2 in zip(other.labels, self.labels)
                ]
            )
            and np.all(
                [
                    np.array_equal(l1.values, l2.values)
                    for l1, l2 in zip(other.labels, self.labels)
                ]
            )
        )

    def __setstate__(self, state):  # noqa: D105
        super(LabelConnectivity, self).__setstate__(state)
        self.labels = [Label(*label) for label in state["labels"]]

    def __getstate__(self):  # noqa: D105
        state = super(LabelConnectivity, self).__getstate__()
        state.update(
            type="label",
            labels=[label.__getstate__() for label in self.labels],
        )
        return state


def _get_vert_ind_from_label(vertices, label):
    """Get the indices of the vertices that fall within a given label.

    Parameters
    ----------
    vertices : list of ndarray
        For each hemisphere, the vertex numbers.
    label : instance of Label | BiHemiLabel
        The label for which to get the vertex indices.

    Returns
    -------
    vertex_ind : ndarray
        The indices of the vertices that fall within the given label.
    """
    if not isinstance(label, Label) and not isinstance(label, BiHemiLabel):
        raise TypeError("Expected Label or BiHemiLabel; got %r" % label)
    if label.hemi == "both":
        vertex_ind_lh = _get_vert_ind_from_label(vertices, label.lh)
        vertex_ind_rh = _get_vert_ind_from_label(vertices, label.rh)
        return np.hstack((vertex_ind_lh, vertex_ind_rh))
    elif label.hemi == "lh":
        verts_present = np.intersect1d(vertices[0], label.vertices)
        return np.searchsorted(vertices[0], verts_present)
    elif label.hemi == "rh":
        verts_present = np.intersect1d(vertices[1], label.vertices)
        return np.searchsorted(vertices[1], verts_present) + len(vertices[0])


def read_connectivity(fname):
    """Read a Connectivity object from an HDF5 file.

    Parameters
    ----------
    fname : str
        The name of the file to read the connectivity from. The extension '.h5'
        will be appended if the given filename doesn't have it already.

    Returns
    -------
    connectivity : instance of Connectivity
        The Connectivity object that was stored in the file.

    See Also
    --------
    Connectivity.save : For saving connectivity objects
    """
    if not fname.endswith(".h5"):
        fname += ".h5"

    con_dict = read_hdf5(fname, title="conpy")
    con_type = con_dict["type"]
    del con_dict["type"]

    if con_type == "all-to-all":
        return VertexConnectivity(
            data=con_dict["data"],
            pairs=con_dict["pairs"],
            vertices=con_dict["vertices"],
            vertex_degree=con_dict["source_degree"],
            subject=con_dict["subject"],
        )
    elif con_type == "label":
        labels = [Label(**label) for label in con_dict["labels"]]
        return LabelConnectivity(
            data=con_dict["data"],
            pairs=con_dict["pairs"],
            labels=labels,
            label_degree=con_dict["source_degree"],
            subject=con_dict["subject"],
        )


def all_to_all_connectivity_pairs(src_or_fwd, min_dist=0.04):
    """Obtain pairs of vertices to compute all-to-all connectivity for.

    This is needed for all-to-all connectivity. Calculates all the pairs of
    vertices that are further away from each other than the selected distance
    limit.

    Parameters
    ----------
    src_or_fwd : instance of SourceSpaces | instance of Forwxard
        The source space or forward model to obtain vertex pairs for.
    min_dist: float
        The minimum distance between vertices (in meters). Defaults to 0.04.

    Returns
    -------
    vert_from : ndarray, shape (n_pairs,)
        For each pair, the index of the first vertex.
    vert_to : ndarray, shape (n_pairs,)
        For each pair, the index of the second vertex.

    See Also
    --------
    one_to_all_connectivity_pairs : Obtain pairs for one-to-all connectivity.
    """
    # Get coordinates of the vertices
    if isinstance(src_or_fwd, SourceSpaces):
        vertno_lh = src_or_fwd[0]["vertno"]
        vertno_rh = src_or_fwd[1]["vertno"]
        grid_points = np.vstack(
            (src_or_fwd[0]["rr"][vertno_lh], src_or_fwd[1]["rr"][vertno_rh])
        )
    elif isinstance(src_or_fwd, Forward):
        grid_points = src_or_fwd["source_rr"]
    else:
        raise ValueError(
            "Source must be instance of Forward or a list", "of SourceSpaces"
        )

    n_sources = len(grid_points)

    # Compute indices of all pairs
    vert_from, vert_to = np.triu_indices(n_sources, k=1)

    # Select the pairs that are further away than the distance limit
    selection = pdist(grid_points) >= min_dist

    # Converting this to a list of tuples is very slow, so let's keep it like
    # this for now.
    return vert_from[selection], vert_to[selection]


def one_to_all_connectivity_pairs(src_or_fwd, ref_point, min_dist=0):
    """Obtain pairs of vertices to compute one-to-all connectivity for.

    This is needed for one-to-all connectivity. Calculates all the pairs where
    the vertex is further away from reference point than the selected distance
    limit.

    Parameters
    ----------
    src_or_fwd : instance of SourceSpaces | instance of Forward
        The source space or forward model to obtain vertex pairs for.
    ref_point: int
        Index of the vertex that will serve as reference point.
    min_dist: float
        The minimum distance between vertices (in meters). Defaults to 0.

    Returns
    -------
    vert_from : ndarray, shape (n_pairs,)
        For each pair, the index of the first vertex. This is always the index
        of the refence point.
    vert_to : ndarray, shape (n_pairs,)
        For each pair, the index of the second vertex.

    See Also
    --------
    all_to_all_connectivity_pairs : Obtain pairs for all-to-all connectivity.
    """
    # Get coordinates of the vertices
    if isinstance(src_or_fwd, SourceSpaces):
        vertno_lh = src_or_fwd[0]["vertno"]
        vertno_rh = src_or_fwd[1]["vertno"]
        grid_points = np.vstack(
            (src_or_fwd[0]["rr"][vertno_lh], src_or_fwd[1]["rr"][vertno_rh])
        )
    elif isinstance(src_or_fwd, Forward):
        grid_points = src_or_fwd["source_rr"]
    else:
        raise ValueError(
            "Source must be instance of Forward or a list", "of SourceSpaces"
        )

    # Select the pairs that are further away than the distance limit
    dist = cdist(grid_points[ref_point][np.newaxis], grid_points)
    vert_to = np.flatnonzero(dist >= min_dist)

    n_pairs = len(vert_to)
    vert_from = np.asarray(ref_point).repeat(n_pairs)

    return vert_from, vert_to


try:
    import numba as nb

    @nb.jit(nb.complex128[:, :, :](nb.complex128[:, :, :], nb.complex128[:, :]))
    def _compute_opt1(x, y):
        r = np.zeros((x.shape[0], y.shape[1], y.shape[1]), dtype=nb.complex128)
        for i in range(len(x)):
            r[i, :, :] = np.dot(np.dot(y.T, x[i, :, :]), y)
        return r

    @nb.jit(
        nb.complex128[:, :, :](
            nb.complex128[:, :, :], nb.complex128[:, :, :], nb.int64[:], nb.int64[:]
        )
    )
    def _compute_power_cross_inv(x, y, x_ind, y_ind):
        r = np.zeros((x_ind.shape[0], x.shape[0], y.shape[2]), dtype=nb.complex128)
        i = 0
        for x_i, y_i in zip(x_ind, y_ind):
            r[i, :, :] = np.dot(x[:, x_i, :], y[:, y_i, :])
            i += 1
        return r

    @nb.jit(nb.complex128[:, :, :](nb.complex128[:, :, :], nb.complex128[:, :, :]))
    def _compute_power_cross_inv2(x, y):
        r = np.zeros((x.shape[1], x.shape[0], y.shape[2]), dtype=nb.complex128)
        for i in range(x.shape[1]):
            r[i, :, :] = np.dot(x[:, i, :], y[:, i, :])
        return r

    numba_enabled = True
except Exception:
    numba_enabled = False


def _compute_dics_coherence(
    W,
    G,
    vert_ind_from,
    vert_ind_to,
    spec_power_inv,
    orientations,
    coh_metric="absolute",
):
    """Compute the coherence between two sources using a DICS beamformer.

    Computes the coherence between two dipoles for different angles and returns
    the maximum value.

    Parameters
    ----------
    W : ndarray, shape (n_orient, n_sources, n_sensors)
        The beamformer filter weights.
    G : ndarray, shape (n_sensors, n_sources, n_orient)
        The leadfield.
    vert_ind_from : ndarray, shape (n_pairs,)
        For each vertex-pair to compute the connectivity for, the index of the
        first vertex.
    vert_ind_to : ndarray, shape (n_pairs,)
        For each vertex-pair to compute the connectivity for, the index of the
        second vertex.
    spec_power_inv : ndarray, shape (n_sources, n_orient, n_orient)
        Inverse of cross-spectral power between the dipoles at each source
        location.
    orientations : ndarray, shape (n_orient, n_angles)
        For each angle to try, a unit vector pointing in the direction of the
        angle.
    coh_metric : 'absolute' | 'imaginary'
        The coherence metric to use. Either the square of absolute coherence
        ('absolute') or the square of the imaginary part of the coherence
        ('imaginary'). Defaults to 'absolute'.

    Returns
    -------
    coherence : ndarray, shape (n_pairs,)
        For each vertex-pair, the coherence in the direction of maximum
        coherence.
    """
    power_from_inv = spec_power_inv[vert_ind_from]
    power_to_inv = spec_power_inv[vert_ind_to]

    if numba_enabled:
        power_cross_inv = _compute_power_cross_inv(
            W, G.astype("complex"), vert_ind_from, vert_ind_to
        )

        opt1 = _compute_opt1(power_cross_inv, orientations.astype("complex"))
    else:
        # Computes W @ G
        power_cross_inv = np.einsum(
            "ijk,kjl->jil", W[:, vert_ind_from, :], G[:, vert_ind_to, :]
        )

        # Computes orientations.T @ power_cross_inv @ orientations
        opt1 = power_cross_inv.dot(orientations)
        opt1 = opt1.transpose(0, 2, 1).dot(orientations).transpose(0, 2, 1)

    if coh_metric == "absolute":
        opt1 = np.abs(opt1)
    elif coh_metric == "imaginary":
        opt1 = np.imag(opt1)

    # Computes np.diag(orientations.T @ power_from_inv @ orientations)
    opt2 = np.sum(orientations * power_from_inv.dot(orientations), axis=1)

    # Computes np.diag(orientations.T @ power_to_inv @ orientations)
    opt3 = np.sum(orientations * power_to_inv.dot(orientations), axis=1)

    # Compute coherence for each orientation
    opt = (opt1**2) / (opt2[:, :, np.newaxis] * opt3[:, np.newaxis, :])

    # Pick the best orientation as the final coherence value
    return np.real(np.max(opt, axis=(1, 2)))


@verbose
def dics_connectivity(
    vertex_pairs,
    fwd,
    data_csd,
    reg=0.05,
    coh_metric="absolute",
    n_angles=50,
    block_size=10000,
    n_jobs=1,
    verbose=None,
):
    """Compute spectral connectivity using a DICS beamformer.

    Calculates the connectivity between the given vertex pairs using a DICS
    beamformer [1]_ [2]_. Connectivity is defined in terms of coherence:

    C = Sxy^2 [Sxx * Syy]^-1

    Where Sxy is the cross-spectral density (CSD) between dipoles x and y, Sxx
    is the power spectral density (PSD) at dipole x and Syy is the PSD at
    dipole y.

    Parameters
    ----------
    vertex_pairs : pair of lists (vert_from_idx, vert_to_idx)
        Vertex pairs between which connectivity is calculated. The pairs are
        specified using two lists: the first list contains, for each pair, the
        index of the first vertex. The second list contains, for each pair, the
        index of the second vertex.
    fwd : instance of Forward
        Subject's forward solution, possibly restricted to only include
        vertices that are close to the sensors. For 'canonical' mode, the
        orientation needs to be tangential or free.
    data_csd : instance of CrossSpectralDensity
        The cross spectral density of the data.
    reg : float
        Tikhonov regularization parameter to control for trade-off between
        spatial resolution and noise sensitivity. Defaults to 0.05.
    coh_metric : 'absolute' | 'imaginary'
        The coherence metric to use. Either the square of absolute coherence
        ('absolute') or the square of the imaginary part of the coherence
        ('imaginary'). Defaults to 'absolute'.
    n_angles : int
        Number of angles to try when optimizing dipole orientations. Defaults
        to 50.
    block_size : int
        Number of pairs to process in a single batch. Beware of memory
        requirements, which are ``n_jobs * block_size``. Defaults to 10000.
    n_jobs : int
        Number of blocks to process simultaneously. Defaults to 1.
    verbose : bool | str | int | None
        If not None, override default verbose level (see :func:`mne.verbose`
        and :ref:`Logging documentation <tut_logging>` for more).

    Returns
    -------
    connectivity : instance of Connectivity
        The adjacency matrix.

    See Also
    --------
    all_to_all_connectivity_pairs : Obtain pairs for all-to-all connectivity.
    one_to_all_connectivity_pairs : Obtain pairs for one-to-all connectivity.

    References
    ----------
    .. [1] Gross, J., Kujala, J., Hamalainen, M., Timmermann, L., Schnitzler,
           A., & Salmelin, R. (2001). Dynamic imaging of coherent sources:
           Studying neural interactions in the human brain. Proceedings of the
           National Academy of Sciences, 98(2), 694–699.
    .. [2] Kujala, J., Gross, J., & Salmelin, R. (2008). Localization of
           correlated network activity at the cortical level with MEG.
           NeuroImage, 39(4), 1706–1720.
    """
    fwd = pick_channels_forward(fwd, data_csd.ch_names)
    data_csd = pick_channels_csd(data_csd, fwd["info"]["ch_names"])

    vertex_from, vertex_to = vertex_pairs
    if len(vertex_from) != len(vertex_to):
        raise ValueError("Lengths of the two lists of vertices do not match.")
    n_pairs = len(vertex_from)

    G = fwd["sol"]["data"].copy()
    n_orient = G.shape[1] // fwd["nsource"]

    if n_orient == 1:
        raise ValueError(
            "A forward operator with free or tangential " "orientation must be used."
        )
    elif n_orient == 3:
        # Convert forward to tangential orientation for more speed.
        fwd = forward_to_tangential(fwd)
        G = fwd["sol"]["data"]
        n_orient = 2

    G = G.reshape(G.shape[0], fwd["nsource"], n_orient)

    # Normalize the lead field
    G /= np.linalg.norm(G, axis=0)

    Cm = data_csd.get_data()
    Cm_inv, alpha, _ = reg_pinv(Cm, reg)
    del Cm

    W = np.dot(G.T, Cm_inv)

    # Pre-compute spectral power at each unique vertex
    unique_verts, vertex_map = np.unique(
        np.r_[vertex_from, vertex_to], return_inverse=True
    )
    spec_power_inv = np.array(
        [np.dot(W[:, vert, :], G[:, vert, :]) for vert in unique_verts]
    )

    # Map vertex indices to unique indices, so the pre-computed spectral power
    # can be retrieved
    vertex_from_map = vertex_map[: len(vertex_from)]
    vertex_to_map = vertex_map[len(vertex_from) :]

    coherence = np.zeros((len(vertex_from)))

    # Define a search space for dipole orientations
    angles = np.arange(n_angles) * np.pi / n_angles
    orientations = np.vstack((np.sin(angles), np.cos(angles)))

    # Create chunks of pairs to evaluate at once
    n_blocks = int(np.ceil(n_pairs / float(block_size)))
    blocks = [
        slice(i * block_size, min((i + 1) * block_size, n_pairs))
        for i in range(n_blocks)
    ]

    parallel, my_compute_dics_coherence, _ = parallel_func(
        _compute_dics_coherence, n_jobs, verbose
    )

    logger.info(
        "Computing coherence between %d source pairs in %d blocks..."
        % (n_pairs, n_blocks)
    )
    if numba_enabled:
        logger.info("Using numba optimized code path.")
    coherence = np.hstack(
        parallel(
            my_compute_dics_coherence(
                W,
                G,
                vertex_from_map[block],
                vertex_to_map[block],
                spec_power_inv,
                orientations,
                coh_metric,
            )
            for block in blocks
        )
    )
    logger.info("[done]")

    return VertexConnectivity(
        data=coherence,
        pairs=[v[: len(coherence)] for v in vertex_pairs],
        vertices=[s["vertno"] for s in fwd["src"]],
        vertex_degree=None,  # Compute this in the constructor
        subject=fwd["src"][0]["subject_his_id"],
    )
