# -*- coding: utf-8 -*-
"""
Plotting functions.

Authors: Susanna Aro <susanna.aro@aalto.fi>
         Marijn van Vliet <w.m.vanvliet@gmail.com>
"""
import numpy as np

from mne.viz.circle import circular_layout
from mne_connectivity.viz import plot_connectivity_circle


def plot_connectivity(con, n_lines=None, node_angles=None, node_width=None,
                      node_colors=None, facecolor='black', textcolor='white',
                      node_edgecolor='black', linewidth=1.5, colormap='hot',
                      vmin=None, vmax=None, colorbar=True, title=None,
                      colorbar_size=0.2, colorbar_pos=(-0.3, 0.1),
                      fontsize_title=12, fontsize_names=8, fontsize_colorbar=8,
                      padding=6., fig=None, subplot=111, interactive=True,
                      node_linewidth=2., show=True):
    """Visualize parcellated connectivity as a circular graph.

    Parameters
    ----------
    con : instance of LabelConnectivity
        The parcellated connectivity to visualize.
    n_lines : int | None
        If not None, only the n_lines strongest connections
        (strength=abs(con)) are drawn.
    node_angles : array, shape=(len(node_names,)) | None
        Array with node positions in degrees. If None, the nodes are
        equally spaced on the circle. See :func:`mne.viz.circular_layout`.
    node_width : float | None
        Width of each node in degrees. If None, the minimum angle between
        any two nodes is used as the width.
    node_colors : list of tuples | list of str
        List with the color to use for each node. If fewer colors than
        nodes are provided, the colors will be repeated. Any color
        supported by matplotlib can be used, e.g., RGBA tuples, named
        colors.
    facecolor : str
        Color to use for background. See matplotlib.colors.
    textcolor : str
        Color to use for text. See matplotlib.colors.
    node_edgecolor : str
        Color to use for lines around nodes. See matplotlib.colors.
    linewidth : float
        Line width to use for connections.
    colormap : str
        Colormap to use for coloring the connections.
    vmin : float | None
        Minimum value for colormap. If None, it is determined
        automatically.
    vmax : float | None
        Maximum value for colormap. If None, it is determined
        automatically.
    colorbar : bool
        Display a colorbar or not.
    title : str
        The figure title.
    colorbar_size : float
        Size of the colorbar.
    colorbar_pos : 2-tuple
        Position of the colorbar.
    fontsize_title : int
        Font size to use for title.
    fontsize_names : int
        Font size to use for node names.
    fontsize_colorbar : int
        Font size to use for colorbar.
    padding : float
        Space to add around figure to accommodate long labels.
    fig : None | instance of matplotlib.pyplot.Figure
        The figure to use. If None, a new figure with the specified
        background color will be created.
    subplot : int | 3-tuple
        Location of the subplot when creating figures with multiple plots.
        E.g.  121 or (1, 2, 1) for 1 row, 2 columns, plot 1. See
        matplotlib.pyplot.subplot.
    interactive : bool
        When enabled, left-click on a node to show only connections to that
        node. Right-click shows all connections.
    node_linewidth : float
        Line with for nodes.
    show : bool
        Show figure if True.

    Returns
    -------
    fig : instance of matplotlib.pyplot.Figure
        The figure handle.
    axes : instance of matplotlib.axes.PolarAxesSubplot
        The subplot handle.
    """
    names = [l.name for l in con.labels]

    if node_colors is None:
        node_colors = [l.color for l in con.labels]

    if node_angles is None:
        # Try to construct a sensible default layout

        # First, we reorder the labels based on their location in the left
        # hemisphere.
        lh_labels = [name for name in names if name.endswith('lh')]

        # Get the y-location of the label
        label_ypos = list()
        for name in lh_labels:
            idx = names.index(name)
            ypos = np.mean(con.labels[idx].pos[:, 1])
            label_ypos.append(ypos)

        # Reorder the labels based on their location
        lh_labels = [label
                     for (yp, label) in sorted(zip(label_ypos, lh_labels))]

        # Second, we reorder the labels based on their location in the right
        # hemisphere.
        rh_labels = [name for name in names if name.endswith('rh')]
        # For the right hemi
        # Get the y-location of the label
        rlabel_ypos = list()
        for name in rh_labels:
            idx = names.index(name)
            ypos = np.mean(con.labels[idx].pos[:, 1])
            rlabel_ypos.append(ypos)

        # Reorder the labels based on their location
        rh_labels = [label
                     for (yp, label) in sorted(zip(rlabel_ypos, rh_labels), reverse=True)]

        # Save the plot order and create a circular layout
        node_order = list()
        node_order.extend(lh_labels[::-1])  # reverse the order
        node_order.extend(rh_labels[::-1])  # reverse the order

        node_angles = circular_layout(names, node_order, start_pos=90,
                                      group_boundaries=[0, len(names) / 2])

    return plot_connectivity_circle(
        con.data, node_names=names, indices=con.pairs,
        n_lines=n_lines, node_angles=node_angles, node_width=node_width,
        node_colors=node_colors, facecolor=facecolor, textcolor=textcolor,
        node_edgecolor=node_edgecolor, linewidth=linewidth,
        colormap=colormap, vmin=vmin, vmax=vmax, colorbar=colorbar,
        title=title, colorbar_size=colorbar_size,
        colorbar_pos=colorbar_pos, fontsize_title=fontsize_title,
        fontsize_names=fontsize_names, fontsize_colorbar=fontsize_colorbar,
        padding=padding, fig=fig, interactive=interactive,
        node_linewidth=node_linewidth, show=show
    )
