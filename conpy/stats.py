# encoding: utf-8
"""
Statistics for Connectivity objects.

Authors: Susanna Aro <susanna.aro@aalto.fi>
         Marijn van Vliet <w.m.vanvliet@gmail.com>
"""
from warnings import warn

import numpy as np
from scipy import stats
from scipy.cluster.hierarchy import fclusterdata
from scipy.stats import ttest_1samp
from mne.parallel import parallel_func
from mne.utils import logger, verbose, ProgressBar

# from .connectivity import VertexConnectivity
from .connectivity import VertexConnectivity


def group_connectivity_ttest(cond1, cond2, df=None, tail=None):
    """Paired t-test comparing connectivity between two conditions.

    Parameters
    ----------
    cond1 : list of VertexConnectivity | LabelConnectivity
        For each subject, the connectivity object corresponding to the first
        experimental condition.
    cond2 : list of VertexConnectivity | LabelConnectivity
        For each subject, the connectivity object corresponding to the second
        experimental condition.
    df : int | None
        Degrees of freedom. If ``None``, defaults to ``n_subjects - 1``.
    tail: 'right' | 'left' | None
        Which tailed t-test to use. If ``None``, two-tailed t-test is
        performed.

    Returns
    -------
    t : ndarray, shape (n_pairs,)
        t-values for all connections.
    pval : ndarray, shape (n_pairs,)
        p-values for all connections.
    """
    if len(cond1) != len(cond2):
        raise ValueError('The number of subjects in each condition must be '
                         'the same.')
    n_subjects = len(cond1)

    # Check compatibility of the connection objects
    pairs1 = cond1[0].pairs
    for con in cond1[1:] + cond2:
        if not np.array_equal(pairs1, con.pairs):
            raise ValueError('Not all Connectivity objects have the same '
                             'connection pairs defined.')

    # Perform a paired t-test
    X1 = np.array([con.data for con in cond1])
    X2 = np.array([con.data for con in cond2])
    t, pval = stats.ttest_rel(X1, X2)

    if not df:
        df = n_subjects - 1

    if tail is not None:
        # Scipy gives out only two-tailed p-values
        if tail == 'right':
            pval = stats.t.cdf(-t, df)
        elif tail == 'left':
            pval = stats.t.cdf(t, df)
        else:
            raise ValueError('Tail must be "right", "left" or None.')
    return t, pval


@verbose
def cluster_threshold(con, src, min_size=20, max_spread=0.013,
                      method='single', verbose=None):
    """Threshold connectivity using clustering.

    First, connections are grouped into "bundles". A bundle is a group of
    connections which start and end points are close together. Then, only
    bundles with a sufficient amount of connections are retained.

    Parameters
    ----------
    con : instance of Connectivity
        Connectivity to threshold.
    src : instance of SourceSpace
        The source space for which the connectivity is defined.
    min_size : int
        Minimum amount of connections that a bundle must contain in order to be
        accepted.
    max_spread : float
        Maximum amount the position (in metres) of the start and end points
        of the connections may vary in order for them to be considered part of
        the same "bundle". Defaults to 0.013.
    method : str
        Linkage method for fclusterdata. Defaults to 'single'. See
        documentation for ``scipy.cluster.hierarchy.fclusterdata`` for for more
        information.
    verbose : bool | str | int | None
        If not ``None``, override default verbose level
        (see :func:`mne.verbose` and :ref:`Logging documentation <tut_logging>`
        for more).

    Returns
    -------
    thresholded_connectivity : instance of Connectivity
        Instance of connectivity with the thresholded data.
    """
    grid_points = np.vstack([s['rr'][v] for s, v in zip(src, con.vertices)])
    X = np.hstack([grid_points[inds] for inds in con.pairs])
    clust_no = fclusterdata(X, max_spread, criterion='distance', method=method)

    # Remove clusters that do not pass the threshold
    clusters, counts = np.unique(clust_no, return_counts=True)
    big_clusters = clusters[counts >= min_size]
    logger.info('Found %d bundles, of which %d are of sufficient size.' %
                (len(clusters), len(big_clusters)))

    # Restrict the connections to only those found in the big bundles
    mask = np.in1d(clust_no, big_clusters)
    data = con.data[mask]
    pairs = [p[mask] for p in con.pairs]

    return VertexConnectivity(
        data=data,
        pairs=pairs,
        vertices=con.vertices,
        vertex_degree=con.source_degree,
        subject=con.subject
    )


@verbose
def cluster_permutation_test(cond1, cond2, cluster_threshold, src, alpha=0.05,
                             tail=0, n_permutations=1024, max_spread=0.013,
                             cluster_method='single', seed=None,
                             return_details=False, n_jobs=1, verbose=None):
    """Find significant bundles of connections using a permutation test.

    This is a variation on the cluster permutation test described in [1]_.

    First, connections are thresholded using a paired 2-sided t-test. Any
    connections that survive the threshold are grouped into "bundles". A bundle
    is a group of connections which start and end points are close together.
    Then, for each bundle, the sum of the t-values of its connections is
    evaluated against those obtained from using the same procedure on random
    permutations of the data. For further information, see [2]_.

    Parameters
    ----------
    cond1 : list of VertexConnectivity
        For each subject, the connectivity object corresponding to the first
        experimental condition. Each connectivity object should define the same
        connections.
    cond2 : list of VertexConnectivity
        For each subject, the connectivity object corresponding to the second
        experimental condition. Each connectivity object should define the same
        connections.
    cluster_threshold : float
        The threshold to use for forming the intial bundles. Only connections
        with a t-value that is either higher than ``cluster_threshold`` or
        lower than ``-cluster_threshold`` are kept.
    tail : -1 | 0 | 1
        Which "tail" of the distribution of the test statistic to use:

            -1: the hypothesis is that cond1 < cond2.
             0: the hypothesis is that cond1 != cond2.
             1: the hypothesis is that cond1 > cond2.

        Defaults to 0, meaning a two-tailed test.
    src : instance of SourceSpace
        The source space for which the connectivity is defined.
    alpha : float
        The p-value to use for null-hypothesis testing. Using random
        permutations, the distribution of t-values is estimated. Bundles with a
        t-value in the requested percentile will be deemed significant.
        Defaults to 0.05.
    n_permutations : int
        The number of random permutations to use to estimate the distribution
        of t-values. Defaults to 1024.
    max_spread : float
        Maximum amount the position (in metres) of the start and end points
        of the connections may vary in order for them to be considered part of
        the same "bundle". Defaults to 0.013.
    cluster_method : str
        Linkage method for fclusterdata. Defaults to 'single'. See
        documentation for ``scipy.cluster.hierarchy.fclusterdata`` for for more
        information.
    seed : int | None
        The seed to use for the random number generator. Use this to reproduce
        a specific result. Defaults to ``None`` so a different seed is used
        every time.
    return_details : bool
        Whether to return details about the bundles and the permulation stats.
        Defaults to False.
    n_jobs : int
        Number of jobs to run in parallel. Note that a copy of ``cond1`` and
        ``cond2`` will be made for each job in memory. Defaults to 1.
    verbose : bool | str | int | None
        If not ``None``, override default verbose level
        (see :func:`mne.verbose` and :ref:`Logging documentation <tut_logging>`
        for more).

    Returns
    -------
    connection_indices : ndarray, shape (n_connections,)
        Indices of the connections that are part of a significant bundle.
    bundles : list of list of int (optional)
        For each found bundle, the indices of the connections that are part of
        the bundle. Only returned when ``return_details=True`` is specified.
    bundle_ts : ndarray, shape (n_bundles,) (optional)
        For each found bundle, the sum of the t-values for all connections that
        are part of the bundle. These are the t-values that were used to
        determine the initial threshold for the connections. They are not
        indicative of the null-hypothesis.
        Only returned when ``return_details=True`` is specified.
    bundle_ps : ndarray, shape (n_bundles,) (optional)
        For each found bundle, the p-value based on the permutation test,
        indicative for the likelyhood that the null-hypothesis holds.
        Only returned when ``return_details=True`` is specified.
    H0 : ndarray, shape (n_permutations,) (optional)
        The maximum observed t-value during each random permutation.
        Only returned when ``return_details=True`` is specified.

    References
    ----------
    .. [1] Maris/Oostenveld (2007), "Nonparametric statistical testing of
           EEG- and MEG-data" Journal of Neuroscience Methods,
           Vol. 164, No. 1., pp. 177-190. doi:10.1016/j.jneumeth.2007.03.024.
    .. [2] van Vliet, M., Liljeström, M., Aro, S., Salmelin, R., & Kujala, J.
           (2018). Analysis of functional connectivity and oscillatory power
           using DICS: from raw MEG data to group-level statistics in Python.
           bioRxiv, 245530, 1-25. https://doi.org/10.1101/245530
    """
    if len(cond1) != len(cond2):
        raise ValueError('The number of subjects in each condition must be '
                         'the same.')
    n_subjects = len(cond1)

    # Check compatibility of the connection objects
    for con in cond1 + cond2:
        if not isinstance(con, VertexConnectivity):
            raise ValueError('All connectivity objects must by of type '
                             'VertexConnectivity.')

        if not np.array_equal(con.pairs, cond1[0].pairs):
            raise ValueError('Not all Connectivity objects have the same '
                             'connection pairs defined.')

    if tail not in [-1, 0, 1]:
        raise ValueError('The `tail` parameter should be -1, 0, or 1, but the '
                         'value {} was supplied'.format(tail))

    # Create pairwise contrast. We'll do a t-test against the null hypothesis
    # that the mean of this contrast is zero. This is equivalent to a paired
    # t-test.
    Xs = np.array([con1.data - con2.data for con1, con2 in zip(cond1, cond2)])

    # Get the XYZ coordinates for the vertices between which the connectivity
    # is defined.
    grid_points = np.vstack([s['rr'][v]
                             for s, v in zip(src, cond1[0].vertices)])
    grid_points = np.hstack([grid_points[inds] for inds in cond1[0].pairs])

    logger.info('Forming initial bundles of connectivity...')
    _, bundles, bundle_ts = _do_single_permutation(
        Xs, cluster_threshold, tail, grid_points, max_spread, cluster_method,
        return_bundles=True
    )
    if len(bundle_ts) == 0:
        warn('No clusters found, returning empty connectivity_indices')
        if return_details:
            return [], [], [], []
        else:
            return []
    else:
        logger.info('Retained %d connections, grouped into %d bundles.' %
                    (np.sum([len(b) for b in bundles]), len(bundles)))

    parallel, my_perm_func, _ = parallel_func(_do_single_permutation, n_jobs,
                                              verbose=verbose)

    logger.info('Permuting %d times...' % n_permutations)
    rng = np.random.RandomState(seed)

    # We are doing random permutations using sign flips
    sign_flips = [rng.choice([-1, 1], size=n_subjects)[:, np.newaxis]
                  for _ in range(n_permutations)]

    def permutations():
        """Generator for the permutations with optional progress bar."""
        if verbose:
            progress = ProgressBar(len(sign_flips),
                                   mesg='Performing permutations')
            for i, sign_flip in enumerate(sign_flips):
                progress.update(i)
                yield sign_flip
        else:
            for sign_flip in sign_flips:
                yield sign_flip

    # Compute the random permutation stats
    perm_stats = parallel(
        my_perm_func(Xs * sign_flip, cluster_threshold, tail, grid_points,
                     max_spread, cluster_method)
        for sign_flip in permutations()
    )
    H0 = np.concatenate(perm_stats)

    # Compute p-values for each initial bundle
    bundle_ps = np.array([np.mean(np.abs(H0) >= abs(t)) for t in bundle_ts])

    # All connections that are part of a significant bundle
    significant_bundles = [b for b, p in zip(bundles, bundle_ps) if p <= alpha]
    if len(significant_bundles) > 0:
        connection_indices = np.unique(np.concatenate(significant_bundles))
    else:
        connection_indices = []

    logger.info('Found %d bundles with significant p-values, containing in '
                'total %d connections.' %
                (len(significant_bundles), len(connection_indices)))

    if return_details:
        return connection_indices, bundles, bundle_ts, bundle_ps, H0
    else:
        return connection_indices


def _do_single_permutation(Xs, cluster_threshold, tail, grid_points,
                           max_spread, cluster_method, return_bundles=False):
    """Perform a single clustering permutation.

    Parameters
    ----------
    Xs : ndarray, shape(n_subjects, n_connections)
        The connectivity data: a constrast between two conditions.
    cluster_threshold : float
        The initial t-value threshold to prune connections.
    tail : -1 | 0 | 1
        The tail of the distribution to use.
    grid_points : ndarray, shape (n_vertices, 3)
        The XYZ coordinates for each vertex between which connectivity is
        defined.
    max_spread : float
        Maximum amount the position (in metres) of the start and end points
        of the connections may vary in order for them to be considered part of
        the same "bundle". Defaults to 0.013.
    cluster_method : str
        Linkage method for fclusterdata. Defaults to 'single'. See
        documentation for ``scipy.cluster.hierarchy.fclusterdata`` for for more
        information.
    return_bundles : bool
        Whether to return detailed bundle information. Defaults to False.

    Returns
    -------
    max_stats : pair of float
        The minimum negative t-value and maximum positive t-value found. These
        are used to construct the 'left' and 'right' tails of the t-value
        distribution.
    all_bundles : list of list of int (optional)
        For each of the found bundle, the indices of the connections belonging
        to the bundle. These are all bundles, not only the significant ones.
        Only returned when ``return_bundles == True``
    all_bundle_ts : list of float (optional)
        For each bundle, the sum of the t-values of the connections within the
        bundle.
        Only returned when ``return_bundles == True``
    """
    t, _ = ttest_1samp(Xs, 0, axis=0)
    perm_stats = []

    if return_bundles:
        all_bundles = []
        all_bundle_ts = []

    # Threshold the data, according to the desired tail
    if tail == -1:
        masks = [t < -cluster_threshold]
    elif tail == 0:
        masks = [t < -cluster_threshold, t >= cluster_threshold]
    elif tail == 1:
        masks = [t >= cluster_threshold]
    else:
        raise ValueError('Invalid value for the `tail` parameter.')

    for mask in masks:
        n_connections = mask.sum()
        if n_connections == 0:
            clust_no = []
            bundle_ts = []
        elif n_connections == 1:
            # Only one connection survived, don't attempt any clustering
            clust_no = [0]
            bundle_ts = t[mask]
        else:
            # Cluster the connections into bundles
            clust_no = fclusterdata(grid_points[mask], max_spread,
                                    criterion='distance',
                                    method=cluster_method)
            # Cluster numbers start at 1, which trips up np.bincount()
            clust_no -= 1
            bundle_ts = np.bincount(clust_no, t[mask])

        if len(bundle_ts) > 0:
            perm_stats.append(np.max(bundle_ts))
        else:
            perm_stats.append(0)

        if return_bundles:
            all_bundles.extend(
                _cluster_assignment_to_list_of_lists(np.flatnonzero(mask),
                                                     clust_no)
            )
            all_bundle_ts.append(bundle_ts)

    if return_bundles:
        all_bundle_ts = np.hstack(all_bundle_ts)
        return perm_stats, all_bundles, all_bundle_ts
    else:
        return perm_stats


def _cluster_assignment_to_list_of_lists(elements, assignment):
    """Convert a list of cluster assignments to a list of list of clusters.

    Parameters
    ----------
    elements : list
        The elements that were assigned to clusters.
    assignment : list of int
        For each element, the cluster it belongs to.

    Returns
    -------
    clusters : list of lists
        For each cluster, the elements that belong to the cluster.

    Examples
    --------
    >>> _cluster_assignment_to_list_of_lists(['a', 'b', 'c', 'd', 'e'],
    ...                                      [0, 4, 1, 4, 3])
    [['a', 'e'], ['b', 'd'], ['d']]
    """
    elements = np.asarray(elements)
    assignment = np.asarray(assignment)
    order = np.argsort(assignment)
    assignment_sort = assignment[order]
    split_points = np.flatnonzero(assignment_sort[1:] - assignment_sort[:-1])
    cluster_inds = np.split(order, split_points + 1)
    clusters = [elements[i].tolist() for i in cluster_inds]
    return clusters
