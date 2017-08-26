"""Recursive nearest agglomeration (ReNA):
    fastclustering for approximation of structured signals

Author:
    Andres Hoyos idrobo, Gael Varoquaux, Jonas Kahn and  Bertrand Thirion
"""
import numpy as np
from scipy.sparse import csgraph, coo_matrix, dia_matrix


def _compute_weights(data):
    """Measuring the Euclidean distance: computer the weights in the direction
    of each axis

    Note: Here we are assuming a square lattice (no diagonal connections)
    """
    dims = len(data.shape)
    weights = []
    for axis in range(dims):
        weights.append(np.sum(np.diff(data, axis=axis) ** 2, axis=-1).ravel())

    return np.hstack(weights)


def _compute_edges(data, is_mask=False):
    """
    """
    dims = len(data.shape)
    edges = []
    for axis in range(dims):
        vertices_axis = np.swapaxes(data, 0, axis)

        if is_mask:
            edges.append(np.logical_and(
                vertices_axis[:-1].swapaxes(axis, 0).ravel(),
                vertices_axis[1:].swapaxes(axis, 0).ravel()))
        else:
            edges.append(np.vstack(
                [vertices_axis[:-1].swapaxes(axis, 0).ravel(),
                 vertices_axis[1:].swapaxes(axis, 0).ravel()]))

    return np.hstack(edges)


def _create_ordered_edges(data, mask):
    """
    """
    shape = mask.shape
    n_features = np.prod(shape)

    vertices = np.arange(n_features).reshape(shape)
    weights = _compute_weights(data)
    edges = _compute_edges(vertices, is_mask=False)
    edges_mask = _compute_edges(mask, is_mask=True)

    # Apply the mask
    weights = weights[edges_mask]
    edges = edges[:, edges_mask]

    # Reorder the indices of the graph
    max_index = edges.max()
    order = np.searchsorted(np.unique(edges.ravel()), np.arange(max_index + 1))
    edges = order[edges]

    return edges, weights, edges_mask


def weighted_connectivity_graph(data, n_features, mask):
    """ Creating weighted graph
    data and topology, encoded by a connectivity matrix
    """
    edges, weight, edges_mask = _create_ordered_edges(data, mask=mask)
    connectivity = coo_matrix(
        (weight, edges), (n_features, n_features)).tocsr()
    # Making it symmetrical
    connectivity = (connectivity + connectivity.T) / 2

    return connectivity


def _nn_connectivity(connectivity, thr):
    """ Fast implementation of nearest neighbor connectivity

    connectivity: weighted connectivity matrix
    """
    n_features = connectivity.shape[0]

    connectivity_ = coo_matrix(
        (1. / connectivity.data, connectivity.nonzero()),
        (n_features, n_features)).tocsr()

    inv_max = dia_matrix((1. / connectivity_.max(axis=0).toarray()[0], 0),
                         shape=(n_features, n_features))

    connectivity_ = inv_max * connectivity_

    # Dealing with eccentricities
    edge_mask = connectivity_.data > 1 - thr

    j_idx = connectivity_.nonzero()[1][edge_mask]
    i_idx = connectivity_.nonzero()[0][edge_mask]

    weight = np.ones_like(j_idx)
    edges = np.array((i_idx, j_idx))

    nn_connectivity = coo_matrix((weight, edges), (n_features, n_features))

    return nn_connectivity


def reduce_data_and_connectivity(labels, n_labels, connectivity, data, thr):
    """
    """
    n_features = len(labels)

    incidence = coo_matrix(
        (np.ones(n_features), (labels, np.arange(n_features))),
        shape=(n_labels, n_features), dtype=np.float32).tocsc()

    inv_sum_col = dia_matrix(
        (np.array(1. / incidence.sum(axis=1)).squeeze(), 0),
        shape=(n_labels, n_labels))

    incidence = inv_sum_col * incidence

    # reduced data
    reduced_data = (incidence * data.T).T
    reduced_connectivity = (incidence * connectivity) * incidence.T

    reduced_connectivity = reduced_connectivity - dia_matrix(
        (reduced_connectivity.diagonal(), 0), shape=(reduced_connectivity.shape))

    i_idx, j_idx = reduced_connectivity.nonzero()

    data = np.maximum(
        thr, np.sum((reduced_data[:, i_idx] - reduced_data[:, j_idx]) ** 2, 0))
    reduced_connectivity.data = data

    return reduced_connectivity, reduced_data


def nearest_neighbor_grouping(connectivity, data_matrix, n_clusters, thr):
    """ Cluster according to nn and reduce the data and connectivity
    """
    # Nearest neighbor conenctivity
    nn_connectivity = _nn_connectivity(connectivity, thr)

    n_features = connectivity.shape[0]

    n_labels = n_features - (nn_connectivity + nn_connectivity.T).nnz / 2

    if n_labels < n_clusters:
        # cut some links to achieve the desired number of clusters
        alpha = n_features - n_clusters

        nn_connectivity = nn_connectivity + nn_connectivity.T

        edges_ = np.array(nn_connectivity.nonzero())

        plop = edges_[0] - edges_[1]

        select = np.argsort(plop)[:alpha]

        nn_connectivity = coo_matrix(
            (np.ones(2 * alpha),
             np.hstack((edges_[:, select], edges_[::-1, select]))),
            (n_features, n_features))

    # Clustering step: getting the connected components of the nn matrix
    n_labels, labels = csgraph.connected_components(nn_connectivity)

    # Reduction step: reduction by averaging
    reduced_connectivity, reduced_data_matrix = reduce_data_and_connectivity(
        labels, n_labels, connectivity, data_matrix, thr)

    return reduced_connectivity, reduced_data_matrix, labels


def recursive_nearest_agglomeration(data_matrix, connectivity, n_clusters,
                                    n_iter=10, thr=1e-7):
    """
    """
    # Initialization
    labels = np.arange(connectivity.shape[0])
    n_labels = connectivity.shape[0]

    for i in range(n_iter):
        connectivity, data_matrix, reduced_labels = nearest_neighbor_grouping(
            connectivity, data_matrix, n_clusters, thr)

        labels = reduced_labels[labels]
        n_labels = connectivity.shape[0]

        if n_labels <= n_clusters:
            break

    return labels


def reduce_data(X, labels):
    unique_labels = np.unique(labels)
    nX = []
    for l in unique_labels:
        nX.append(np.mean(X[:, labels == l], axis=1))
    return np.array(nX).T


def approximate_data(X, labels):
    _, inverse = np.unique(labels, return_inverse=True)
    return X[..., inverse]

