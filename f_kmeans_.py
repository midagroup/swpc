# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 15:34:24 2015

@author: Federico Benvenuto & Annalisa Perasso
"""

import warnings

import numpy as np

from scipy.spatial.distance import cdist
from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted
from sklearn.cluster.k_means_ import _init_centroids
from sklearn.utils.extmath import row_norms


def _distance_data_centers(X, centers, distance):
    # Calculate the distances between X (data) and centers
    n_samples = X.shape[0]
    d = np.zeros(shape=(n_samples,), dtype=np.float64)
    d = cdist(X, centers, distance)
    d = np.fmax(d, np.finfo(np.float64).eps)

    return d


def _labels_computation(u):
    # Labels computation
    labels = np.argmax(u, axis=1)

    return labels


def _fp_coeff(u):
    """
    Fuzzy partition coefficient `fpc` relative to fuzzy c-partitioned
    matrix u. Measures 'fuzziness' in partitioned clustering.
    Parameters
    ----------
    u : 2d array (C, N)
        Fuzzy c-partitioned matrix; N = number of data points and C = number
        of clusters.
    Returns
    -------
    fpc : float
        Fuzzy partition coefficient.
    """
    n = u.shape[1]

    return np.trace(u.dot(u.T)) / float(n)




def _init_memberships(X, centers, distance):
    d = _distance_data_centers(X, centers, distance)

    # fuzzy-probabilistic-style membership initialization with fixed fuzzyfier
    # parameter m=2
    u = _memberships_update_probabilistic(d, 2)

    return u, d


def _memberships_update_probabilistic(d, m):
    u = d ** (- 2. / (m - 1))
    u /= np.sum(u, axis=1)[:, np.newaxis]
    u = np.fmax(u, np.finfo(np.float64).eps)

    return u



def _centers_update(X, um):
    centers = np.dot(um.T, X)
    centers /= um.sum(axis=0)[:, np.newaxis]

    return centers




def _f_k_means_probabilistic(X, u_old, n_clusters, m, distance):
    #
    um = u_old ** m

    # Calculate cluster centers
    centers = _centers_update(X, um)

    # Calculate the distances between X (data) and centers
    d = _distance_data_centers(X, centers, distance)

    # Probabilistic cost function calculation da controllare!
    jm = (um * d ** 2).sum()

    # Membership update
    u = _memberships_update_probabilistic(d, m)

    return centers, u, jm, d





def f_k_means_main_loop(X, n_clusters, m, u, centers, d, tol_memberships,
                        tol_centroids, max_iter, constraint, distance):
    # Initialization loop parameters
    p = 0
    jm = np.empty(0)

    # Main fcmeans loop
    while p <  max_iter - 1:
        u_old = u.copy()
        centers_old = centers.copy()

        [centers, u, inertia, d] = \
            _f_k_means_probabilistic(X,
                                     u_old,
                                     n_clusters,
                                     m,
                                     distance)


        jm = np.hstack((jm, inertia))
        p += 1

        # Stopping rule on memberships
        if np.linalg.norm(u - u_old) < tol_memberships:
            print('Stopping rule on memberships')
            break

        # Stopping rule on centroids
        if np.linalg.norm(centers - centers_old) < tol_centroids:
            print('Stopping rule on centroids')
            break

    # Final calculations
    fpc = _fp_coeff(u)

    # Labels computation
    labels = _labels_computation(u)

    return centers, labels, inertia, p, u, fpc



def f_k_means(X, n_clusters, m, tol_memberships, tol_centroids, max_iter, init,
              constraint, distance, n_init):
    # if the initialization method is not 'k-means++',
    # an array of centroids is passed
    # and it is converted in float type
    if hasattr(init, '__array__'):
        n_clusters = init.shape[0]
        init = np.asarray(init, dtype=np.float64)

    # Initialize centers and memberships
    n_samples, n_features = X.shape

    centers = _init_centroids(X,
                              n_clusters,
                              init,
                              random_state=True,
                              x_squared_norms=row_norms(X, squared=True))

    u, d = _init_memberships(X, centers, distance)
    labels = _labels_computation(u)
    # Choose the optimization method

    centers, labels, inertia, n_iter, u, fpc = \
        f_k_means_main_loop(X,
                            n_clusters,
                            m,
                            u,
                            centers,
                            d,
                            tol_memberships,
                            tol_centroids,
                            max_iter,
                            constraint,
                            distance)

    return centers, labels



class FKMeans(BaseEstimator, ClusterMixin, TransformerMixin):
    def __init__(self, n_clusters, distance, constraint = 'probabilistic', m = 2.0,
                 init='k-means++', n_init=10, max_iter=100,
                 tol_memberships=1e-8, tol_centroids=1e-4):

        self.n_clusters = n_clusters
        self.m = m
        self.init = init
        self.max_iter = max_iter
        self.tol_memberships = tol_memberships
        self.tol_centroids = tol_centroids
        self.n_init = n_init
        self.constraint = constraint
        self.distance = distance


    def _check_fit_data(self, X):
        """Verify that the number of samples given is larger than k"""
        X = check_array(X, accept_sparse='csr', dtype=np.float64)
        if X.shape[0] == 1:
            X = X.T

        if X.shape[0] < self.n_clusters:
            raise ValueError("n_samples=%d should be >= n_clusters=%d" % (
                X.shape[0], self.n_clusters))

        return X

    def _check_test_data(self, X):
        X = check_array(X, accept_sparse='csr')
        if X.shape[0] == 1:
            X = X.T
        n_samples, n_features = X.shape
        expected_n_features = self.cluster_centers_.shape[1]
        if not n_features == expected_n_features:
            raise ValueError("Incorrect number of features. "
                             "Got %d features, expected %d" % (
                                 n_features, expected_n_features))
        if X.dtype.kind != 'f':
            warnings.warn("Got data type %s, converted to float "
                          "to avoid overflows" % X.dtype,
                          RuntimeWarning, stacklevel=2)
            X = X.astype(np.float)

        return X

    def fit(self, X, Y=None):
        """Compute fuzzy c-means clustering.
        Parameters
        ----------
        X : array-like or sparse matrix, shape=(n_samples, n_features)
        """

        if Y is None:
            self.centers = _init_centroids(X,
                                           self.n_clusters,
                                           init=self.init,
                                           random_state=None,
                                           x_squared_norms=row_norms(X, squared=True))
        else:
            n_labels = int(np.max(Y))
            self.centers = np.zeros([n_labels + 1, np.shape(X)[1]])
            for l in np.arange(n_labels + 1):
                self.centers[l, :] = np.mean(X[Y == l], axis=0)

        u, d = _init_memberships(X, self.centers, self.distance)

        cluster_centers, predicted_labels = \
            f_k_means(X,
                      n_clusters=self.n_clusters,
                      m=self.m,
                      tol_memberships=self.tol_memberships,
                      tol_centroids=self.tol_centroids,
                      max_iter=self.max_iter,
                      init=self.centers,
                      constraint=self.constraint,
                      distance=self.distance,
                      n_init=self.n_init)

        self.labels_ = predicted_labels
        self.cluster_centers_ = cluster_centers

        return self

    def predict(self, X):
        """Predict the closest cluster each sample in X belongs to.
        In the vector quantization literature, `cluster_centers_` is called
        the code book and each value returned by `predict` is the index of
        the closest code in the code book.
        Parameters
        ----------
         X : {array-like, sparse matrix}, shape = [n_samples, n_features]
                New data to predict.
        Returns
        -------
        labels : array, shape [n_samples,]
            Index of the cluster each sample belongs to.
        """
        d = _distance_data_centers(X, self.cluster_centers_, self.distance)
        predicted_labels = np.argmin(d, axis=1)

        return predicted_labels