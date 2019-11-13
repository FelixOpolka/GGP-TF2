import numpy as np
import gpflow
from gpflow import Parameter
from gpflow.inducing_variables.inducing_variables import InducingPointsBase
from gpflow import covariances as cov
import tensorflow as tf

from utils import sparse_mat_to_sparse_tensor


class GraphPolynomial(gpflow.kernels.base.Kernel):
    """
    GraphPolynomial kernel for node classification as introduced in
    Yin Chen Ng, Nicolo Colombo, Ricardo Silva: "Bayesian Semi-supervised
    Learning with Graph Gaussian Processes".
    """

    def __init__(self, sparse_adj_mat, feature_mat, degree=3.0, variance=1.0,
                 offset=1.0):
        super().__init__([1])
        self.degree = degree
        self.offset = Parameter(offset, transform=gpflow.utilities.positive())
        self.variance = Parameter(variance, transform=gpflow.utilities.positive())
        # Pre-compute the P-matrix for transforming the base covariance matrix
        # (c.f. paper for details).
        sparse_adj_mat[np.diag_indices(sparse_adj_mat.shape[0])] = 1.0
        self.sparse_P = sparse_mat_to_sparse_tensor(sparse_adj_mat)
        self.sparse_P = self.sparse_P / sparse_adj_mat.sum(axis=1)
        self.feature_mat = feature_mat

    def K(self, X, Y=None, presliced=False):
        X = tf.reshape(tf.cast(X, tf.int32), [-1])
        X2 = tf.reshape(tf.cast(Y, tf.int32), [-1]) if Y is not None else X

        base_cov = (self.variance * tf.matmul(self.feature_mat, self.feature_mat, transpose_b=True) + self.offset) ** self.degree
        cov = tf.sparse.sparse_dense_matmul(self.sparse_P, base_cov)
        cov = tf.sparse.sparse_dense_matmul(self.sparse_P, cov, adjoint_b=True)
        cov = tf.gather(tf.gather(cov, X, axis=0), X2, axis=1)
        # print(f"Kff: {cov.shape}")
        return cov

    def K_diag(self, X, presliced=False):
        return tf.linalg.diag_part(self.K(X))


class NodeInducingPoints(InducingPointsBase):
    """
    Set of real-valued inducing points. See parent-class for details.
    """
    pass


@cov.Kuu.register(NodeInducingPoints, GraphPolynomial)
def Kuu_graph_polynomial(inducing_variable, kernel, jitter=None):
    """
    Computes the covariance matrix between the inducing points (which are not
    associated with any node).
    :param inducing_variable: Set of inducing points of type
    NodeInducingPoints.
    :param kernel: Kernel of type GraphPolynomial.
    :return: Covariance matrix between the inducing variables.
    """
    Z = inducing_variable.Z
    cov = (kernel.variance * (tf.matmul(Z, Z, transpose_b=True)) + kernel.offset) ** kernel.degree
    return cov


@cov.Kuf.register(NodeInducingPoints, GraphPolynomial, tf.Tensor)
def Kuf_graph_polynomial(inducing_variable, kernel, X):
    """
    Computes the covariance matrix between inducing points (which are not
    associated with any node) and normal inputs.
    :param inducing_variable: Set of inducing points of type
    NodeInducingPoints.
    :param kernel: Kernel of type GraphPolynomial.
    :param X: Normal inputs. Note, however, that to simplify the
    implementation, we pass in the indices of the nodes rather than their
    features directly.
    :return: Covariance matrix between inducing variables and inputs.
    """
    X = tf.reshape(tf.cast(X, tf.int32), [-1])
    Z = inducing_variable.Z
    base_cov = (kernel.variance * tf.matmul(kernel.feature_mat, Z, adjoint_b=True) + kernel.offset)**kernel.degree
    cov = tf.sparse.sparse_dense_matmul(kernel.sparse_P, base_cov)
    cov = tf.gather(tf.transpose(cov), X, axis=1)
    return cov
