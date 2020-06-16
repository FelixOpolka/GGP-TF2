import numpy as np
import gpflow
from gpflow import Parameter
from gpflow.inducing_variables.inducing_variables import InducingPointsBase
from gpflow import covariances as cov
import tensorflow as tf

from utils import sparse_mat_to_sparse_tensor, get_submatrix


class GraphPolynomial(gpflow.kernels.base.Kernel):
    """
    GraphPolynomial kernel for node classification as introduced in
    Yin Chen Ng, Nicolo Colombo, Ricardo Silva: "Bayesian Semi-supervised
    Learning with Graph Gaussian Processes".
    """

    def __init__(self, sparse_adj_mat, feature_mat, idx_train, degree=3.0,
                 variance=1.0, offset=1.0):
        super().__init__(None)
        self.degree = degree
        self.offset = Parameter(offset, transform=gpflow.utilities.positive())
        self.variance = Parameter(variance, transform=gpflow.utilities.positive())
        # Pre-compute the P-matrix for transforming the base covariance matrix
        # (c.f. paper for details).
        sparse_adj_mat[np.diag_indices(sparse_adj_mat.shape[0])] = 1.0
        self.sparse_P = sparse_mat_to_sparse_tensor(sparse_adj_mat)
        self.sparse_P = self.sparse_P / sparse_adj_mat.sum(axis=1)
        self.feature_mat = feature_mat
        # Compute data required for efficient computation of training
        # covariance matrix.
        (self.tr_feature_mat, self.tr_sparse_P,
         self.idx_train_relative) = self._compute_train_data(
            sparse_adj_mat, idx_train, feature_mat,
            tf.sparse.to_dense(self.sparse_P).numpy())

    def _compute_train_data(self, adj_matrix, train_idcs, feature_mat,
                            conv_mat):
        """
        Computes all the variables required for computing the covariance matrix
        for training in a computationally efficient way. The idea is to cut out
        those features from the original feature matrix that are required for
        predicting the training labels, which are the training nodes' features
        and their neihbors' features.
        :param adj_matrix: Original dense adjacency matrix of the graph.
        :param train_idcs: Indices of the training nodes.
        :param feature_mat: Original dense feature matrix.
        :param conv_mat: Original matrix used for computing the graph
        convolutions.
        :return: Cut outs of only the relevant nodes.
            - Feature matrix containing features of only the "relevant" nodes,
            i.e. the training nodes and their neighbors. Shape [num_rel,
            num_feats].
            - Convolutional matrix for only the relevant nodes. Shape [num_rel,
            num_rel].
            - Indices of the training nodes within the relevant nodes. Shape
            [num_rel].
        """
        sub_node_idcs = get_submatrix(adj_matrix, train_idcs)
        # Compute indices of actual train nodes (excluding their neighbours)
        # within the sub node indices
        relative_train_idcs = np.isin(sub_node_idcs, train_idcs)
        relative_train_idcs = np.where(relative_train_idcs == True)[0]
        return (feature_mat[sub_node_idcs],
                conv_mat[sub_node_idcs, :][:, sub_node_idcs],
                relative_train_idcs)

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

    def K_diag_tr(self):
        base_cov = (self.variance * tf.matmul(self.tr_feature_mat, self.tr_feature_mat, transpose_b=True) + self.offset) ** self.degree
        if self.sparse:
            cov = tf.sparse.sparse_dense_matmul(self.tr_sparse_P, base_cov)
            cov = tf.sparse.sparse_dense_matmul(self.tr_sparse_P, cov, adjoint_b=True)
        else:
            cov = tf.matmul(self.tr_sparse_P, base_cov)
            cov = tf.matmul(self.tr_sparse_P, cov, adjoint_b=True)
        cov = tf.gather(tf.gather(cov, self.idx_train_relative, axis=0), self.idx_train_relative, axis=1)
        return tf.linalg.diag_part(cov)


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
