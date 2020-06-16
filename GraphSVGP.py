from typing import Tuple

import gpflow
import tensorflow as tf
from gpflow.models.svgp import SVGP


class GraphSVGP(SVGP):

    def log_likelihood(self, data: Tuple[tf.Tensor, tf.Tensor]) -> tf.Tensor:
        """
        Overrides the default log_likelihood implementation of SVGP to employ
        more efficient computation that is possible because we operate on a
        graph (rather than the Euclidean domain). Otherwise the SVGP produces
        OOM errors.
        """
        kl = self.prior_kl()
        f_mean, f_var = self._predict_f_graph(data[0])
        var_exp = self.likelihood.variational_expectations(f_mean, f_var,
                                                           data[1])
        if self.num_data is not None:
            scale = self.num_data / tf.shape(self.X)[0]
        else:
            scale = tf.cast(1.0, kl.dtype)
        likelihood = tf.reduce_sum(var_exp) * scale - kl
        return likelihood

    def _predict_f_graph(self, X):
        kernel = self.kernel; f = self.q_mu
        Z = self.inducing_variable.Z
        num_data = Z.shape[0]  # M
        num_func = f.shape[1]  # K
        Kmn = kernel.Kzx(Z, X)
        Kmm = kernel.Kzz(Z) + tf.eye(num_data, dtype=gpflow.default_float()) * gpflow.default_jitter()
        Lm = tf.linalg.cholesky(Kmm)

        # Compute projection matrix A
        A = tf.linalg.triangular_solve(Lm, Kmn, lower=True)

        # Compute the covariance due to the conditioning
        f_var = kernel.K_diag_tr() - tf.reduce_sum(tf.square(A), 0)
        shape = tf.stack([num_func, 1])
        f_var = tf.tile(tf.expand_dims(f_var, 0), shape)  # Shape [K, N, N] or [K, N]

        # Another backsubstitution in the unwhitened case
        if not self.whiten:
            A = tf.linalg.triangular_solve(tf.transpose(Lm), A, lower=False)

        # Construct the conditional mean
        f_mean = tf.matmul(A, f, transpose_a=True)
        if self.q_sqrt is not None:
            if self.q_sqrt.shape.ndims == 2:
                LTA = A * tf.expand_dims(tf.transpose(self.q_sqrt), 2)  # Shape [K, M, N]
            elif self.q_sqrt.shape.ndims == 3:
                L = tf.linalg.band_part(self.q_sqrt, -1, 0)    # Shape [K, M, M]
                A_tiled = tf.tile(tf.expand_dims(A, 0), tf.stack([num_func, 1, 1]))
                LTA = tf.matmul(L, A_tiled, transpose_a=True)   # Shape [K, M, N]
            else:
                raise ValueError(f"Bad dimension for q_sqrt: {self.q_sqrt.shape.ndims}")
            f_var = f_var + tf.reduce_sum(tf.square(LTA), 1)    # Shape [K, N]
        f_var = tf.transpose(f_var)     # Shape [N, K] or [N, N, K]
        return f_mean + self.mean_function(X), f_var