import numpy as np
import gpflow
import tensorflow as tf
from gpflow.mean_functions import Constant
from gpflow.models import SVGP
from scipy.cluster.vq import kmeans2

from graph_kernel import GraphPolynomial, NodeInducingPoints
from utils import load_dataset


def training_step(X_train, y_train, optimizer, gprocess):
    with tf.GradientTape(watch_accessed_variables=False) as tape:
        tape.watch(gprocess.trainable_variables)
        objective = -gprocess.elbo(X_train, y_train)
        gradients = tape.gradient(objective, gprocess.trainable_variables)
    optimizer.apply_gradients(zip(gradients, gprocess.trainable_variables))
    return objective


def evaluate(X_val, y_val, gprocess):
    pred_y, pred_y_var = gprocess.predict_y(X_val)
    pred_classes = np.argmax(pred_y.numpy(), axis=-1)
    acc = np.mean(pred_classes == y_val)
    return acc


def run_training():
    (adj_matrix, node_feats, node_labels, idx_train, idx_val,
     idx_test) = load_dataset("cora", tfidf_transform=True,
                              float_type=np.float64)
    idx_train = tf.constant(idx_train)
    idx_val = tf.constant(idx_val)
    idx_test = tf.constant(idx_test)
    num_classes = len(np.unique(node_labels))

    # Init kernel
    kernel = GraphPolynomial(adj_matrix, node_feats)

    # Init inducing points
    inducing_points = kmeans2(node_feats, len(idx_train), minit='points')[0]    # use as many inducing points as training samples
    inducing_points = NodeInducingPoints(inducing_points)

    # Init GP model
    mean_function = Constant()
    gprocess = SVGP(kernel, gpflow.likelihoods.MultiClass(num_classes),
                    inducing_points, mean_function=mean_function,
                    num_latent=num_classes, whiten=True, q_diag=False)

    # Init optimizer
    optimizer = tf.optimizers.Adam()

    for epoch in range(2000):
        elbo = -training_step(idx_train, node_labels[idx_train], optimizer,
                              gprocess)
        elbo = elbo.numpy()

        acc = evaluate(idx_test, node_labels[idx_test], gprocess)
        print(f"{epoch}:\tELBO: {elbo:.5f}\tAcc: {acc:.3f}")


if __name__ == '__main__':
    run_training()