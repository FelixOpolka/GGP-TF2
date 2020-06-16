"""
Data loading methods are adapted from
https://github.com/tkipf/gcn/blob/master/gcn/utils.py
"""
import os
import sys
import zipfile
from urllib import request
from urllib.error import URLError, HTTPError
import pickle as pk
import networkx as nx
import tensorflow as tf
import numpy as np
import scipy.sparse as sp
import pathlib

from sklearn.feature_extraction.text import TfidfTransformer


def _download_file(url, local_file_path):
    try:
        f = request.urlopen(url)
        print("downloading " + url)
        with open(local_file_path, "wb") as local_file:
            local_file.write(f.read())
    except HTTPError as e:
        print("HTTP Error:", e.code, url)
    except URLError as e:
        print("URL Error:", e.reason, url)


def _check_and_download_dataset(data_name):
    dataset_dir = pathlib.Path(os.getenv("PWD")) / "Dataset"
    if not(dataset_dir.is_dir()):
        dataset_dir.mkdir()
    if data_name == 'citation_networks':
        data_url = "https://www.dropbox.com/s/tln5wxqqp3o691s/citation_networks.zip?dl=1"
        data_dir = dataset_dir / "citation_networks"
    else:
        raise RuntimeError(f"Unsupported dataset {data_name}")
    if data_dir.is_dir():
        return True
    else:
        print("Downloading from "+data_url)
        _download_file(data_url, dataset_dir / f"{data_name}.zip")
        print("Download complete. Extracting to "+str(dataset_dir))
        zip_handler = zipfile.ZipFile(dataset_dir / f"{data_name}.zip", 'r')
        zip_handler.extractall(dataset_dir)
        zip_handler.close()
        return True


def _parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def _fix_citeseer(test_idx_reorder, test_idx_range, tx, ty, num_features,
                  num_classes):
    """
    Fix citeseer dataset (there are some isolated nodes in the graph).
    Find isolated nodes, add them as zero-vectors into the right position.
    :param test_idx_reorder: List of indices of the test set
    :param test_idx_range: NumPy array of indices of the test set. Hence, same
    as `test_idx_reorder` but sorted.
    :param num_features: Number of node features.
    :param num_classes: Number of output classes.
    :param tx: Sparse csr_matrix containing node features of the test set.
    :param ty: Sparse csr_matrix containing node labels (one-hot encoded) of
    the test set.
    :return: Extended tx and ty.
        - tx is a sparse lil_matrix.
        - ty is a numpy array.
    """
    test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
    tx_extended = sp.lil_matrix((len(test_idx_range_full), num_features))
    tx_extended[test_idx_range - min(test_idx_range), :] = tx
    tx = tx_extended
    ty_extended = np.zeros((len(test_idx_range_full), num_classes))
    ty_extended[test_idx_range - min(test_idx_range), :] = ty
    ty = ty_extended
    return tx, ty


def _load_raw_data(dataset_str):
    """Load data."""
    data_path = pathlib.Path(os.getenv('PWD')) / "Dataset/citation_networks/"
    _check_and_download_dataset('citation_networks')
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open(data_path / f"ind.{dataset_str}.{names[i]}", 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pk.load(f, encoding='latin1'))
            else:
                objects.append(pk.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = _parse_index_file(data_path / f"ind.{dataset_str}.test.index")
    test_idx_range = np.sort(test_idx_reorder)

    return x, y, tx, ty, allx, ally, graph, test_idx_reorder, test_idx_range


def load_dataset(dataset_str, tfidf_transform=True, float_type=np.float32):
    (x, y, tx, ty, allx, ally, graph, test_idx_reorder,
     test_idx_range) = _load_raw_data(dataset_str)
    num_features, num_classes = x.shape[1], y.shape[1]

    # Fix citeseer dataset.
    if dataset_str == 'citeseer':
        tx, ty = _fix_citeseer(test_idx_reorder, test_idx_range, tx, ty,
                               num_features, num_classes)

    # Construct features
    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    if dataset_str.lower() != 'pubmed' and tfidf_transform: # tf-idf transform features, unless it's pubmed, which already comes with tf-idf
        transformer = TfidfTransformer(smooth_idf=True)
        features = transformer.fit_transform(features)
    features = features.toarray()
    features = features.astype(float_type)

    # Construct adjacency matrix
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    adj = adj.astype(float_type)

    # Construct labels
    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]
    labels = labels.argmax(axis=1)

    # Generate train, val, test split
    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val = range(len(y), len(y)+500)

    return adj, features, labels, idx_train, idx_val, idx_test


def sparse_mat_to_sparse_tensor(sparse_mat):
    """
    Converts a scipy csr_matrix to a tensorflow SparseTensor.
    """
    coo = sparse_mat.tocoo()
    indices = np.stack([coo.row, coo.col], axis=-1)
    tensor = tf.sparse.SparseTensor(indices, sparse_mat.data, sparse_mat.shape)
    return tensor


def get_submatrix(adj_matrix, node_idcs):
    """
    Returns indices of nodes that are neighbors of any of the nodes in
    node_idcs.
    """
    adj_matrix[np.diag_indices(adj_matrix.shape[0])] = 1.0
    sub_mat = adj_matrix[node_idcs, :].tocoo()
    rel_node_idcs = np.unique(sub_mat.col)
    return rel_node_idcs


if __name__ == '__main__':
    load_dataset("cora", float_type=np.float64)