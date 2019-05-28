import numpy as np
import scipy.sparse as sp


def sparse_to_tuple(mat):
    sparse_mx = sp.csr_matrix(mat)
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = mat.shape
    return coords, values, shape


def preprocess_graph(adj):
    adj = sp.coo_matrix(adj)
    adj_ = adj
    # adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    return sparse_to_tuple(adj_normalized)


def construct_feed_dict(struct_adj_norms, struct_adj_origs, struct_features, temporal_adj_origs,
                        struct_pos_weights, struct_norms, temporal_pos_weights, temporal_norms,
                        placeholders):
    # construct feed dictionary
    feed_dict = dict()
    for i, d in zip(placeholders['struct_adj_norms'], struct_adj_norms):
        feed_dict.update({i: d})
    for i, d in zip(placeholders['struct_adj_origs'], struct_adj_origs):
        feed_dict.update({i: d})
    for i, d in zip(placeholders['struct_features'], struct_features):
        feed_dict.update({i: d})
    for i, d in zip(placeholders['temporal_adj_origs'], temporal_adj_origs):
        feed_dict.update({i: d})

    feed_dict.update({placeholders['struct_pos_weights']: struct_pos_weights})
    feed_dict.update({placeholders['struct_norms']: struct_norms})
    feed_dict.update({placeholders['temporal_pos_weights']: temporal_pos_weights})
    feed_dict.update({placeholders['temporal_norms']: temporal_norms})

    return feed_dict



