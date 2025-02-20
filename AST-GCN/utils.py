import numpy as np
import scipy.sparse as sp
from scipy.sparse import csr_matrix, diags, coo_matrix
import tensorflow as tf


def calculate_laplacian(adj):
    # 确保邻接矩阵是方阵
    assert adj.shape[0] == adj.shape[1], "Adjacency matrix must be square"

    # 添加自环并归一化
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    adj_normalized = sp.csr_matrix(adj_normalized)

    # 计算度矩阵
    degree = np.array(adj_normalized.sum(axis=1)).flatten()

    # 处理孤立节点（度为0的情况）
    degree[degree == 0] = 1e-12  # 避免除以零

    # 计算度矩阵的逆平方根（对角矩阵）
    degree_mat_inv_sqrt = sp.diags(1.0 / np.sqrt(degree))

    # 计算对称归一化的拉普拉斯矩阵
    lap = sp.eye(adj.shape[0]) - degree_mat_inv_sqrt @ adj_normalized @ degree_mat_inv_sqrt

    return adj_to_sparse_tensor(lap)




def normalize_adj(adj):
    """对称归一化邻接矩阵: D^{-1/2} (A + I) D^{-1/2}"""
    # 确保输入是 scipy 稀疏矩阵
    if not isinstance(adj, csr_matrix):
        adj = csr_matrix(adj)

    # 添加自环（self-loop）
    adj_self_loop = adj + diags(np.ones(adj.shape[0]))  # 等价于 A + I

    # 计算度矩阵（对角矩阵）
    degree = np.array(adj_self_loop.sum(axis=1)).flatten()
    degree[degree == 0] = 1e-12  # 避免除以零

    # 计算 D^{-1/2}
    degree_inv_sqrt = diags(1.0 / np.sqrt(degree))

    # 对称归一化：D^{-1/2} (A + I) D^{-1/2}
    adj_normalized = degree_inv_sqrt @ adj_self_loop @ degree_inv_sqrt
    return adj_normalized.tocsr()  # 返回 CSR 格式稀疏矩阵


def adj_to_sparse_tensor(adj):
    """将 scipy 稀疏矩阵转换为 TensorFlow 的 SparseTensor"""
    if not isinstance(adj, (csr_matrix, coo_matrix)):
        raise TypeError("Input must be a scipy.sparse.csr_matrix or coo_matrix")

    # 转换为 COO 格式提取索引和数据
    adj_coo = adj.tocoo()
    indices = np.vstack((adj_coo.row, adj_coo.col)).T  # shape [num_edges, 2]
    values = adj_coo.data.astype(np.float32)  # shape [num_edges]
    shape = adj_coo.shape

    # 构建 tf.SparseTensor
    return tf.SparseTensor(
        indices=indices,
        values=values,
        dense_shape=shape
    )