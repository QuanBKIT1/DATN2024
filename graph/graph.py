import numpy as np
import torch


def k_adjacency(A, k, with_self=False, self_factor=1):
    # A is a 2D square array
    if isinstance(A, torch.Tensor):
        A = A.data.cpu().numpy()
    assert isinstance(A, np.ndarray)
    Iden = np.eye(len(A), dtype=A.dtype)
    if k == 0:
        return Iden
    Ak = np.minimum(np.linalg.matrix_power(A + Iden, k), 1) - \
        np.minimum(np.linalg.matrix_power(A + Iden, k - 1), 1)
    if with_self:
        Ak += (self_factor * Iden)
    return Ak


def edge2mat(link, num_node):
    A = np.zeros((num_node, num_node))
    for i, j in link:
        A[j, i] = 1
    return A


def normalize_digraph(A, dim=0):
    # A is a 2D square array
    Dl = np.sum(A, dim)
    h, w = A.shape
    Dn = np.zeros((w, w))

    for i in range(w):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i] ** (-1)

    AD = np.dot(A, Dn)
    return AD


def get_hop_distance(num_node, edge, max_hop=1):
    A = np.eye(num_node)

    for i, j in edge:
        A[i, j] = 1
        A[j, i] = 1

    # compute hop steps
    hop_dis = np.zeros((num_node, num_node)) + np.inf
    transfer_mat = [
        np.linalg.matrix_power(A, d) for d in range(max_hop + 1)
    ]
    arrive_mat = (np.stack(transfer_mat) > 0)
    for d in range(max_hop, -1, -1):
        hop_dis[arrive_mat[d]] = d
    return hop_dis


class Graph:
    """The Graph to model the skeletons.

    Args:
        layout (str): must be one of the following candidates: 'wlasl_hrnet', 
        mode (str): must be one of the following candidates: 'stgcn_spatial', 'spatial'. Default: 'spatial'.
        max_hop (int): the maximal distance between two connected nodes.
            Default: 1
    """

    def __init__(self,
                 layout='wlasl_hrnet',
                 mode='spatial',
                 max_hop=1,
                 nx_node=1,
                 num_filter=3,
                 init_std=0.02,
                 init_off=0.04):

        self.max_hop = max_hop
        self.layout = layout
        self.mode = mode
        self.num_filter = num_filter
        self.init_std = init_std
        self.init_off = init_off
        self.nx_node = nx_node

        assert nx_node == 1 or mode == 'random', "nx_node can be > 1 only if mode is 'random'"
        assert layout in ['keypoint-27', 'keypoint-31', 'keypoint-49']

        self.get_layout(layout)
        self.hop_dis = get_hop_distance(self.num_node, self.inward, max_hop)

        assert hasattr(self, mode), f'Do Not Exist This Mode: {mode}'
        self.A = getattr(self, mode)()

    def __str__(self):
        return self.A

    def get_layout(self, layout):
        self.center = 0
        
        if layout == 'keypoint-27':
            self.inward = [(0, 1), (0, 2), (0, 3), (0, 4), (3, 5), (4, 6), (5, 7), (7, 8), (7, 9), (7, 11), (7, 13), (7, 15), (9, 10), (11, 12),
                           (13, 14), (15, 16), (6, 17), (17, 18), (17, 19), (17, 21), (17, 23), (17, 25), (19, 20), (21, 22), (23, 24), (25, 26)]
            self.num_node = 27
        
        elif layout == 'keypoint-31':
            self.inward = [(0, 1), (0, 2), (0, 3), (0, 4), (3, 5), (4, 6), (7, 8), (7, 10), (8, 9), (9, 10), (5, 11), (11, 12), (11, 13), (11, 15), (11, 17), (
                11, 19), (13, 14), (15, 16), (17, 18), (19, 20), (6, 21), (21, 22), (21, 23), (21, 25), (21, 27), (21, 29), (23, 24), (25, 26), (27, 28), (29, 30)]
            self.num_node = 31
        
        elif layout == 'keypoint-49':
            self.inward = [(0, 1), (0, 2), (0, 3), (0, 4), (3, 5), (4, 6), (5, 7), (7, 8), (7, 12), (7, 16), (7, 20), (7, 24), (8, 9), (9, 10), (10, 11), (12, 13), (13, 14), (14, 15), (16, 17), (17, 18), (18, 19), (20, 21), (21, 22), (22, 23), (24, 25),
             (25, 26), (26, 27), (6, 28), (28, 29), (28, 33), (28, 37), (28, 41), (28, 45), (29, 30), (30, 31), (31, 32), (33, 34), (34, 35), (35, 36), (37, 38), (38, 39), (39, 40), (41, 42), (42, 43), (43, 44), (45, 46), (46, 47), (47, 48)]
            self.num_node = 49
        
        else:
            raise ValueError(f'Do Not Exist This Layout: {layout}')

        self.self_link = [(i, i) for i in range(self.num_node)]
        self.outward = [(j, i) for (i, j) in self.inward]
        self.neighbor = self.inward + self.outward

    def stgcn_spatial(self):
        adj = np.zeros((self.num_node, self.num_node))
        adj[self.hop_dis <= self.max_hop] = 1
        normalize_adj = normalize_digraph(adj)
        hop_dis = self.hop_dis
        center = self.center

        A = []
        for hop in range(self.max_hop + 1):
            a_close = np.zeros((self.num_node, self.num_node))
            a_further = np.zeros((self.num_node, self.num_node))
            for i in range(self.num_node):
                for j in range(self.num_node):
                    if hop_dis[j, i] == hop:
                        if hop_dis[j, center] >= hop_dis[i, center]:
                            a_close[j, i] = normalize_adj[j, i]
                        else:
                            a_further[j, i] = normalize_adj[j, i]
            A.append(a_close)
            if hop > 0:
                A.append(a_further)
        return np.stack(A)

    def spatial(self):
        Iden = edge2mat(self.self_link, self.num_node)
        In = normalize_digraph(edge2mat(self.inward, self.num_node))
        Out = normalize_digraph(edge2mat(self.outward, self.num_node))
        A = np.stack((Iden, In, Out))
        return A

    def binary_adj(self):
        A = edge2mat(self.inward + self.outward, self.num_node)
        return A[None]

    def random(self):
        num_node = self.num_node * self.nx_node
        return np.random.randn(self.num_filter, num_node, num_node) * self.init_std + self.init_off
