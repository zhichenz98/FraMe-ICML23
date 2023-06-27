"""Fused Gromov-Wasserstein Mixture Model for Graph Template Learning

Implementation details:
ipot: solve OT problem with inexact proximal point iteration [Xie 2019]
fgwb: solve FGW barycenter problem with BCD iteration [Vayer 2020]
frame: optimize FGW mixture model based on the log likelihood
frame1: optimize FGW mixture model based on the sum of weighted FGW

Varibale Naming:
# input graphs: N
# barycenters: K
# nodes in barycenters: M
# attributes: d
transport plan: T
graph membership: gamma
base model weight: pi
"""
import numpy as np
from frame.fgwb import fgw_barycenters
import os

def load_data(dataset):
    A = np.load("./dataset/"+dataset+"/A.npy",allow_pickle=True).tolist()
    graph_labels = np.load("./dataset/"+dataset+"/graph_labels.npy",allow_pickle=True).tolist()
    node_labels = np.load("./dataset/"+dataset+"/node_labels.npy",allow_pickle=True).tolist()
    if os.path.exists("./dataset/"+dataset+"/X.npy"):  # for attributed
        X = np.load("./dataset/"+dataset+"/X.npy",allow_pickle=True).tolist()
        X = [np.array(ele) for ele in X]
    else:   # for plain
        X = []
        for i in range(len(A)):
            X.append(np.sum(A[i], axis = 1).reshape(-1,1)) 
    return A, X, np.array(graph_labels), np.array(node_labels)

def frame(A_list, X_list, atom_size, alpha = 0.5, bcd_iter = 10, max_iter = 10, sigma = 10.0, supervised = False, gid = [], gl = []):
    """ Fused Gromov-Wasserstein mixture model

    Args:
        A_list ([np.array]): a list of adjacency matrices, shape = [ni,ni]
        X_list ([np.array]): a list of node attribute matrices, shape = [ni,d]
        atom_size (np.array): a list of number of barycenter nodes, shape = [K]
        alpha (float): balancing WD and GWD. Defaults to 0.5.
        bcd_iter (int): number of iteation in BCD. Defaults to 50.
        max_iter (int): number of EM iteration. Defaults to 10.
        sigma (float): hyperpara for RBF kernel. Defaults to 10. higher the value, more deterministic the membership
        normalize (bool): normalize adjacency matrix and node attribute or not
        supervised (bool): incorporate supervision or not
        gid (np.array): graph_id for labelled graphs
        gl (np.array): class labels
    Returns:
        gamma (np.array): posterior prob/graph embed, shape = [N,K]
        node_embed (np.array): node_embed, shape = [n_G, n_B]
    """
    N = len(A_list) # number of input graphs
    K = len(atom_size)  # number of barycenters
    gamma = np.random.rand(N,K) # confidence of each graph w.r.t. each barycenter
    gamma = gamma / np.sum(gamma, axis = 1).reshape((-1,1))
    if supervised:
        label = np.zeros((len(gl),K))
        label[np.arange(len(gl)), gl] = 1
        gamma[gid] = label
    pi = np.zeros(K)    # weight for each barycenter
    BA = []    # list of barycenter adjacency matrices
    BX = []    # list of barycenter node attribute matrices
    T = []  # list of transport plans
    for i in range(K):
        BA.append([])
        BX.append([])
        T.append([])
    ## row-normalize A and X to make WD and GWD at the same scale
    mu_list = []    # marginal dist.
    for i in range(N):
        A_list[i] /= np.clip(np.max(abs(A_list[i]), axis = 0).reshape(1,-1), a_min = 1e-5, a_max = 1e5)
        X_list[i] /= np.clip(np.max(abs(X_list[i]), axis = 0).reshape(1,-1), a_min = 1e-5, a_max = 1e5)
        mu_list.append(np.ones(len(A_list[i]))/len(A_list[i]))
    FGW = np.zeros((N,K)).astype(np.float32)  # FGW distances between graphs and barycenters
    for iter in range(max_iter):
        gamma_old = gamma.copy()
       ## maximization
        for i in range(K):
            if iter == 0:
                BX[i], BA[i], T[i], FGW[:,i] = fgw_barycenters(atom_size[i], X_list, A_list, mu_list, gamma[:,i] / np.sum(gamma[:,i]), alpha = alpha, max_iter = bcd_iter, tol = 0)
            else:
                BX[i], BA[i], T[i], FGW[:,i] = fgw_barycenters(atom_size[i], X_list, A_list, mu_list, gamma[:,i] / np.sum(gamma[:,i]), alpha = alpha, max_iter = bcd_iter, init_C = BA[i], init_X = BX[i], tol = 0)
            pi[i] = np.sum(gamma[:,i]) / N
        ## expectation
        gamma = np.clip(np.exp(-sigma * FGW) * pi.reshape((1,-1)), a_min = 1e-10, a_max = 1e10) # in case of overflow
        gamma = gamma / np.sum(gamma, axis = 1).reshape((-1,1))
        if supervised:
            gamma[gid] = label
        
    T = [list(x) for x in zip(*T)]    # transpose T
    ## generate node embedding
    node_embed = np.empty((0, np.sum(atom_size)))
    for i in range(len(A_list)):
        tmp_embed = T[i][0] * gamma[i][0]
        for j in range(1, len(atom_size)):
            tmp_embed = np.concatenate((tmp_embed, T[i][j] * gamma[i][j]), axis = 1)
        node_embed = np.concatenate((node_embed, tmp_embed * tmp_embed.shape[0]), axis = 0)
    return gamma, node_embed


def reconstruct(T_list, gamma, BA, BX):
    """reconstruct graphs based on learned embeddings

    Args:
        T_list ([[np.array]]): transport plans, shape = [N,K,n_G,n_B]
        gamma (np.array): posterior prob/graph embed, shape = [N,K]
        BA ([np.array]): a list of barycenter adjacency matrices, shape = [Mj,Mj]
        BX ([np.array]): a list of barycenter node attribute matrices, shape = [Mj,d]
    """
    N, K = gamma.shape
    RA = []
    RX = []
    d = len(BX[0][0])
    for i in range(N):
        temp_n = len(T_list[i][0])
        temp_A = np.zeros((temp_n, temp_n))
        temp_X = np.zeros((temp_n, d))
        for j in range(K):
            temp_T = T_list[i][j] * temp_n
            temp_A += gamma[i,j] * temp_T @ BA[j] @ temp_T.T
            temp_X += gamma[i,j] * temp_T @ BX[j]
        RA.append(temp_A)
        RX.append(temp_X)
    return RA, RX   
