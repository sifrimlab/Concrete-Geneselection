from scipy import sparse
import scanpy as sc
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


from sklearn.metrics import pairwise_distances, pairwise_distances_chunked
from joblib import Parallel, delayed
import numba
from tqdm import tqdm

def compute_ranking_matrix_parallel(D):
    """ Compute ranking matrix in parallel. Input (D) is distance matrix
    """
    # if data is small, no need for parallel
    if len(D) > 1000:
        n_jobs = -1
    else:
        n_jobs = 1
    r1 = Parallel(n_jobs, prefer="threads")(
            delayed(np.argsort)(i)
            for i in D.T
        )
    r2 = Parallel(n_jobs, prefer="threads")(
            delayed(np.argsort)(i)
            for i in r1
        )
    # write as a single array
    r2_array = np.zeros((len(r2), len(r2[0])), dtype='int32')
    for i, r2row in enumerate(r2):
        r2_array[i] = r2row
    return r2_array

@numba.njit(fastmath=True)
def populate_Q(Q, i, m, R1, R2):
    """ populate coranking matrix using numba for speed
    """
    for j in range(m):
        k = R1[i, j]
        l = R2[i, j]
        Q[k, l] += 1
    return Q

def iterate_compute_distances(data):
    """ Compute pairwise distance matrix iteratively, so we can see progress
    """
    n = len(data)
    D = np.zeros((n, n), dtype='float32')
    col = 0
    with tqdm(desc="computing pairwise distances", leave=False) as pbar:
        for i, distances in enumerate(
                pairwise_distances_chunked(data, n_jobs=-1),
            ):
            D[col : col + len(distances)] = distances
            col += len(distances)
            if i ==0:
                pbar.total = int(len(data) / len(distances))
            pbar.update(1)
    return D

def compute_coranking_matrix(data_ld, data_hd = None, D_hd = None):
    """ Compute the full coranking matrix
    """
    # compute pairwise probabilities
    if D_hd is None:
        D_hd = iterate_compute_distances(data_hd)
    D_ld =iterate_compute_distances(data_ld)
    n = len(D_ld)
    # compute the ranking matrix for high and low D
    rm_hd = compute_ranking_matrix_parallel(D_hd)
    rm_ld = compute_ranking_matrix_parallel(D_ld)
    # compute coranking matrix from_ranking matrix
    m = len(rm_hd)
    Q = np.zeros(rm_hd.shape, dtype='int16')
    for i in range(m):
        Q = populate_Q(Q,i, m, rm_hd, rm_ld)
    Q = Q[1:,1:]
    return Q

@numba.njit(fastmath=True)
def qnx_crm(crm, k):
    """ Average Normalized Agreement Between K-ary Neighborhoods (QNX)
    # QNX measures the degree to which an embedding preserves the local
    # neighborhood around each observation. For a value of K, the K closest
    # neighbors of each observation are retrieved in the input and output space.
    # For each observation, the number of shared neighbors can vary between 0
    # and K. QNX is simply the average value of the number of shared neighbors,
    # normalized by K, so that if the neighborhoods are perfectly preserved, QNX
    # is 1, and if there is no neighborhood preservation, QNX is 0.
    #
    # For a random embedding, the expected value of QNX is approximately
    # K / (N - 1) where N is the number of observations. Using RNX
    # (\code{rnx_crm}) removes this dependency on K and the number of
    # observations.
    #
    # @param crm Co-ranking matrix. Create from a pair of distance matrices with
    # \code{coranking_matrix}.
    # @param k Neighborhood size.
    # @return QNX for \code{k}.
    # @references
    # Lee, J. A., & Verleysen, M. (2009).
    # Quality assessment of dimensionality reduction: Rank-based criteria.
    # \emph{Neurocomputing}, \emph{72(7)}, 1431-1443.
    Python reimplmentation of code by jlmelville
    (https://github.com/jlmelville/quadra/blob/master/R/neighbor.R)
    """
    qnx_crm_sum = np.sum(crm[:k, :k])
    return qnx_crm_sum / (k * len(crm))

@numba.njit(fastmath=True)
def rnx_crm(crm, k):
    """ Rescaled Agreement Between K-ary Neighborhoods (RNX)
    # RNX is a scaled version of QNX which measures the agreement between two
    # embeddings in terms of the shared number of k-nearest neighbors for each
    # observation. RNX gives a value of 1 if the neighbors are all preserved
    # perfectly and a value of 0 for a random embedding.
    #
    # @param crm Co-ranking matrix. Create from a pair of distance matrices with
    # \code{coranking_matrix}.
    # @param k Neighborhood size.
    # @return RNX for \code{k}.
    # @references
    # Lee, J. A., Renard, E., Bernard, G., Dupont, P., & Verleysen, M. (2013).
    # Type 1 and 2 mixtures of Kullback-Leibler divergences as cost functions in
    # dimensionality reduction based on similarity preservation.
    # \emph{Neurocomputing}, \emph{112}, 92-108.
    Python reimplmentation of code by jlmelville
    (https://github.com/jlmelville/quadra/blob/master/R/neighbor.R)
    """
    n = len(crm)
    return ((qnx_crm(crm, k) * (n - 1)) - k) / (n - 1 - k)

#@numba.njit(fastmath=True)
def rnx_auc_crm(crm):
    """ Area Under the RNX Curve
    # The RNX curve is formed by calculating the \code{rnx_crm} metric for
    # different sizes of neighborhood. Each value of RNX is scaled according to
    # the natural log of the neighborhood size, to give a higher weight to smaller
    # neighborhoods. An AUC of 1 indicates perfect neighborhood preservation, an
    # AUC of 0 is due to random results.
    #
    # param crm Co-ranking matrix.
    # return Area under the curve.
    # references
    # Lee, J. A., Peluffo-Ordo'nez, D. H., & Verleysen, M. (2015).
    # Multi-scale similarities in stochastic neighbour embedding: Reducing
    # dimensionality while preserving both local and global structure.
    # \emph{Neurocomputing}, \emph{169}, 246-261.
    Python reimplmentation of code by jlmelville
    (https://github.com/jlmelville/quadra/blob/master/R/neighbor.R)
    """
    n = len(crm)
    num = 0
    den = 0
    qnx_crm_sum = 0
    for k in range(1, n - 2):
        #for k in (range(1, n - 2)):
        qnx_crm_sum += np.sum(crm[(k-1), :k]) + np.sum(crm[:k, (k-1)]) - crm[(k-1), (k-1)]
        qnx_crm = qnx_crm_sum / (k * len(crm))
        rnx_crm = ((qnx_crm * (n - 1)) - k) / (n - 1 - k)
        num += rnx_crm / k
        den += 1 / k
    return num / den

def _anndata_loader(adata, batch_size, shuffle=False):
    """
    Load Anndata object into pytorch standard dataloader.
    Args:
        adata (AnnData): Scanpy Anndata object.
        batch_size (int): Cells per batch.
        shuffle (bool): Whether to shuffle data or not.
    Return:
        sc_dataloader (torch.DataLoader): Dataloader containing the data.
    """
    if sparse.issparse(adata.X):
        data = adata.X.A
    else:
        data = adata.X
    data = torch.Tensor(data).to('cuda')
    sc_dataloader = torch.utils.data.DataLoader(data, shuffle=shuffle, batch_size=batch_size)
    return sc_dataloader

def _anndata_splitter(adata, train_size):
    """
    Splits Anndata object into a training and test set. Test proportion is 1-train_size.
    Args:
        adata (Anndata): Scanpy Anndata object.
        train_size (float): Proportion of whole dataset in training. Between 0 and 1.
    Returns:
        train_adata (Anndata): Training data subset.
        test_adata (Anndata): Test data subset.
    """
    assert train_size != 0
    n = len(adata)
    n_train = int(train_size*n)
    #n_test = n - n_train
    np.random.seed(seed=4445)
    perm_idx = np.random.permutation(n)
    train_idx = perm_idx[:n_train]
    test_idx = perm_idx[n_train:]
    train_adata = adata.copy()[train_idx,:]
    if len(test_idx) != 0:
        test_adata = adata.copy()[test_idx,:]
    else:
        test_adata = False
    return train_adata, test_adata

def read_data(FILE_PATH, platform=None,batch_size=None, train_size=0.8, shuffle=True,layer=None):
    ad = sc.read(FILE_PATH)
    if platform == 'Xenium':
        noprobes = pd.read_csv('/home/luna.kuleuven.be/u0137663/CAE_pytorch/Xenium_noprobes.csv',header=None)[0]
        ad = ad[:, ~ad.var_names.isin(noprobes)].copy()
    if batch_size is None:
        batch_size = max(ad.shape[0] // 256, 16)
    if layer is None:
        print('No layer specified')
    if layer is not None:
        ad.X = ad.layers[layer]
    # Subset hvg
#     if "highly_variable" in ad.var.keys():
#         ad = ad[:, ad.var["highly_variable"]==1].copy()

    train_adata, test_adata = _anndata_splitter(ad, train_size=train_size)
    train_loader = _anndata_loader(train_adata, batch_size=batch_size, shuffle=shuffle)
    test_loader = _anndata_loader(test_adata, batch_size=batch_size, shuffle=shuffle)

    return ad, train_loader, test_loader, batch_size, test_adata

def save_selected_genes(model,output_file):
    with open(output_file,'w') as file:
        gene_names = []
        if model.preselected_genes is not None:
            for g in model.preselected_genes:
                gene_names = list(model.preselected_genes)
                #gene_names = list(model.var_names[model.var_names.isin(model.preselected_genes)])
            for i in model.selector_layer.get_selected_feats():
                gene_names.append(model.var_names[~model.var_names.isin(model.preselected_genes)][i])
        else:
            for i in model.selector_layer.get_selected_feats():
                gene_names.append(model.var_names[i])
        for gene in gene_names:
            file.write(gene+'\n')

def plot_loss(epoch_hist, filepath=None):
#     fig,axs = plt.subplots(1,4)
    fig, ((ax1, ax2), (ax3, ax4),) = plt.subplots(2, 2)
    plt.rcParams['figure.figsize']=[40,40]
    #plot the loss
    ax1.plot(epoch_hist["train_loss"],color = "b")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("loss")
    if filepath != None:
        plt.savefig(filepath)
    #plot the validation loss
    ax2.plot(epoch_hist["valid_loss"], color = "g")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("validation_loss")
    if filepath != None:
        plt.savefig(filepath)
    ax3.plot(epoch_hist["mean_max_prob"], color = "r")
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("mean_max_prob")
    if filepath != None:
        plt.savefig(filepath)
    ax4.plot(epoch_hist["temperature"], color = "y")
    ax4.set_xlabel("Epoch")
    ax4.set_ylabel("temperature")
    if filepath != None:
        plt.savefig(filepath)
    plt.show()

def save_values(epoch_hist,filepath = None):
    model_values = pd.DataFrame({
                                "loss":epoch_hist["train_loss"],
                                 "val_loss":epoch_hist["valid_loss"],
                                 "mean_max_prob":epoch_hist["mean_max_prob"],
                                 "temperature":epoch_hist["temperature"],
                                 })
    model_values.to_csv(filepath)
