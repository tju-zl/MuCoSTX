import numpy as np
import torch
import faiss
import scanpy as sc
import anndata as ad
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected


# * project to common pca space
def pca_transform(adata_list, n_components=50):
    adata_comb = ad.concat(adata_list, label='batch', join='inner')
    sc.pp.pca(adata_comb, n_comps=n_components)
    start_idx = 0
    for adata in adata_list:
        end_idx = start_idx + adata.shape[0]
        adata.obsm['X_pca'] = adata_comb.obsm['X_pca'][start_idx:end_idx, :]
        start_idx = end_idx
    
    return adata_list


# * search mnn between each pair of sections
def cal_pair_mnn(adata1, adata2, start_idx_1, start_idx_2, k=10):
    """
    Calculate the MNN between two AnnData objects 
    (only calculate the nearest neighbors between two slices 
    to ensure that they are not searched within the same slice).
    Returns the edges that are the nearest neighbors to each other.
    """
    X1 = adata1.obsm['X_pca']
    X2 = adata2.obsm['X_pca']
    
    d = X1.shape[1]
    index1 = faiss.IndexFlatL2(d)
    index2 = faiss.IndexFlatL2(d)
    index1.add(X2.astype(np.float32))
    index2.add(X1.astype(np.float32))
    distances_1_to_2, indices_1_to_2 = index1.search(X1.astype(np.float32), k)
    distances_2_to_1, indices_2_to_1 = index2.search(X2.astype(np.float32), k)
    mnn_edges = []
    n1 = adata1.shape[0]
    n2 = adata2.shape[0]
    
    for i in range(n1):
        for j in indices_1_to_2[i]:
            if i in indices_2_to_1[j]:
                mnn_edges.append((i + start_idx_1, j + start_idx_2))
    print(f'section {len(X1)} and {len(X2)} has {len(mnn_edges)} mnn edges')
    
    return mnn_edges


# * get the feature graph on pca space.
def get_pyg_graph(adata_list, k=10, n_components=50):
    """
    Calculates pairwise MNN between multiple AnnData objects and 
    returns a PyTorch Geometric graph object.
    - adata_list: a list containing multiple AnnData objects.
    - k: number of nearest neighbors per cell.
    """
    adata_list = pca_transform(adata_list, n_components)
    
    X_all = np.vstack([adata.obsm['X_pca'] for adata in adata_list])
    n_cells_total = X_all.shape[0]
    
    edge_list = []
    
    start_idx = 0
    for i in range(len(adata_list)):
        adata1 = adata_list[i]
        iter_idx = start_idx + adata1.shape[0]
        for j in range(i + 1, len(adata_list)):
            adata2 = adata_list[j]
            # 计算两个 AnnData 之间的 MNN，传入各自的起始索引
            mnn_edges = cal_pair_mnn(adata1, adata2, start_idx, iter_idx, k)
            edge_list.extend(mnn_edges)
            iter_idx += adata2.shape[0]

        # 更新当前数据集的起始索引
        start_idx += adata_list[i].shape[0]
        
    edge_index = np.array(edge_list).T
    
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    # ! 转成无向边
    edge_index = to_undirected(edge_index)
    x = torch.tensor(X_all, dtype=torch.float)
    
    data = Data(x=x, edge_index=edge_index)
    print('Get the overall mnn graph...')
    print(f"number of node: {data.num_nodes}, {n_cells_total}")
    print(f"number of edges: {data.num_edges}")
    print(f"edge_index:\n{data.edge_index}")
    return data