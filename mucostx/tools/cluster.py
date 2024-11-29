import random
import numpy as np
import torch
import ot
from torch_geometric.nn import radius_graph, knn_graph
from torch_geometric.utils import to_undirected


def mclust_multi_batch(adata_list, n_cluster, refine=False):
    n_sec = len(adata_list)
    for i, sec in enumerate(adata_list):
        pass
    
    print(f'All {n_sec} sections have clustered with {n_cluster}, \
          the overall ari is, the medium ari of all sections')
    
    
    
    
def mclust(adata, arg, refine=False, use=''):
    from sklearn.decomposition import PCA
    pca = PCA(n_components=32, random_state=arg.seed)
    embedding = pca.fit_transform(adata.obsm[use].copy())
    adata.obsm['emb_pca'] = embedding
    import rpy2.robjects as r_objects
    r_objects.r.library("mclust")
    import rpy2.robjects.numpy2ri
    rpy2.robjects.numpy2ri.activate()
    r_random_seed = r_objects.r['set.seed']
    r_random_seed(arg.seed)
    rmclust = r_objects.r['Mclust']
    res = rmclust(rpy2.robjects.numpy2ri.numpy2rpy(adata.obsm['emb_pca']), arg.n_domain, 'EEE')

    mclust_res = np.array(res[-2])
    adata.obs['mclust'] = mclust_res
    adata.obs['mclust'] = adata.obs['mclust'].astype('int')
    adata.obs['mclust'] = adata.obs['mclust'].astype('category')

    if refine:
        new_type = refine_label(adata, radius=25, key='mclust')
        adata.obs['mclust'] = new_type
    return adata


def refine_label(adata, radius=0, key='label'):
    n_neigh = radius
    new_type = []
    old_type = adata.obs[key].values

    # calculate distance
    position = adata.obsm['spatial']
    distance = ot.dist(position, position, metric='euclidean')

    n_cell = distance.shape[0]

    for i in range(n_cell):
        vec = distance[i, :]
        index = vec.argsort()
        neigh_type = []
        for j in range(1, n_neigh + 1):
            neigh_type.append(old_type[index[j]])
        max_type = max(neigh_type, key=neigh_type.count)
        new_type.append(max_type)

    new_type = [str(i) for i in list(new_type)]
    return new_type