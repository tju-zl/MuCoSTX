import torch
import scanpy as sc
import anndata as ad
import tqdm.notebook as tq
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from torch.optim.lr_scheduler import StepLR
import numpy as np
from .model import Model, ModelHD
from .utils import *


class MuCoSTX:
    def __init__(self, adata_list, args):
        self.args = args
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        
        self.new_adata_list = []
        self.spe_list = []
        self.x_list = []
        self.args.n_adata = len(adata_list)
        
        # compute the global hvg
        combined_adata = ad.concat(adata_list.copy(), join='inner')
        sc.pp.highly_variable_genes(combined_adata, flavor="seurat_v3", n_top_genes=3000)
        global_hvg = combined_adata.var_names[combined_adata.var['highly_variable']]
        
        for i, sec in enumerate(adata_list):
            adata, spatial_edge = self.adata_process(sec, global_hvg)
            raw_feature = torch.FloatTensor(adata.X).to(self.device)
            # add batch number to last dim of X
            batch_size = raw_feature.shape[0]
            batch_idx = torch.full((batch_size, 1), i, dtype=torch.float32, device=self.device)
            raw_feature = torch.cat([raw_feature, batch_idx], dim=-1)
            self.new_adata_list.append(adata)
            self.spe_list.append(spatial_edge)
            self.x_list.append(raw_feature)
            
        self.spe_all = self.tensor_process()
        self.fee_all = self.get_fea_edges()
        print(self.spe_all, self.fee_all)
        
        self.model = Model(self.args, raw_feature.shape[1]-1).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3, weight_decay=1e-4)
        # self.scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
        # self.scheduler = StepLR(self.optimizer, step_size=1, gamma=0.9)
    
    def train_emb(self):
        losses = []
        self.model.train()
        loader = NeighborLoader(self.fee_all, num_neighbors=[-1], 
                                batch_size=512, shuffle=True)
        for ep in tq.tqdm(range(1, 101)):
            running_loss = 0
            
            for batch in loader:
                # with torch.cuda.amp.autocast(enabled=self.args.amp):
                x = batch.x[:, :-1]
                b_idx = batch.x[:, -1]
                s_edge_index = batch.edge_index[:, batch.edge_attr == 0]
                sf_edge_index = batch.edge_index
                n_x = self.fee_all.x[np.random.choice(len(self.fee_all.x), size=len(x), replace=False)][:, :-1]
                self.optimizer.zero_grad()
                loss = self.model(x, n_x, b_idx, s_edge_index, sf_edge_index)[-1]
                loss.backward()
                # self.scaler.scale(loss).backward()
                # self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)
                # self.scaler.step(self.optimizer)
                # self.scaler.update()
                self.optimizer.step()
                running_loss += loss.item()
            # self.scheduler.step()
            
            losses.append(running_loss)
            print(f'x dim {len(x)}, running loss: {running_loss}')
            
        del batch
        del self.fee_all
        del self.spe_all
        torch.cuda.empty_cache()
        
        x = range(1, len(losses) + 1)
        plt.plot(x, losses)
        plt.show()
            
    def get_adata_new(self, idx):
        self.model.eval()
        with torch.no_grad():
            latent = self.model.encoder(self.x_list[idx][:, :-1], self.spe_list[idx])
            batch_idx = torch.full((len(latent), 1), idx, dtype=torch.long, device=self.device)
            latent = latent - 0.1*self.model.batch_emb(batch_idx.squeeze())
            self.new_adata_list[idx].obsm['mx'] = latent.cpu().detach().numpy()
        return self.new_adata_list[idx]
    
    def get_adata_all(self):
        self.model.eval()
        with torch.no_grad():
            for idx in range(self.args.n_adata):
                latent = self.model.encoder(self.x_list[idx][:, :-1], self.spe_list[idx])
                batch_idx = torch.full((len(latent), 1), idx, dtype=torch.long, device=self.device)
                # print(batch_idx.shape, self.model.batch_emb(batch_idx.squeeze()).shape)
                latent = latent - 0.1*self.model.batch_emb(batch_idx.squeeze())
                self.new_adata_list[idx].obsm['mx'] = latent.cpu().detach().numpy()
        return self.new_adata_list
        
    def adata_process(self, adata, global_hvg):
        # adata.var_names_make_unique()
        # sc.pp.filter_genes(adata, min_cells=10)
        sc.pp.filter_cells(adata, min_genes=200)
        # sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        adata = adata[:, global_hvg]
        sc.pp.scale(adata, zero_center=True, max_value=10)
        # adata = adata[:, adata.var['highly_variable']]
        
        spatial_edge = spatial_rknn(torch.FloatTensor(adata.obsm['spatial']), self.args).to('cuda')
        return adata, spatial_edge
    
    def tensor_process(self):
        graphs = (Data(x=self.x_list[i], edge_index=self.spe_list[i]) for i in range(self.args.n_adata))
        # 合并图
        all_edge_indices = []
        all_node_features = []
        node_offset = 0

        for graph in graphs:
            # 调整 edge_index 的节点编号
            edge_index = graph.edge_index + node_offset
            all_edge_indices.append(edge_index)
            # 累积节点特征
            all_node_features.append(graph.x)
            # 更新节点偏移
            node_offset += graph.x.size(0)

        # 合并所有图的 edge_index 和 x
        merged_edge_index = torch.cat(all_edge_indices, dim=1)
        merged_node_features = torch.cat(all_node_features, dim=0)

        # 构建合并后的图
        merged_graph = Data(x=merged_node_features, edge_index=merged_edge_index)
        return merged_graph
    
    def get_fea_edges(self):
        x = self.spe_all.x
        pca = PCA(n_components=self.args.latent_dim, random_state=self.args.seed)
        embedding = pca.fit_transform(x.cpu().numpy())
        feature_edge = feature_knn(torch.FloatTensor(embedding).to('cuda'), self.args)
        sfe = torch.concatenate([self.spe_all.edge_index, feature_edge], dim=-1)
        sfe_attr = torch.cat([torch.zeros(self.spe_all.edge_index.size(1)), 
                              torch.ones(feature_edge.size(1))], dim=0)
        return Data(x=x, edge_index=sfe, edge_attr=sfe_attr)
    

class MuCoSTHD:
    def __init__(self, adata, args):
        self.args = args
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        print('initial')
        adata, spatial_edge = self.adata_process(adata)
        print('pre')
        raw_feature = torch.FloatTensor(adata.X)
        self.x = raw_feature.to(self.device)
        self.spa_edge = spatial_edge
        print('spe')
        self.spe_all = self.tensor_process()
        self.fee_all = self.get_fea_edges()
        print(self.spe_all, self.fee_all)
        
        self.model = ModelHD(self.args, raw_feature.shape[1]).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3, weight_decay=1e-4)
        # self.scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
        # self.scheduler = StepLR(self.optimizer, step_size=1, gamma=0.9)
    
    def train_emb(self):
        losses = []
        self.model.train()
        loader = NeighborLoader(self.fee_all, num_neighbors=[-1], 
                                batch_size=256, shuffle=True)
        for ep in tq.tqdm(range(1, 51)):
            running_loss = 0
            
            for batch in loader:
                # with torch.cuda.amp.autocast(enabled=self.args.amp):
                x = batch.x
                s_edge_index = batch.edge_index[:, batch.edge_attr == 0]
                sf_edge_index = batch.edge_index
                n_x = self.fee_all.x[np.random.choice(len(self.fee_all.x), size=len(x), replace=False)]
                self.optimizer.zero_grad()
                loss = self.model(x, n_x, s_edge_index, sf_edge_index)[-1]
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)

                self.optimizer.step()
                running_loss += loss.item()
            # self.scheduler.step()
            
            losses.append(running_loss)
            print(f'x dim {len(x)}, running loss: {running_loss}')
            
        del batch
        del self.fee_all
        del self.spe_all
        torch.cuda.empty_cache()
        
        x = range(1, len(losses) + 1)
        plt.plot(x, losses)
        plt.show()
            
    def get_adata_new(self, adata):
        self.model.eval()
        with torch.no_grad():
            latent = self.model.encoder(self.x, self.spa_edge)
            adata.obsm['mx'] = latent.cpu().detach().numpy()
        return adata
        
    def adata_process(self, adata):
        sc.pp.filter_genes(adata, min_cells=10)
        sc.pp.filter_cells(adata, min_genes=50)
        sc.pp.filter_cells(adata, max_genes=8000)
        sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=2000)
        print('hvg')
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        print('hvg1')
        adata = adata[:, adata.var['highly_variable']]
        sc.pp.scale(adata, zero_center=True, max_value=10)
         
        spatial_edge = spatial_rknn(torch.FloatTensor(adata.obsm['spatial']), self.args).to('cuda')
        print(spatial_edge.size()[1] / adata.n_obs)
        print('hvg2')
        return adata, spatial_edge
    
    def tensor_process(self):
        graphs = Data(x=self.x, edge_index=self.spa_edge)
        return graphs
    
    def get_fea_edges(self):
        x = self.spe_all.x
        pca = PCA(n_components=self.args.latent_dim, random_state=self.args.seed)
        embedding = pca.fit_transform(x.cpu().numpy())
        print('pca')
        feature_edge = feature_knn(torch.FloatTensor(embedding).to('cuda'), self.args)
        print('knn')
        sfe = torch.concatenate([self.spe_all.edge_index, feature_edge], dim=-1)
        sfe_attr = torch.cat([torch.zeros(self.spe_all.edge_index.size(1)), 
                              torch.ones(feature_edge.size(1))], dim=0)
        return Data(x=x, edge_index=sfe, edge_attr=sfe_attr)
    