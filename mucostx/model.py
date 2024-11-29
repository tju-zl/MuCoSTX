import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.nn import Module
from torch_geometric.nn import GCNConv, BatchNorm, Linear
from torch_geometric.utils import mask_feature


class Model(Module):
    def __init__(self, args, in_dim):
        super().__init__()
        self.args = args
        
        self.encoder = GCNConv(in_dim, args.latent_dim, flow=args.flow, improved=True)
        self.decoder = GCNConv(args.latent_dim, in_dim, flow=args.flow, improved=True)
        self.batch_emb = nn.Embedding(args.n_adata, args.latent_dim)
       
        self.norm = BatchNorm(args.latent_dim)
        self.act = nn.ELU()
        self.info_nce = InfoNCE()
        
    def forward(self, x, n_x, batch_k, s_edge_index, sf_edge_index):
        x_p = mask_feature(x, p=0.2)[0]
        int_idx = batch_k.to(torch.long)
        batch_emb = self.batch_emb(int_idx)
        
        hio = self.encoder(x, s_edge_index)
        h = self.decoder(hio, s_edge_index)
        hi = hio - 0.1*batch_emb
        h0 = self.act(self.norm(hi))
        
        h1 = self.encoder(x_p, sf_edge_index)
        h1 = h1 - 0.1*batch_emb
        h1 = self.act(self.norm(h1))
        
        h2 = self.encoder(n_x, s_edge_index)
        h2 = self.act(self.norm(h2))
        
        loss = self.compute_loss(x, h, hio, hi, h0, h1, h2, batch_emb)
        return hi, h, loss
        
    def compute_loss(self, x, y, hio, hi, p, p1, p2, batch_emb):
        loss_rec = F.mse_loss(x, y)
        loss_ctr = self.info_nce(p, p1, p2, temperature=0.05)
        loss_okk = self.info_nce(hi, hio, temperature=0.05)
        # print(f'rec loss {loss_rec.item()}, ctr loss {loss_ctr.item()}')
        
        loss_orth = torch.mean(torch.sum(p*batch_emb, dim=1)**2)
        # loss_pano = torch.mean(torch.sum(batch_emb.pow(2), dim=1))
        
        # loss_pano = torch.mean(torch.sum(torch.abs(batch_emb), dim=1))
        
        return loss_rec + loss_ctr + loss_orth + loss_okk #+ loss_pano
    

class ModelHD(Module):
    def __init__(self, args, in_dim):
        super().__init__()
        self.args = args
        
        self.encoder = GCNConv(in_dim, args.latent_dim, flow=args.flow, improved=True)
        self.decoder = GCNConv(args.latent_dim, in_dim, flow=args.flow, improved=True)
       
        self.norm = BatchNorm(args.latent_dim)
        self.act = nn.ELU()
        self.info_nce = InfoNCE()
        
    def forward(self, x, n_x, s_edge_index, sf_edge_index):
        x_p = mask_feature(x, p=0.2)[0]
        
        hi = self.encoder(x, s_edge_index)
        h = self.decoder(hi, s_edge_index)

        h0 = self.act(self.norm(hi))
        
        h1 = self.encoder(x_p, sf_edge_index)
        h1 = self.act(self.norm(h1))
        
        h2 = self.encoder(n_x, s_edge_index)
        h2 = self.act(self.norm(h2))
        
        loss = self.compute_loss(x, h, h0, h1, h2)
        return hi, h, loss
        
    def compute_loss(self, x, y, p, p1, p2):
        loss_rec = F.mse_loss(x, y)
        loss_ctr = self.info_nce(p, p1, p2, temperature=0.05)
        
        return loss_rec + loss_ctr


class InfoNCE(nn.Module):
    def __init__(self, reduction='mean', negative_mode='unpaired'):
        super().__init__()
        self.reduction = reduction
        self.negative_mode = negative_mode

    def forward(self, query, positive_key, negative_keys=None, temperature=1.):
        return info_nce(query, positive_key, negative_keys,
                        temperature=temperature,
                        reduction=self.reduction,
                        negative_mode=self.negative_mode)


def info_nce(query, positive_key, negative_keys=None, temperature=1., reduction='mean', negative_mode='unpaired'):
    # Check input dimensionality.
    if query.dim() != 2:
        raise ValueError('<query> must have 2 dimensions.')
    if positive_key.dim() != 2:
        raise ValueError('<positive_key> must have 2 dimensions.')
    if negative_keys is not None:
        if negative_mode == 'unpaired' and negative_keys.dim() != 2:
            raise ValueError("<negative_keys> must have 2 dimensions if <negative_mode> == 'unpaired'.")
        if negative_mode == 'paired' and negative_keys.dim() != 3:
            raise ValueError("<negative_keys> must have 3 dimensions if <negative_mode> == 'paired'.")

    # Check matching number of samples.
    if len(query) != len(positive_key):
        raise ValueError('<query> and <positive_key> must must have the same number of samples.')
    if negative_keys is not None:
        if negative_mode == 'paired' and len(query) != len(negative_keys):
            raise ValueError("If negative_mode == 'paired', then <negative_keys> must have the same number of samples as <query>.")

    # Embedding vectors should have same number of components.
    if query.shape[-1] != positive_key.shape[-1]:
        raise ValueError('Vectors of <query> and <positive_key> should have the same number of components.')
    if negative_keys is not None:
        if query.shape[-1] != negative_keys.shape[-1]:
            raise ValueError('Vectors of <query> and <negative_keys> should have the same number of components.')

    # Normalize to unit vectors
    query, positive_key, negative_keys = normalize(query, positive_key, negative_keys)
    if negative_keys is not None:
        # Explicit negative keys

        # Cosine between positive pairs
        positive_logit = torch.sum(query * positive_key, dim=1, keepdim=True)

        if negative_mode == 'unpaired':
            # Cosine between all query-negative combinations
            negative_logits = query @ transpose(negative_keys)

        elif negative_mode == 'paired':
            query = query.unsqueeze(1)
            negative_logits = query @ transpose(negative_keys)
            negative_logits = negative_logits.squeeze(1)

        # First index in last dimension are the positive samples
        logits = torch.cat([positive_logit, negative_logits], dim=1)
        labels = torch.zeros(len(logits), dtype=torch.long, device=query.device)
    else:
        # Negative keys are implicitly off-diagonal positive keys.
        # Cosine between all combinations
        logits = query @ transpose(positive_key)
        # Positive keys are the entries on the diagonal
        labels = torch.arange(len(query), device=query.device)
    return F.cross_entropy(logits / temperature, labels, reduction=reduction)


def transpose(x):
    return x.transpose(-2, -1)


def normalize(*xs):
    return [None if x is None else F.normalize(x, dim=-1) for x in xs]