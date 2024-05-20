import torch
import torch.nn as nn
import torch.nn.functional as F

from dgl.nn.pytorch import GATConv
from layers import StructuralAttentionLayer, TemporalAttentionLayer


class SemanticAttention(nn.Module):
    def __init__(self, in_size, hidden_size=128):
        super(SemanticAttention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False),
        )

    def forward(self, z):
        w = self.project(z).mean(0)  # (M, 1)
        beta = torch.softmax(w, dim=0)  # (M, 1)
        beta = beta.expand((z.shape[0],) + beta.shape)  # (N, M, 1)

        return (beta * z).sum(1)  # (N, D * K)


class HANLayer(nn.Module):
    """
    HAN layer.

    Arguments
    ---------
    num_meta_paths : number of homogeneous graphs generated from the metapaths.
    in_size : input feature dimension
    out_size : output feature dimension
    layer_num_heads : number of attention heads
    dropout : Dropout probability

    Inputs
    ------
    g : list[DGLGraph]
        List of graphs
    h : tensor
        Input features

    Outputs
    -------
    tensor
        The output feature
    """

    def __init__(
        self, args, num_meta_paths, in_size, out_size, layer_num_heads, dropout
    ):
        super(HANLayer, self).__init__()

        self.args = args
        # One GAT layer for each meta path based adjacency matrix
        self.gat_layers = nn.ModuleList()
        for i in range(num_meta_paths):
            self.gat_layers.append(
                GATConv(
                    in_size,
                    out_size,
                    layer_num_heads,
                    dropout,
                    dropout,
                    activation=F.elu,
                )
            )
        self.semantic_attention = SemanticAttention(
            in_size=out_size * layer_num_heads
        )
        self.num_meta_paths = num_meta_paths
    

    def forward(self, args, gs, h, nodes_num):
        structural_embeddings = []
        for i, g in enumerate(gs):
            a = self.gat_layers[i](g, h)
            structural_embeddings.append(a.flatten(1))
        structural_embeddings = torch.stack(structural_embeddings, dim=1)  # (N, M, D * K): N nodes, M meta-paths, D*K embeddings

        semantic_embeddings = self.semantic_attention(structural_embeddings)  # (N, D * K)
        return semantic_embeddings


class HAN(nn.Module):
    def __init__(
        self, args, time_snaps, num_meta_paths, in_size, hidden_size, out_size, num_heads, dropout
    ):
        super(HAN, self).__init__()

        self.args = args
        self.in_size = in_size
        self.time_snaps = time_snaps
        self.layers = nn.ModuleList()
        self.layers.append(
            HANLayer(
                args, num_meta_paths, in_size, hidden_size, num_heads[0], dropout
            )
        )
        for l in range(1, len(num_heads)):
            self.layers.append(
                HANLayer(
                    args,
                    num_meta_paths,
                    hidden_size * num_heads[l - 1],
                    hidden_size,
                    num_heads[l],
                    dropout,
                )
            )
        self.embedding_size = hidden_size * num_heads[-1]
        self.temporal_attn = nn.Sequential()
        layer = TemporalAttentionLayer(input_dim=self.embedding_size,
                                        n_heads=16,
                                        num_time_steps=self.time_snaps,
                                        attn_drop=dropout,
                                        residual=True)
        self.temporal_attn.add_module(name="temporal_layer_{}".format(0), module=layer)
        
        if self.args['ablation_type'] == 'only_time':
            self.time_semantic_attention = SemanticAttention(
                in_size=in_size
            )
            self.predict = nn.Linear(in_size, out_size)
        else:
            self.time_semantic_attention = SemanticAttention(
                in_size=hidden_size * num_heads[-1]
            )
            self.predict = nn.Linear(hidden_size * num_heads[-1], out_size)
        
        

    def forward(self, args, gs_pkg, nodes_num):
        all_embeddings = []
        
        for gs in gs_pkg:
            # metapath embedding for every graph
            structural_embeddings = []
            feat = gs['features']
            
            if self.args['ablation_type'] != 'only_time':
                for gnn in self.layers:
                    feat = gnn(args, gs['mgraphs'], feat, nodes_num)
            
            all_embeddings.append(feat)
        
        nodes_embeddings = [g[:,None,:] for g in all_embeddings] # list of [Ni, 1, F]
        
        # padding
        all_paddings = []
        for idx, embedding in enumerate(nodes_embeddings):
            valid_nodes = gs_pkg[idx]['valid_nodes']
            if self.args['ablation_type'] == 'only_time':
                emb_padding = torch.zeros(nodes_num, 1, self.in_size).to(args["device"])
            else:
                emb_padding = torch.zeros(nodes_num, 1, self.embedding_size).to(args["device"])
            for i in range(len(embedding)):
                emb_padding[valid_nodes[i]] = embedding[i]
            all_paddings.append(emb_padding)
        all_paddings = torch.cat(all_paddings, dim=1) # [nodes, snaps, feature dim]
        
        # Temporal Attention forward
        temporal_out = self.time_semantic_attention(all_paddings)

        return self.predict(temporal_out)
