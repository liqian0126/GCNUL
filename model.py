import sys, os
import torch
import random

import torch.nn as nn
import torch.nn.functional as F
from utils import complexSparsemax

class UserLinkageClassifer(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, weight=None) -> None:
        super().__init__()
        self.fc_1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.Tanh()

        self.fc_2 = nn.Linear(hidden_size, output_size)

        self.ce_loss = nn.CrossEntropyLoss(weight=None)
        
        # for n, p in self.named_parameters():
        #     if "bias" in n:
        #         nn.init.constant_(p, 0.1)
        #     else:
        #         nn.init.xavier_normal_(p)

    def forward(self, x):
        """
        para:
            labels: [bsz, input_size]
        return:
            loss:
            y_logits: [bsz, output_size]
        """
        
        h_1 = self.relu(self.fc_1(x))
        y_logits = self.fc_2(h_1)
        
        return y_logits


def batch_normalize_adj(self, adj):
        """Symmetrically normalize adjacency matrix."""
        rowsum = torch.sum(adj, dim=2)  # [bsz, node_num]
        d_inv_sqrt = torch.pow(rowsum, -0.5)
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = torch.diag_embed(d_inv_sqrt) # [bsz, node_num, node_num]
        adj = torch.matmul(torch.matmul(d_mat_inv_sqrt, adj), d_mat_inv_sqrt).float()
        return adj

class GCNLayer(nn.Module):
    def __init__(self, input_size, output_size, dropout):
        super().__init__()

        
        self.output_size = output_size

        self.linear = nn.Linear(input_size,output_size, bias=True)

        self.scoring_fn_target = nn.Parameter(torch.Tensor(1,output_size))
        self.scoring_fn_source = nn.Parameter(torch.Tensor(1, output_size))
        self.bias = nn.Parameter(torch.Tensor(1, output_size))

        self.leakyReLU = nn.LeakyReLU(0.2)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(p=dropout)

        # self.init_paras()
    
    def init_paras(self):
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.xavier_uniform_(self.scoring_fn_target)
        nn.init.xavier_uniform_(self.scoring_fn_source)
        torch.nn.init.zeros_(self.bias)
    
    def forward(self, x, adj):
        """
        args:
            x:   [bsz, num_nodes, input_size], num_nodes = max_a_len + max_b_len
            adj: [bsz, num_nodes, num_nodes]
        return:
            out_nodes_features: [bsz, num_nodes,output_size]
        """
        bsz = x.size()[0]
        num_nodes = x.size()[1]
        # x = self.dropout(x)
        nodes_features_proj = self.leakyReLU(self.linear(x).view(bsz, -1,self.output_size)) # [bsz, num_nodes, output_szie]
        # adj_norm = self.batch_normalize_adj(adj)
        adj_weight = adj.view(bsz, num_nodes, num_nodes)

        out_nodes_features = torch.matmul(adj_weight, nodes_features_proj) # [bsz, num_hodes, output_size]
        # out_nodes_features = out_nodes_features.permute(0, 2, 1, 3)  # [bsz, num_nodes, head_num, output_size_per_head]
        out_nodes_features = out_nodes_features.contiguous()

        # out_nodes_features = out_nodes_features.view(bsz, -1, self.output_size_per_head) # [bsz, num_nodes, head_num*output_size_per_head]

        return out_nodes_features # [bsz, num_hodes, output_size]


class CoAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.soft_max = nn.Softmax(dim=-1)
    def forward(self, a_grid_emb, b_grid_emb, a_feat, b_feat, a_mask=None, b_mask=None):
        """
        args:
            a_mask: [bsz, seq_len_a, 1]
            b_mask: [bsz, seq_len_b, 1]
        """
        a_feat = a_feat.unsqueeze(1)  # [bsz, 1, hidden_size]
        b_feat = b_feat.unsqueeze(1)  # [bsz, 1, hidden_size]

        # [bsz, 1, hidden_size] * [bsz, hidden_size, seq_len_b]  -> [bsz, 1, seq_len_b]
        sim_A_B = torch.matmul(a_feat, b_grid_emb.transpose(1,2)).contiguous() # [bsz, 1, seq_len_b]
        b_mask = b_mask.squeeze(-1).unsqueeze(1) # [bsz, 1, seq_len_b]
        score = torch.masked_fill(sim_A_B, b_mask, float("-inf"))

        if self.softmax_strategy == 'complex':
            score = torch.masked_fill(sim_A_B, b_mask, -1 * 1e8)
            sim_A_B = self.complexSoftmax(score)         
        else:
            score = torch.masked_fill(sim_A_B, b_mask, float("-inf"))
            sim_A_B = self.soft_max(score)
        
        y_b = torch.matmul(sim_A_B, b_grid_emb).squeeze(1)  # [bsz, 1, seq_len_b] * [bsz, seq_len_b, hidden_size]

        sim_B_A = torch.matmul(b_feat, a_grid_emb.transpose(1,2)).contiguous() # [bsz, 1, seq_len_b]
        
        a_mask = a_mask.squeeze(-1).unsqueeze(1) # [bsz, 1, seq_len_a]
        sim_B_A = self.soft_max(torch.masked_fill(sim_B_A, a_mask, float("-inf")))
        y_a = torch.matmul(sim_B_A, a_grid_emb).squeeze(1) # [bsz, 1, seq_len_b] * [bsz, seq_len_b, hidden_size]

        return y_a, y_b


class Basic_Model_pair_graph(nn.Module):
    def __init__(self, args, grid_num, embedding_dim, gcn_output_size, hidden_size, label_num) -> None:
        super().__init__()

        self.embedding = nn.Embedding(grid_num, embedding_dim) 

        self.gcn_layer_1 = GCNLayer(embedding_dim, gcn_output_size, 0.5)
        self.gcn_layer_2 = GCNLayer(gcn_output_size, gcn_output_size, 0.5)
        self.gcn_layer_3 = GCNLayer(gcn_output_size, gcn_output_size, 0.5)
        self.gcn_layer_4 = GCNLayer(gcn_output_size, gcn_output_size, 0.5)
        self.gcn_layer_5 = GCNLayer(gcn_output_size, gcn_output_size, 0.5)

        self.gcn_output_size = gcn_output_size

        # self.fc = nn.Sequential(
        #     nn.Linear(embedding_dim, self.gat_output_size),
        #     nn.LeakyReLU(0.2)
        # )
        self.tanh = nn.Tanh()
        # self.w = nn.Parameter(torch.rand([hidden_size]))
        self.co_att = CoAttention()

        self.classifier = nn.Sequential(
            nn.Linear(6*self.gcn_output_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, label_num),
        )
        self.hidden_size = hidden_size
        self.softmax = nn.Softmax()
    
    def get_seq_mask(self, len_a):
        """
        args:
            len_a: [bsz]
        return:
            mask: [bsz, seq_len, ]
        """
        bsz = len_a.size()[0]
        max_len = len_a.max().item()
        
        seq = torch.arange(0, max_len, device=len_a.device).type_as(len_a)
        seq = seq.unsqueeze(0).expand(bsz, max_len)

        mask = seq.ge(len_a.unsqueeze(1))
        return mask
    
    def gcn(self, a_grid_emb, b_grid_emb, graph):
        """
        return:
            a_grid_emb: [bsz, a_max_len, gcn_output_size]
            b_grid_emb: [bsz, b_max_len, gcn_output_size]
        """
        a_max_len = a_grid_emb.size(1)
        b_max_len = b_grid_emb.size(1)
        grid_emb = torch.cat([a_grid_emb, b_grid_emb], dim=1)  # [bsz, a_max_len + b_max_len, emb_size]

        gcn_out = self.gcn_layer_1(grid_emb, graph)
        gcn_out = self.gcn_layer_2(gcn_out, graph) # [bsz, a_max_len + b_max_len, gcn_output_size_per_head*head_num]
        # gcn_out = self.gcn_layer_3(gcn_out, graph)
        # gcn_out = self.gcn_layer_4(gcn_out, graph)
        # gcn_out = self.gcn_layer_5(gcn_out, graph)

        a_grid_emb = gcn_out[:, :a_max_len, :]
        b_grid_emb = gcn_out[:, a_max_len:, :]
        
        return a_grid_emb, b_grid_emb

    def forward(self, a_input_ids,  b_input_ids, len_a, len_b, graph):
        """
        args:
            len_a: [bsz],
            len_b: [bsz]
        """
        a_grid_emb = self.embedding(a_input_ids)  # [bsz, seq_len_a, emb_size]
        b_grid_emb = self.embedding(b_input_ids)  # [bsz, seq_len_b, emb_size]

        # a_grid_emb = self.fc(a_grid_emb)  # [bsz, seq_len_a, hidden_size]
        # b_grid_emb = self.fc(b_grid_emb)

        a_grid_emb, b_grid_emb = self.gcn(a_grid_emb, b_grid_emb, graph) # [bsz, a_max_len, gcn_output_size]

        a_mask = self.get_seq_mask(len_a) # [bsz, seq_len]
        b_mask = self.get_seq_mask(len_b) # [bsz, seq_len]


        a_mask = a_mask.unsqueeze(-1) # [bsz, seq_len, 1]
        b_mask = b_mask.unsqueeze(-1) # [bsz, seq_len, 1]
        a_grid_emb = torch.masked_fill(a_grid_emb, a_mask, float("-inf")) # [bsz, seq_len,hidden_size]
        b_grid_emb = torch.masked_fill(b_grid_emb, b_mask, float("-inf"))   # [bsz, seq_len,hidden_size]

        a_feat, _ = torch.max(a_grid_emb, dim=1)  # [bsz, hidden_size]
        b_feat, _ = torch.max(b_grid_emb, dim=1)  # [bsz, hidden_size]
        
        a_grid_emb = torch.masked_fill(a_grid_emb, a_mask, 0)
        b_grid_emb = torch.masked_fill(b_grid_emb, b_mask, 0)

        c_a, c_b = self.co_att(a_grid_emb, b_grid_emb, a_feat, b_feat, a_mask, b_mask)
        
        sub = torch.abs(a_feat-b_feat)
        dot = a_feat * b_feat
        # fc1 =torch.cat([a_feat, b_feat,sub,dot], dim = -1)
        fc1 = torch.cat([a_feat, b_feat, sub, dot, c_a, c_b], dim=-1) # [bsz, n*hidden_size]
        pred = self.classifier(fc1)
        return pred


