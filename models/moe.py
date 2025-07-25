import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy



class FFN(nn.Module):
    def __init__(self, fc1, input_dim, hidden_dim, dropout_rate=0.2):
        super(FFN, self).__init__()
        #self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc1 = fc1  #第一层共享
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)    #第二层独立
        self.norm = nn.LayerNorm(hidden_dim)
        self.res_proj = nn.Identity() if input_dim == hidden_dim else nn.Linear(input_dim, hidden_dim)

    def forward(self, x):
        residual = self.res_proj(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.norm(x + residual)
        return x


class GatedMoE(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_experts=4, top_k=2, tau=1.0):
        super(GatedMoE, self).__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)  # 第一层共享
        # 构造每个专家：共享fc1，独立fc2
        self.experts = nn.ModuleList([
            FFN(self.fc1, input_dim, hidden_dim) for _ in range(num_experts)
        ])

        #  使用 Xavier 初始化 gating 参数
        #self.Wg = nn.Parameter(torch.randn(input_dim, num_experts))
        #self.b = nn.Parameter(torch.randn(num_experts))
        self.Wg = nn.Parameter(torch.empty(input_dim, num_experts))
        self.b = nn.Parameter(torch.zeros(num_experts))
        nn.init.xavier_uniform_(self.Wg)

        #self.Wnoise = nn.Parameter(torch.randn(input_dim, num_experts))
        self.top_k = top_k
        #self.tau = tau
        self.final_fc = nn.Linear(hidden_dim, output_dim)


    def forward(self, x_list):
        batchsize = x_list[0].size(0)

        expert_outputs = []
        gates = []
        expert_probabilities = []

        for x in x_list:
            # Step 1: Add noise
            H = x @ self.Wg + self.b
            #noise = torch.randn_like(H)  # StandardNormal()
            #H += noise * F.softplus(x @ self.Wnoise)
            expert_probabilities.append(F.softmax(H, dim=-1))

            top_k_values, _ = torch.topk(H, self.top_k, dim=-1)
            mask = H >= top_k_values[:, -1, None]
            H_top_k = H.masked_fill(~mask, float('-inf'))

            #G = F.gumbel_softmax(H_top_k, tau=self.tau, hard=False)
            G = F.softmax(H_top_k, dim=1)
            gates.append(G)
            expert_outputs.append(torch.stack([expert(x) for expert in self.experts], dim=1))

        # Combine expert outputs with gates
        combined_output = sum(torch.sum(G.unsqueeze(-1) * e, dim=1)
                              for G, e in zip(gates, expert_outputs)) / len(x_list)

        output = self.final_fc(combined_output)

        # expert_probabilities = [F.softmax(prob, dim=-1) for prob in expert_probabilities]
        return output, expert_probabilities


def entropy(p):
    return -torch.sum(p * torch.log(p + 1e-9), dim=-1)


def entropy_regularization_loss(expert_probabilities):
    M = len(expert_probabilities)
    batch_size = expert_probabilities[0].size(0)
    H_mj = [entropy(prob).mean() for prob in expert_probabilities]
    avg_prob = torch.stack(expert_probabilities, dim=0).mean(dim=0)

    H_avg = entropy(avg_prob).mean()
    E = torch.abs((1 / M) * sum(H_mj) - H_avg)

    return E