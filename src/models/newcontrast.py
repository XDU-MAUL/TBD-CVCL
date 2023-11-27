import numpy as np
import torch
from torch import nn


class InstanceLoss(nn.Module):
    def __init__(self, batch_size, temperature, device):
        super(InstanceLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device

        self.mask = self.mask_correlated_samples(batch_size)
        # self.criterion = nn.CrossEntropyLoss(reduction="sum")

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        # mask = mask.bool()
        return mask

    def forward(self, z_i, z_j, S, k=200):

        N = 2 * self.batch_size
        z = torch.cat((z_i, z_j), dim=0)
        sim = torch.matmul(z, z.T) / self.temperature

        # neg

        _, sorted_indices = torch.sort(S, descending=True, dim=-1)
        sort = sorted_indices[:, 0:k]
        for i in range(self.batch_size):
            S[i, sort[i, :]] = 1

        S[S < 1] = 0
        # S[S < 0] = 0
        # S[S > 0] = 1
        P = torch.cat([S, S], dim=0)
        P = torch.cat([P, P], dim=1)
        P = self.mask.cuda() - P
        P[P < 0] = 0
        P = P.bool()
        negative_samples = sim[P]  # (7493744,) -> 14392000
        neg_exp = torch.exp(negative_samples)
        print(neg_exp.size())
        neg_score = neg_exp.sum()

        # pos
        sim_i_j = torch.diag(sim, self.batch_size)
        sim_j_i = torch.diag(sim, -self.batch_size)
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0)  # （4000，）
        pos_exp = torch.exp(positive_samples)
        pos_score = pos_exp.sum()

        loss = -torch.log(pos_score / (pos_score + neg_score)).mean()
        return loss
