import torch
import torch.nn as nn
from pytorch_metric_learning import miners
from torch.nn import TripletMarginLoss


class TripletLoss(nn.Module):
    def __init__(self, device, weights):
        super().__init__()
        self.device = device
        self.triplet = TripletMarginLoss(reduction='none')
        self.weights = torch.tensor(weights, device=device)

    def forward(self, input, target, **kwargs):
        triplet_miner = miners.TripletMarginMiner(margin=0.2,
                                                  type_of_triplets='semihard')
        a, p, n = triplet_miner(input, target)
        batch_weights = self.weights[target[a]]
        return torch.nan_to_num(
            (self.triplet(input[a], input[p], input[n]) * batch_weights).sum()
            / batch_weights.sum(), 0)
