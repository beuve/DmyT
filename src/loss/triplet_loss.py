import torch.nn as nn
from pytorch_metric_learning import miners
from torch.nn import TripletMarginLoss


class TripletLoss(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.triplet = TripletMarginLoss()

    def forward(self, input, target, **kwargs):
        triplet_miner = miners.TripletMarginMiner(margin=0.2,
                                                  type_of_triplets="semihard")
        a, p, n = triplet_miner(input, target)
        return self.triplet(input[a], input[p], input[n])
