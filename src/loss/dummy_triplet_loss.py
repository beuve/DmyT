import torch
import torch.nn as nn

from torch.nn import TripletMarginLoss, TripletMarginWithDistanceLoss


def get_binary_dummies(size, device):
    target_real = torch.ones(size).to(device)
    target_real[:target_real.size(0) // 2] = 0
    target_fake = torch.ones(size).to(device)
    target_fake[target_fake.size(0) // 2:] = 0
    return torch.cat([target_fake.unsqueeze(0), target_real.unsqueeze(0)])


class DummyTripletLoss(nn.Module):
    def __init__(self,
                 device,
                 size,
                 dummies=None,
                 antagonists=None,
                 distance=None):
        super().__init__()
        self.device = device
        if distance == None:
            self.triplet = TripletMarginLoss()
        else:
            self.triplet = TripletMarginWithDistanceLoss(
                distance_function=distance)
        self.dummies = get_binary_dummies(
            size, device) if dummies == None else dummies
        self.antagonists = torch.tensor(
            [1, 0], device=device) if antagonists == None else antagonists

    def forward(self, input, target, **kwargs):
        p = self.dummies[target]
        n = self.dummies[self.antagonists[target]]
        return self.triplet(input, p, n)
