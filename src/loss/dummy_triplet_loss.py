import torch
import torch.nn as nn

from torch.nn import TripletMarginWithDistanceLoss
from loss.dummy_config import get_dummy


def get_binary_dummies(size, device):
    target_real = torch.ones(size).to(device)
    target_real[:target_real.size(0) // 2] = 0
    target_fake = torch.ones(size).to(device)
    target_fake[target_fake.size(0) // 2:] = 0
    return torch.cat([target_fake.unsqueeze(0), target_real.unsqueeze(0)])


class DummyTripletLoss(nn.Module):
    def __init__(self, device, size, distance=None, weights=None):
        super().__init__()
        self.device = device
        self.triplet = TripletMarginWithDistanceLoss(
            distance_function=distance, reduction='none')
        self.dummies, self.antagonists, _ = get_dummy(size, device)
        self.weights = torch.tensor(
            [1] * self.dummies.size(0),
            device=device) if weights == None else torch.tensor(weights,
                                                                device=device)

    def forward(self, input, target, **kwargs):
        p = self.dummies[target]
        n = self.dummies[self.antagonists[target]]
        batch_weights = self.weights[target]
        return (self.triplet(input, p, n) *
                batch_weights).sum() / batch_weights.sum()
