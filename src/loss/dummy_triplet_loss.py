import torch
import torch.nn as nn

from torch.nn import TripletMarginLoss, TripletMarginWithDistanceLoss


def get_binary_dummies(size, device):
    target_real = torch.ones(size).to(device)
    target_real[:target_real.size(0) // 2] = 0
    target_fake = torch.ones(size).to(device)
    target_fake[target_fake.size(0) // 2:] = 0
    return [(target_fake.unsqueeze(0), 1), (target_real.unsqueeze(0), 0)]


class DummyTripletLoss(nn.Module):
    def __init__(self, device, size, dummies=None, loss=None):
        super().__init__()
        self.device = device
        if loss == None:
            self.triplet = TripletMarginLoss()
        else:
            self.triplet = TripletMarginWithDistanceLoss(
                distance_function=loss)
        self.dummies = get_binary_dummies(
            size, device) if dummies == None else dummies

    def forward(self, input, target, **kwargs):
        p = self.dummies[target[0].item()][0]
        n = self.dummies[self.dummies[target[0].item()][1]][0]
        for t in target[1:]:
            p = torch.cat((p, self.dummies[t.item()][0]), 0)
            n = torch.cat((n, self.dummies[self.dummies[t.item()][1]][0]), 0)

        return self.triplet(input, p, n)
