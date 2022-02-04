import torch
import torch.nn as nn

from torch.nn import TripletMarginWithDistanceLoss, CosineSimilarity
from loss.dummy_config import get_dummy


class DummyTripletLoss(nn.Module):
    def __init__(self, nb_labels, device, size, weights=None, dummies=None):
        super().__init__()
        self.device = device
        distance = lambda x, y: -CosineSimilarity()(
            x, y) if nb_labels > 2 else None
        self.triplet = TripletMarginWithDistanceLoss(
            distance_function=distance, reduction='none')
        self.dummies, self.antagonists, _ = dummies if dummies != None else get_dummy(
            nb_labels, size, device)
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
