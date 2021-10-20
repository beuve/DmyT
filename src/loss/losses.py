from loss.dummy_triplet_loss import DummyTripletLoss
from loss.triplet_loss import TripletLoss
from torch.nn import CrossEntropyLoss
import torch


class Losses(torch.nn.Module):
    def __init__(self, name, loss):
        super(Losses, self).__init__()
        self.name = name
        self._loss = loss

    def from_string(name, device, nb_labels, size, weights=None):
        if name == 'BCE':
            return Losses(name, CrossEntropyLoss(weight=weights))
        if name == 'Triplet':
            weights = weights if weights != None else torch.tensor(
                [1] * nb_labels, device=device)
            return Losses(name, TripletLoss(device, weights=weights))
        if name == 'DmyT':
            return Losses(name, DummyTripletLoss(device, size,
                                                 weights=weights))
        raise ValueError('This loss name is undefined.')

    def forward(self, input, target):
        return self._loss(input, target)
