import torch
from loss.dummy_triplet_loss import DummyTripletLoss
from loss.triplet_loss import TripletLoss
from torch.nn import CrossEntropyLoss


class Losses:
    def __init__(self, name, loss):
        self.name = name
        self.value = loss

    def __eq__(self, other):
        self.name == other.name

    def from_string(name, device, size):
        if name == 'BCE':
            return Losses(name, CrossEntropyLoss())
        if name == 'Triplet':
            return Losses(name, TripletLoss(device))
        if name == 'DmyT':
            return Losses(name, DummyTripletLoss(device, size))
        raise ValueError('This loss name is undefined.')

    def BCE():
        return Losses('BCE', lambda x: x)

    def Triplet():
        return Losses('Triplet', lambda x: x)

    def DmyT():
        return Losses('DmyT', lambda x: x)
