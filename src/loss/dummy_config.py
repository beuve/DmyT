import torch


# Real | Fake
def get_binary_dummies(size, device):
    target_real = torch.ones(size).to(device)
    target_real[:target_real.size(0) // 2] = 0
    target_fake = torch.ones(size).to(device)
    target_fake[target_fake.size(0) // 2:] = 0
    dummies = torch.cat([target_fake.unsqueeze(0), target_real.unsqueeze(0)])
    antagonists = torch.tensor([1, 0], device=device)
    labels = ['Fake', 'Real']
    return dummies, antagonists, labels


# Real | Reenactment | Face-Swap
def get_cat_dummies(size, device):
    target_real = torch.zeros((size), device=device)
    target_fake = torch.zeros((size), device=device)
    target_swapp = torch.zeros((size)).to(device)
    target_reenact = torch.zeros((size)).to(device)
    target_real[:size // 2] = 1
    target_fake[size // 2:] = 1
    target_swapp[size // 2:(8 * size) // 10] = 1
    target_reenact[(8 * size) // 10:] = 1
    dummies = torch.cat([
        target_swapp.unsqueeze(0),
        target_fake.unsqueeze(0),
        target_reenact.unsqueeze(0),
        target_fake.unsqueeze(0),
    ])
    antagonists = torch.tensor([1, 3, 1], device=device)
    labels = ['Face-Swap', 'Real', 'Reenactment']
    return dummies, antagonists, labels


# FaceSwap | Face2Face | FaceShifter | Real | Deepfake | NeuralTexture
def get_method_dummies(size, device):
    target_real = torch.zeros((size), device=device)
    target_fake = torch.zeros((size), device=device)
    target_deepfake = torch.zeros((size), device=device)
    target_faceswapp = torch.zeros((size), device=device)
    target_neural = torch.zeros((size), device=device)
    target_face2face = torch.zeros((size), device=device)
    target_faceshifter = torch.zeros((size), device=device)
    target_real[:size // 2] = 1
    target_fake[size // 2:] = 1
    target_deepfake[size // 2:(6 * size) // 10] = 1
    target_faceswapp[(6 * size) // 10:(7 * size) // 10] = 1
    target_faceshifter[(7 * size) // 10:(8 * size) // 10] = 1
    target_neural[(8 * size) // 10:(9 * size) // 10] = 1
    target_face2face[(9 * size) // 10:] = 1
    dummies = torch.cat([
        target_faceswapp.unsqueeze(0),
        target_face2face.unsqueeze(0),
        target_faceshifter.unsqueeze(0),
        target_real.unsqueeze(0),
        target_deepfake.unsqueeze(0),
        target_neural.unsqueeze(0),
        target_fake.unsqueeze(0),
    ])
    antagonists = torch.tensor([3, 3, 3, 6, 3, 3], device=device)
    labels = [
        'FaceSwap', 'Face2Face', 'FaceShifter', 'Real', 'Deepfake',
        'NeuralTexture'
    ]
    return dummies, antagonists, labels


get_dummy = get_method_dummies
