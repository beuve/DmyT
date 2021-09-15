import torch
import timm
from timm.data import resolve_data_config


def get_feature_size(model_name, device):
    model = timm.create_model(
        model_name,
        pretrained=True,
        num_classes=0,
    ).to(device)
    input_size = resolve_data_config({}, model=model)['input_size']
    test_im = torch.randn(input_size).to(device).unsqueeze(0)
    return model.forward(test_im).size(1)