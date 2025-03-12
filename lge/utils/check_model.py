import torch


def check_model(model: torch.nn.Module):
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(model)
    print(f"Total number of trainable parameters: {trainable_params}")
