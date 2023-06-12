import torch
import torch.nn as nn


@torch.no_grad()
def lora_moved(save_weight):
    layer_move = []
    for name, params in save_weight.items():
        if 'lora.up' in name:
            lora_up = params
            lora_down = save_weight[name.replace('lora.up', 'lora.down')]
            weight = lora_up.squeeze() @ lora_down.squeeze()
            dist = weight.flatten().abs().mean().item()
            layer_move.append(dist)
    return sum(layer_move) / len(layer_move)


class LoRALinearLayer(nn.Module):

    def __init__(self, name, original_module, rank=4, alpha=1):
        super().__init__()

        self.name = name

        if original_module.__class__.__name__ == 'Conv2d':
            in_channels, out_channels = original_module.in_channels, original_module.out_channels
            self.down = torch.nn.Conv2d(in_channels, rank, (1, 1), bias=False)
            self.up = torch.nn.Conv2d(rank, out_channels, (1, 1), bias=False)
        else:
            in_features, out_features = original_module.in_features, original_module.out_features
            self.down = nn.Linear(in_features, rank, bias=False)
            self.up = nn.Linear(rank, out_features, bias=False)

        self.register_buffer('alpha', torch.tensor(alpha))

        nn.init.normal_(self.down.weight, std=1 / rank)
        nn.init.zeros_(self.up.weight)

        self.original_forward = original_module.forward
        original_module.forward = self.forward

        self.enable_drop = False

    def forward(self, hidden_states):
        if self.enable_drop and self.training:
            drop_mul = 0
        else:
            drop_mul = 1
        hidden_states = self.original_forward(hidden_states) + drop_mul * self.alpha * self.up(self.down(hidden_states))
        return hidden_states
