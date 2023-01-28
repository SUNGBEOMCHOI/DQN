import numpy as np
import torch 
import torch.nn as nn

from utils import build_network

class Model(nn.Module):
    def __init__(self, model_cfg, device=torch.device('cpu')):
        super().__init__()
        self.device = device
        architecture = model_cfg['architecture']
        self.model = build_network(architecture)
        self.q_value = 0.0

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        if x.dim() == 3: # Image with no batch size
            x = x.unsqueeze(0)
        x = x.to(self.device)
        x = self.model(x)
        return x

    def get_action(self, x):
        x = self.forward(x)
        if x.dim() == 2:
            x = x.squeeze(0)
        self.q_value = x
        action = torch.argmax(x).item()
        return action