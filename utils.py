import numpy as np
import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def preprocess(obs, env):
    """Performs necessary observation preprocessing."""
    if env in ['CartPole-v0']:
        return torch.tensor(obs, device=device).float()
    else:
        obs = obs/255.0  # rescaling observations from 0-255 to 0-1
        return torch.tensor(obs, device=device).float()
