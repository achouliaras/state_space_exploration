import torch
import torch.nn as nn
import numpy as np

def to_np(t):
    if t is None:
        return None
    elif t.nelement() == 0:
        return np.array([])
    else:
        return t.cpu().detach().numpy()
    
def cnn(obs_space, n_input_channels, mode=0):

    channels = [[[n_input_channels,32,64],
                               [32,64,64]],
                [[n_input_channels,16,32],  
                               [16,32,64]]]
    kernel_size = [[8,4,3],[2,2,2]] # Parameterisation
    stride = [[4,2,1],[2,2,2]]
    padding =[[0,0,0],[0,0,0]]

    channels  = channels[mode]
    kernel_size=kernel_size[mode]
    stride = stride[mode]
    padding = padding[mode]

    feature_extractor=nn.Sequential(
        nn.Conv2d(channels[0][0], channels[1][0], kernel_size=kernel_size[0], stride=stride[0], padding=padding[0]),
        nn.ReLU(),
        nn.Conv2d(channels[0][1], channels[1][1], kernel_size=kernel_size[1], stride=stride[1], padding=padding[1]),
        nn.ReLU(),
        nn.Conv2d(channels[0][2], channels[1][2], kernel_size=kernel_size[2], stride=stride[2], padding=padding[2]),
        nn.ReLU(),
        nn.Flatten(),
    )
    
    # Compute shape by doing one forward pass
    with torch.no_grad():
        obs_sample = torch.as_tensor(obs_space.sample()[None]).permute(0, 1, 2, 3)
        n_flatten = feature_extractor(obs_sample.float()).shape[1]
    
    #feature_extractor

    return feature_extractor, n_flatten

def mlp(input_dim, hidden_dim, output_dim, hidden_depth, output_mod=None):
    if not isinstance(input_dim, int):
        input_dim = np.prod(input_dim)
    
    if hidden_depth == 0:
        mods = [nn.Flatten(), nn.Linear(input_dim, output_dim)]
    else:
        mods = [nn.Flatten(), nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True)]
        for i in range(hidden_depth - 1):
            mods += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True)]
        mods.append(nn.Linear(hidden_dim, output_dim))
    if output_mod is not None:
        mods.append(output_mod)
    trunk = nn.Sequential(*mods)
    return trunk