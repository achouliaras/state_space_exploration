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
 
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

def lstm(embedding_size, memory_size):
    memory_module = nn.LSTMCell(embedding_size, memory_size)
    return memory_module

def cnn(obs_space, n_input_channels, mode=0):

    channels = [[[n_input_channels,32,64],
                               [32,64,64]],
                [[n_input_channels,16,32],  
                               [16,32,64]]]
    kernel_size = [[8,4,3],[2,2,2]] # Parameterisation
    stride = [[4,2,1],[1,1,1]]
    padding =[[0,0,0],[0,0,0]]
    pooling = [[1,1,1],[2,1,1]]

    channels  = channels[mode]
    kernel_size=kernel_size[mode]
    stride = stride[mode]
    padding = padding[mode]
    pooling = pooling[mode]

    feature_extractor = nn.Sequential(
        layer_init(nn.Conv2d(channels[0][0], channels[1][0], kernel_size=kernel_size[0], stride=stride[0], padding=padding[0])),
        nn.ReLU(), 
        nn.MaxPool2d(pooling[0]),
        layer_init(nn.Conv2d(channels[0][1], channels[1][1], kernel_size=kernel_size[1], stride=stride[1], padding=padding[1])),
        nn.ReLU(), 
        nn.MaxPool2d(pooling[1]),
        layer_init(nn.Conv2d(channels[0][2], channels[1][2], kernel_size=kernel_size[2], stride=stride[2], padding=padding[2])),
        nn.ReLU(), 
        nn.MaxPool2d(pooling[2]),
        nn.Flatten(),
    )
    
    # Compute shape by doing one forward pass
    with torch.no_grad():
        obs_sample = torch.as_tensor(obs_space.sample()[None]).permute(0, 1, 2, 3)
        n_flatten = feature_extractor(obs_sample.float()).shape[1]
    
    #feature_extractor

    return feature_extractor, n_flatten

def mlp(input_dim, output_dim, hidden_depth=0, hidden_dim=64, activation = nn.ReLU, output_mod=None):
    if not isinstance(input_dim, int):
        input_dim = np.prod(input_dim)
    
    if hidden_depth == 0:
        mods = [nn.Flatten(), layer_init(nn.Linear(input_dim, output_dim),std=0.01)]
    else:
        mods = [nn.Flatten(), layer_init(nn.Linear(input_dim, hidden_dim), activation(inplace=True))]
        for i in range(hidden_depth - 1):
            mods += [layer_init(nn.Linear(hidden_dim, hidden_dim)), activation(inplace=True)]
        mods.append(layer_init(nn.Linear(hidden_dim, output_dim),std=0.01))
    if output_mod is not None:
        mods.append(output_mod)
    trunk = nn.Sequential(*mods)
    return trunk