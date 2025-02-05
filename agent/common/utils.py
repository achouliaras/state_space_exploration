import torch
import torch.nn as nn
import numpy as np
import math

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
    # for name, param in memory_module.named_parameters():
    #     if "bias" in name:
    #         nn.init.constant_(param, 0.0)
    #     elif "weight" in name:
    #         # nn.init.orthogonal_(param, 1.0)
    #         nn.init.xavier_normal_(param,gain=0.1)
    # # Set forget gate bias to 1 for longer memory
    # memory_module.bias_ih[memory_size:2*memory_size].data.fill_(1.0)
    # memory_module.bias_hh[memory_size:2*memory_size].data.fill_(1.0)
    # for name, param in memory_module.named_parameters():
    #     if "bias" in name:
    #         nn.init.constant_(param, 0.0)
    #     elif "weight" in name:
    #         # nn.init.orthogonal_(param, 1.0)
    #         nn.init.xavier_normal_(param,gain=0.1)
    # # Set forget gate bias to 1 for longer memory
    # memory_module.bias_ih[memory_size:2*memory_size].data.fill_(1.0)
    # memory_module.bias_hh[memory_size:2*memory_size].data.fill_(1.0)
    # for name, param in memory_module.named_parameters():
    #     if "bias" in name:
    #         nn.init.constant_(param, 0.0)
    #     elif "weight" in name:
    #         # nn.init.orthogonal_(param, 1.0)
    #         nn.init.xavier_normal_(param,gain=0.1)
    # # Set forget gate bias to 1 for longer memory
    # memory_module.bias_ih[memory_size:2*memory_size].data.fill_(1.0)
    # memory_module.bias_hh[memory_size:2*memory_size].data.fill_(1.0)
    return memory_module

def conv_output_size(input_size, kernel_size, stride=1, padding=0, dilation=1):
    """
    Calculate the output size of a convolutional layer.
    Parameters:
        input_size (int): The size of the input dimension (e.g., height or width).
        kernel_size (int): The size of the kernel.
        stride (int): The stride of the convolution. Default is 1.
        padding (int): The amount of zero-padding. Default is 0.
        dilation (int): The spacing between kernel elements. Default is 1.
    Returns:
        int: The size of the output dimension.
    """
    return math.floor((input_size + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1)

def cnn(obs_shape, n_channels, embedding_size, mode=0):
    w, h = obs_shape
    channels = [[32,64,64],
                [16,32,64]]
    kernel_size = [[8,4,3],[2,2,2]] # Parameterisation
    stride = [[4,2,1],[1,1,1]]
    padding =[[0,0,0],[0,0,0]]

    channels  = channels[mode]
    kernel_size=kernel_size[mode]
    stride = stride[mode]
    padding = padding[mode]

    # Calculate the output size after convolutions
    for i in range(3):
        h = conv_output_size(h, kernel_size[i], stride[i], padding[i])
        w = conv_output_size(w, kernel_size[i], stride[i], padding[i])
    
    # Flatten layer input size
    flattened_size = h * w * channels[2]  # the number of output channels of the last conv layer
    
    feature_extractor = nn.Sequential(
        layer_init(nn.Conv2d(n_channels, channels[0], kernel_size=kernel_size[0], stride=stride[0], padding=padding[0])),
        nn.ReLU(),
        layer_init(nn.Conv2d(channels[0], channels[1], kernel_size=kernel_size[1], stride=stride[1], padding=padding[1])),
        nn.ReLU(), 
        layer_init(nn.Conv2d(channels[1], channels[2], kernel_size=kernel_size[2], stride=stride[2], padding=padding[2])),
        nn.ReLU(),
        nn.Flatten(),
        layer_init(nn.Linear(flattened_size, embedding_size),std=0.01),
        nn.ReLU(), 
    )
    return feature_extractor

def de_cnn(obs_shape, n_channels, embedding_size, mode=0):
    w, h = obs_shape
    channels = [[32,64,64],
                [16,32,64]]
    kernel_size = [[8,4,3],[2,2,2]] # Parameterisation
    stride = [[4,2,1],[1,1,1]]
    padding =[[0,0,0],[0,0,0]]

    channels  = channels[mode]
    kernel_size=kernel_size[mode]
    stride = stride[mode]
    padding = padding[mode]

    # Calculate the output size after convolutions
    for i in range(3):
        h = conv_output_size(h, kernel_size[i], stride[i], padding[i])
        w = conv_output_size(w, kernel_size[i], stride[i], padding[i])
    
    # Flatten layer input size
    flattened_size = h * w * channels[2]  # the number of output channels of the last conv layer
    
    image_reconstructor = nn.Sequential(
            layer_init(nn.Linear(embedding_size, flattened_size),std=0.01),
            nn.ReLU(),
            nn.Unflatten(1, (channels[2], w, h)),
            layer_init(nn.ConvTranspose2d(channels[2], channels[1], kernel_size=kernel_size[2], stride=stride[2], padding=padding[2], output_padding=0)),
            nn.ReLU(),
            layer_init(nn.ConvTranspose2d(channels[1], channels[0], kernel_size=kernel_size[1], stride=stride[1], padding=padding[1], output_padding=0)),
            nn.ReLU(),
            layer_init(nn.ConvTranspose2d(channels[0], n_channels, kernel_size=kernel_size[0], stride=stride[0], padding=padding[0])),
            nn.Sigmoid()  # Output normalized to [0, 1]
        )
    return image_reconstructor
    
def mlp(input_dim, output_dim, hidden_depth=0, hidden_dim=0, activation = nn.ReLU, w_init_std= 0.01, output_mod=None):
def mlp(input_dim, output_dim, hidden_depth=0, hidden_dim=0, activation = nn.ReLU, w_init_std= 0.01, output_mod=None):
def mlp(input_dim, output_dim, hidden_depth=0, hidden_dim=0, activation = nn.ReLU, w_init_std= 0.01, output_mod=None):
    if not isinstance(input_dim, int):
        input_dim = np.prod(input_dim)
    
    if hidden_depth == 0:
        mods = [nn.Flatten(), layer_init(nn.Linear(input_dim, output_dim),std=w_init_std)]
        mods = [nn.Flatten(), layer_init(nn.Linear(input_dim, output_dim),std=w_init_std)]
        mods = [nn.Flatten(), layer_init(nn.Linear(input_dim, output_dim),std=w_init_std)]
    else:
        mods = [nn.Flatten(), layer_init(nn.Linear(input_dim, hidden_dim)), activation(inplace=True)]
        for i in range(hidden_depth - 1):
            mods += [layer_init(nn.Linear(hidden_dim, hidden_dim)), activation(inplace=True)]
        mods.append(layer_init(nn.Linear(hidden_dim, output_dim),std=w_init_std))
    if output_mod is not None:
        mods.extend(output_mod)
    trunk = nn.Sequential(*mods)
    return trunk