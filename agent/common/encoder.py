import numpy as np
import torch
import agent.common.utils as utils
from torch import nn

class Encoder(nn.Module):
    def __init__(self, obs_space, obs_dim, architecture, mode=0):
        super().__init__()
        self.obs_space = obs_space
        self.architecture = architecture
        self.image_embedding_size = None

        if 'CNN' in architecture:
            self.cnn, self.n_flatten = utils.cnn(obs_space, obs_dim[0], mode=mode)
            self.image_embedding_size = self.flatten
        else:
            # flatten
            self.image_embedding_size = 512
            self.mlp = utils.mlp(input_dim=np.array(self.obs_dim).prod(), output_dim=self.image_embedding_size, 
                                 hidden_depth=0, activation=nn.ReLU)

        if 'LSTM' in architecture:
            self.memory_module = utils.lstm(self.image_embedding_size, self.semi_memory_size)
            
        self.embedding_size = self.semi_memory_size

    @property
    def memory_size(self):
        return 2*self.semi_memory_size

    @property
    def semi_memory_size(self):
        return self.image_embedding_size
    
    def forward(self, obs, memory):
        x = obs.permute(0, 1, 2, 3)

        if 'CNN' in self.architecture:
            x = self.cnn(x)
        else:
            x = self.mlp(x)
             
        x = x.reshape(x.shape[0], -1)

        if 'LSTM' in self.architecture:
            hidden = (memory[:, :self.semi_memory_size], memory[:, self.semi_memory_size:])
            hidden = self.memory_module(x, hidden)
            embedding = hidden[0]
            memory = torch.cat(hidden, dim=1)
        else:
            embedding = x
            memory = None
        
        return embedding, memory

    def log(self, logger, step):
        for k, v in self.outputs.items():
            logger.log_histogram(f'train/{k}_hist', v, step)

        if 'CNN' in self.architecture:
            for l, n in enumerate(self.cnn):
                if type(n) == nn.Conv2d:
                    logger.log_param(f'train/conv{l}', n, step)
        else:
            for l, n in enumerate(self.mlp):
                if type(n) == nn.Linear:
                    logger.log_param(f'train/fc{l}', n, step)

        if 'LSTM' in self.architecture:
            for i, m in enumerate(self.memory_module):
                if type(m) == nn.Linear:
                    logger.log_param(f'train/lstm{i}', m, step)