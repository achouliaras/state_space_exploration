import numpy as np
import torch
import agent.common.utils as utils
from torch import nn

class Encoder(nn.Module):
    def __init__(self, obs_shape, latent_dim, architecture, mode=0):
        super().__init__()
        self.c, *self.obs_size = obs_shape
        self.architecture = architecture
        self.image_embedding_size = latent_dim

        if 'CNN' in architecture:
            self.cnn = utils.cnn(self.obs_size, self.c, latent_dim, mode=mode)
        else:
            # flatten
            self.image_embedding_size = 512
            self.mlp = utils.mlp(input_dim=np.array(obs_shape).prod(), 
                                 output_dim=512, 
                                 hidden_depth=1, 
                                 activation=nn.ReLU)

        if 'LSTM' in architecture:
            self.memory_module = utils.lstm(self.image_embedding_size, self.semi_memory_size)
            # self.memory_norm = nn.LayerNorm(self.semi_memory_size)

        self.outputs = dict()

    @property
    def obs_shape(self):
        return (self.c, *self.obs_size)
    
    @property
    def memory_size(self):
        return 2*self.semi_memory_size

    @property
    def semi_memory_size(self):
        return self.image_embedding_size

    @property
    def embedding_size(self):
        return self.semi_memory_size
    
    def forward(self, obs, memory=None):
        x = obs.permute(0, 1, 2, 3)
        
        # print(f'input: {x[0]}')
        if 'CNN' in self.architecture:
            # print('Obs shape ', x[0].shape)
            x = self.cnn(x)
        else:
            x = self.mlp(x) # CHECK ISSUE 
            # print('MLP')
        # print(f'output: {x[0]}')
        # x = x.reshape(x.shape[0], -1)

        if torch.isnan(memory).any(): print(f"Input memory contains NaN!: {memory}")
        if torch.isinf(memory).any(): print(f"Input memory contain Inf!: {memory}")

        if 'LSTM' in self.architecture:
            hidden = (memory[:, :self.semi_memory_size], memory[:, self.semi_memory_size:])
            hidden, cell = self.memory_module(x, hidden)
            # hidden = self.memory_norm(hidden)
            # cell = self.memory_norm(cell)
            embedding = hidden
            memory = torch.cat((hidden, cell), dim=1)
        else:
            embedding = x
        
        if (torch.any(torch.isnan(embedding))):
            print('Emb=', embedding)
            print('Mem=', memory)
        return embedding, memory

    def log(self, logger, step):
        for k, v in self.outputs.items():
            logger.log_histogram(f'train_encoder/{k}_hist', v, step)

        if 'CNN' in self.architecture:
            for l, n in enumerate(self.cnn):
                if type(n) == nn.Conv2d:
                    logger.log_param(f'train_encoder/conv{l}', n, step)
        else:
            for l, n in enumerate(self.mlp):
                if type(n) == nn.Linear:
                    logger.log_param(f'train_encoder/fc{l}', n, step)

        if 'LSTM' in self.architecture:
            # for i, m in enumerate(self.memory_module):
            #     if type(m) == nn.LSTMCell:
            # logger.log_param(f'train/lstm', self.memory_module, step)
            pass

class Decoder(nn.Module):
    def __init__(self, obs_shape, latent_dim, architecture, mode=0):
        super().__init__()
        self.c, *self.obs_size = obs_shape
        self.architecture = architecture
        self.image_embedding_size = latent_dim

        if 'CNN' in architecture:
            self.de_cnn = utils.de_cnn(self.obs_size, self.c, latent_dim, mode=mode)
        else:
            # flatten
            self.image_embedding_size = 512
            self.mlp = utils.mlp(input_dim=512, 
                                 output_dim=np.array(obs_shape).prod(), 
                                 hidden_depth=1, 
                                 activation=nn.ReLU,
                                 output_mod=[nn.Sigmoid(),nn.Unflatten(1, obs_shape)])

        # if 'LSTM' in architecture:
        #     self.memory_module = utils.lstm(self.image_embedding_size, self.semi_memory_size)

        self.outputs = dict()

    # @property
    # def memory_size(self):
    #     return 2*self.semi_memory_size

    # @property
    # def semi_memory_size(self):
    #     return self.image_embedding_size

    # @property
    # def embedding_size(self):
    #     return self.semi_memory_size
    
    def forward(self, embedding):
        # if 'LSTM' in self.architecture:
        #     hidden = (memory[:, :self.semi_memory_size], memory[:, self.semi_memory_size:])
        #     hidden = self.memory_module(x, hidden)
        #     x = hidden[0]
        #     memory = torch.cat(hidden, dim=1)
        # else:
        #      x = embedding
        
        x = embedding
        # print(f'input: {x[0]}')
        if 'CNN' in self.architecture:
            # print('Obs shape ', x[0].shape)
            z = self.de_cnn(x)
        else:
            z = self.mlp(x)
            # print('MLP')
        # print(f'output: {x[0]}')
        # x = x.reshape(x.shape[0], -1)

        obs = z.permute(0, 1, 2, 3)
        return obs

    def log(self, logger, step):
        for k, v in self.outputs.items():
            logger.log_histogram(f'train_encoder/{k}_hist', v, step)

        if 'CNN' in self.architecture:
            for l, n in enumerate(self.cnn):
                if type(n) == nn.Conv2d:
                    logger.log_param(f'train_encoder/conv{l}', n, step)
        else:
            for l, n in enumerate(self.mlp):
                if type(n) == nn.Linear:
                    logger.log_param(f'train_encoder/fc{l}', n, step)

        # if 'LSTM' in self.architecture:
        #     # for i, m in enumerate(self.memory_module):
        #     #     if type(m) == nn.LSTMCell:
        #     # logger.log_param(f'train/lstm', self.memory_module, step)
        #     pass

