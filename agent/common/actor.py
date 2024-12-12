import numpy as np
import torch
import agent.common.utils as utils
from torch import nn


class DiagGaussianActor(nn.Module):
    """torch.distributions implementation of an diagonal Gaussian policy for continuous environments."""
    def __init__(self, obs_dim, action_dim, architecture, hidden_dim, hidden_depth,
                 log_std_bounds):
        super().__init__()

        self.log_std_bounds = log_std_bounds
        self.trunk = utils.mlp(obs_dim, hidden_dim, 2 * action_dim, hidden_depth)

        self.outputs = dict()
        # self.apply(utils.weight_init)

    def forward(self, obs):
        mu, log_std = self.trunk(obs).chunk(2, dim=-1)
        # constrain log_std inside [log_std_min, log_std_max]
        log_std = torch.tanh(log_std)
        log_std_min, log_std_max = self.log_std_bounds
        log_std = log_std_min + 0.5 * (log_std_max - log_std_min) * (log_std + 1)
        return mu, log_std

    def log(self, logger, step):
        for k, v in self.outputs.items():
            logger.log_histogram(f'train_actor/{k}_hist', v, step)

        for i, m in enumerate(self.trunk):
            if type(m) == nn.Linear:
                logger.log_param(f'train_actor/fc{i}', m, step)

class SimpleActor(nn.Module):
    """torch.distributions implementation of a categorical policy for discrete environments."""
    def __init__(self, input_dim, output_dim, action_type, hidden_depth, hidden_dim, mode=0):
        super().__init__()
        self.action_type = action_type
        
        if self.action_type == 'Continuous':
            self.actor_mean = utils.mlp(np.array(input_dim).prod(), np.prod(output_dim), 
                                   hidden_depth, hidden_dim, activation=nn.Tanh)
            self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(output_dim)))  
        elif self.action_type == 'Discrete':
            self.actor = utils.mlp(np.array(input_dim).prod(), np.prod(output_dim), 
                                   hidden_depth, hidden_dim, activation=nn.ReLU)
        else:
            raise NotImplementedError
        
        self.outputs = dict()
        
    def forward(self, embedding):
        if self.action_type == 'Continuous':
            x_mean = self.actor_mean(embedding)
            x_logstd = self.actor_logstd.expand_as(x_mean)
            return x_mean, x_logstd
        elif self.action_type == 'Discrete':
            x = self.actor(embedding)
            return x
        else:
            raise NotImplementedError
        
    def log(self, logger, step):
        for k, v in self.outputs.items():
            logger.log_histogram(f'train_actor/{k}_hist', v, step)

        if self.action_type == 'Continuous':
            for l, n in enumerate(self.actor_mean):
                if type(n) == nn.Linear:
                    logger.log_param(f'train_actor/mean_fc{l}', n, step)
            for l, n in enumerate(self.actor_logstd):
                if type(n) == nn.Linear:
                    logger.log_param(f'train_actor/logstd_fc{l}', n, step)
        elif self.action_type == 'Discrete':
            for l, n in enumerate(self.actor):
                if type(n) == nn.Linear:
                    logger.log_param(f'train_actor/fc{l}', n, step)
        else:
            raise NotImplementedError

class CategoricalActor(nn.Module):
    """torch.distributions implementation of a categorical policy for discrete environments."""
    def __init__(self, obs_space, obs_dim, action_dim, architecture, hidden_dim, hidden_depth, mode=0):
        super().__init__()
        self.obs_space = obs_space
        self.architecture = architecture
        
        #print(obs_space.shape[0]) # Needs reshape to 3,7,7
        if 'CNN' in architecture:
            self.cnn, self.flatten = utils.cnn(obs_space, obs_dim[0], mode=mode)
            obs_dim = self.flatten
            
        self.trunk = utils.mlp(obs_dim, hidden_dim, action_dim, hidden_depth)

        self.outputs = dict()
        #self.apply(utils.weight_init)

    def forward(self, obs):
        if 'CNN' in self.architecture:
            x = self.trunk(self.cnn(obs.permute(0, 1, 2, 3)))
        else:
            x = self.trunk(obs)
             
        # logits = F.softmax(x, dim=1)
        logits = x
        return logits

    def log(self, logger, step):
        for k, v in self.outputs.items():
            logger.log_histogram(f'train_actor/{k}_hist', v, step)

        if 'CNN' in self.architecture:
            for l, n in enumerate(self.cnn):
                if type(n) == nn.Conv2d:
                    logger.log_param(f'train_actor/conv{l}', n, step)

        for i, m in enumerate(self.trunk):
            if type(m) == nn.Linear:
                logger.log_param(f'train_actor/fc{i}', m, step)