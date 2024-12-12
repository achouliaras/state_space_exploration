import numpy as np
import torch
import agent.common.utils as utils
from torch import nn

class SimpleCritic(nn.Module):
    """torch.distributions implementation of a categorical policy for discrete environments."""
    def __init__(self, input_dim, output_dim, action_type, hidden_depth, hidden_dim, mode=0):
        super().__init__()
        self.action_type = action_type
        
        if self.architecture == 'Continuous':
            self.actor_mean = utils.mlp(np.array(input_dim).prod(), np.prod(output_dim), 
                                   hidden_depth, hidden_dim, activation=nn.Tanh)
            
        elif self.architecture == 'Discrete':
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

class DoubleQCritic(nn.Module):
    """Critic network, employes double Q-learning."""
    def __init__(self, obs_space, obs_dim, action_dim, action_type, architecture, hidden_dim, hidden_depth, mode = 0):
        super().__init__()
        self.architecture = architecture
        self.action_type = action_type

        # If obs is image-like use feature extraction
        if self.architecture =='CNN':
            self.cnn, self.flatten = utils.cnn(obs_space, obs_dim[0], mode = mode)
            obs_dim = self.flatten

        if self.action_type == 'Continuous':
            self.Q1 = utils.mlp(obs_dim + action_dim, hidden_dim, 1, hidden_depth)
            self.Q2 = utils.mlp(obs_dim + action_dim, hidden_dim, 1, hidden_depth)
        elif self.action_type == 'Discrete':    # Calculate Q-value for every Action on discrete action spaces
            self.Q1 = utils.mlp(obs_dim, hidden_dim, action_dim, hidden_depth)
            self.Q2 = utils.mlp(obs_dim, hidden_dim, action_dim, hidden_depth)

        self.outputs = dict()
        #self.apply(utils.weight_init)

    def forward(self, obs, action=None):
        if self.architecture =='CNN':
            obs = self.cnn(obs.permute(0, 1, 2, 3))
        
        if self.action_type == 'Continuous':
            input_data = torch.cat([obs, action], dim=-1)       # Add Action on continuous action spaces 
        elif self.action_type == 'Discrete':
            input_data = obs

        q1 = self.Q1(input_data)
        q2 = self.Q2(input_data)

        self.outputs['q1'] = q1
        self.outputs['q2'] = q2

        return q1, q2

    def log(self, logger, step):
        for k, v in self.outputs.items():
            logger.log_histogram(f'train_critic/{k}_hist', v, step)

        assert len(self.Q1) == len(self.Q2)
        for i, (m1, m2) in enumerate(zip(self.Q1, self.Q2)):
            assert type(m1) == type(m2)
            if type(m1) is nn.Linear:
                logger.log_param(f'train_critic/q1_fc{i}', m1, step)
                logger.log_param(f'train_critic/q2_fc{i}', m2, step)