import torch
import torch.nn as nn
from agent.common.actor_critic.critic_only import SimpleCritic
from agent.common.actor_critic.actor_only import SimpleActor
from agent.common.feature_extraction.encoder import Encoder

class ACNModel(nn.Module):
    def __init__(self, obs_dim, action_dim, action_type, latent_dim, architecture, mode):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.action_type = action_type
        self.latent_dim = latent_dim
        self.architecture = architecture
        self.mode = mode

        self.network = self.create_network()
        self.actor = self.create_actor()
        self.critic = self.create_critic()

    def create_network(self):
        # CNN, MLP, LSTM
        network = Encoder(obs_shape=self.obs_dim,
                          latent_dim=self.latent_dim,
                          architecture=self.architecture,
                          mode=self.mode)
        return network

    def create_critic(self):
        if self.action_type == 'Continuous':
            critic = SimpleCritic(input_dim=self.obs_dim, 
                                  output_dim=1,
                                  action_type=self.action_type,
                                  hidden_depth=1,
                                  hidden_dim=64
                                  )
        elif self.action_type == 'Discrete':
            critic = SimpleCritic(input_dim=self.network.embedding_size,
                                  output_dim=1,
                                  action_type=self.action_type,
                                  hidden_depth=0,
                                  hidden_dim=0
                                  )
        else:
            raise NotImplementedError
        return critic
    
    def create_actor(self):
        if self.action_type == 'Continuous':
            actor = SimpleActor(input_dim=self.obs_dim, 
                                output_dim=self.action_dim,
                                action_type=self.action_type,
                                hidden_depth=1,
                                hidden_dim=64
                                )
        elif self.action_type == 'Discrete':
            actor = SimpleActor(input_dim=self.network.embedding_size,
                                output_dim=self.action_dim,
                                action_type=self.action_type,
                                hidden_depth=0,
                                hidden_dim=0
                                )
        else:
            raise NotImplementedError
        return actor
    
    def reset_actor(self):
        # reset actor and critic
        self.actor = self.create_actor()
    
    def reset_critic(self):
        # reset actor and critic
        self.critic = self.create_critic()

    def reset_network(self):
        # reset network
        self.network = self.create_network()

    def train(self, training=True):
        self.training = training
        self.network.train(training)
        self.actor.train(training)
        self.critic.train(training)
    
    def save_model(self, model_dir, step):
        torch.save(
            self.network.state_dict(), '%s/network_%s.pt' % (model_dir, step)
        )
        torch.save(
            self.actor.state_dict(), '%s/actor_%s.pt' % (model_dir, step)
        )
        torch.save(
            self.critic.state_dict(), '%s/critic_%s.pt' % (model_dir, step)
        )
    
    def load_actor_critic(self, model_dir, step):
        self.actor.load_state_dict(
            torch.load('%s/actor_%s.pt' % (model_dir, step))
        )
        self.critic.load_state_dict(
            torch.load('%s/critic_%s.pt' % (model_dir, step))
        )

    def load_encoder(self, model_dir, step):
        self.network.load_state_dict(
            torch.load('%s/network_%s.pt' % (model_dir, step))
        )

    def forward(self, obs, memory = None):
        if self.action_type == 'Continuous':
            x = obs
        elif self.action_type == 'Discrete':
            x, memory = self.network(obs, memory)
        logits = self.actor(x)
        state_value = self.critic(x)

        return logits, state_value, memory

    def log(self, logger, step):
        self.network.log(logger,step)
        self.actor.log(logger,step)
        self.critic.log(logger,step)

class ACModel(nn.Module):
    def __init__(self, obs_dim, action_dim, action_type):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.action_type = action_type

        self.actor = self.create_actor()
        self.critic = self.create_critic()

    def create_critic(self):
        if self.action_type == 'Continuous':
            critic = SimpleCritic(input_dim=self.obs_dim, 
                                  output_dim=1,
                                  action_type=self.action_type,
                                  hidden_depth=1,
                                  hidden_dim=64
                                  )
        elif self.action_type == 'Discrete':
            critic = SimpleCritic(input_dim=self.obs_dim,
                                  output_dim=1,
                                  action_type=self.action_type,
                                  hidden_depth=0,
                                  hidden_dim=0
                                  )
        else:
            raise NotImplementedError
        return critic
    
    def create_actor(self):
        if self.action_type == 'Continuous':
            actor = SimpleActor(input_dim=self.obs_dim, 
                                output_dim=self.action_dim,
                                action_type=self.action_type,
                                hidden_depth=1,
                                hidden_dim=64
                                )
        elif self.action_type == 'Discrete':
            actor = SimpleActor(input_dim=self.obs_dim,
                                output_dim=self.action_dim,
                                action_type=self.action_type,
                                hidden_depth=0,
                                hidden_dim=0
                                )
        else:
            raise NotImplementedError
        return actor
    
    def reset_actor(self):
        # reset actor and critic
        self.actor = self.create_actor()
    
    def reset_critic(self):
        # reset actor and critic
        self.critic = self.create_critic()

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)
    
    def save_model(self, model_dir, step):
        torch.save(
            self.actor.state_dict(), '%s/actor_%s.pt' % (model_dir, step)
        )
        torch.save(
            self.critic.state_dict(), '%s/critic_%s.pt' % (model_dir, step)
        )
    
    def load_actor_critic(self, model_dir, step):
        self.actor.load_state_dict(
            torch.load('%s/actor_%s.pt' % (model_dir, step))
        )
        self.critic.load_state_dict(
            torch.load('%s/critic_%s.pt' % (model_dir, step))
        )

    def forward(self, obs, memory = None):
        x = obs
        logits = self.actor(x)
        state_value = self.critic(x)

        return logits, state_value, memory

    def log(self, logger, step):
        self.actor.log(logger,step)
        self.critic.log(logger,step)