import torch
from torch import nn
from agent.common.actor_critic.critic_only import SimpleCritic
from agent.common.actor_critic.actor_only import SimpleActor
from agent.common.feature_extraction.encoder import Encoder

class InverseTransitionModel(torch.nn.Module):
    def __init__(self, obs_dim, action_dim, action_type, latent_dim, architecture, mode):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.action_type = action_type
        self.latent_dim = latent_dim
        self.architecture = architecture
        self.mode = mode

        self.encoder = self.create_network()
        self.inverse_model = self.create_actor()

    @property
    def embedding_size(self):
        return self.encoder.embedding_size
    
    def create_network(self):
        # CNN, MLP, LSTM
        encoder = Encoder(obs_shape=self.obs_dim,
                          latent_dim=self.latent_dim,
                          architecture=self.architecture,
                          mode=self.mode)
        return encoder
    
    def create_actor(self):
        if self.action_type == 'Continuous':
            actor = SimpleActor(input_dim=self.obs_dim*2, 
                                output_dim=self.action_dim,
                                action_type=self.action_type,
                                hidden_depth=1,
                                hidden_dim=64
                                )
        elif self.action_type == 'Discrete':
            actor = SimpleActor(input_dim=self.encoder.embedding_size*2,
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
        self.inverse_model = self.create_actor()
    
    def reset_network(self):
        # reset encoder
        self.encoder = self.create_network()

    def train(self, training=True):
        self.training = training
        self.encoder.train(training)
        self.inverse_model.train(training)
    
    def save_model(self, model_dir, step):
        torch.save(
            self.encoder.state_dict(), '%s/encoder_%s.pt' % (model_dir, step)
        )
        torch.save(
            self.inverse_model.state_dict(), '%s/inverse_model_%s.pt' % (model_dir, step)
        )

    def forward(self, obs, next_obs, memory = None, next_memory_mask = None):
        if self.action_type == 'Continuous':
            x_t = obs
            x_t1 = next_obs
        elif self.action_type == 'Discrete':
            x_t, memory = self.encoder(obs, memory)
            x_t1, next_memory = self.encoder(next_obs, memory*next_memory_mask)
        x = torch.cat((x_t, x_t1), dim=1)
        logits = self.inverse_model(x)
        return logits, memory, next_memory

    def log(self, logger, step):
        self.encoder.log(logger,step)
        self.inverse_model.log(logger,step)

class LatentMDPModel(torch.nn.Module):
    def __init__(self, obs_dim, action_dim, action_type, latent_dim, architecture, mode):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.action_type = action_type
        self.latent_dim = latent_dim
        self.architecture = architecture
        self.mode = mode
        self.action_embedding_dim = 64

        self.encoder = self.create_network()

        if self.action_type == 'Continuous':
            self.action_embedding = SimpleActor(input_dim=self.action_dim, 
                                                output_dim=self.action_embedding_dim,
                                                action_type=self.action_type,
                                                hidden_depth=0,
                                                hidden_dim=0
                                                )
            self.inverse_model = self.create_actor(input_dim=self.obs_dim*2,
                                                   output_dim=self.action_dim)
        elif self.action_type == 'Discrete':
            self.action_embedding = nn.Embedding(action_dim,self.action_embedding_dim)
            self.inverse_model = self.create_actor(input_dim=self.encoder.embedding_size*2,
                                                   output_dim=self.action_dim)

        self.transition_model = self.create_actor(input_dim=self.encoder.embedding_size+self.action_embedding_dim,
                                                  output_dim=self.encoder.embedding_size)

    @property
    def embedding_size(self):
        return self.encoder.embedding_size
    
    def create_network(self):
        # CNN, MLP, LSTM
        encoder = Encoder(obs_shape=self.obs_dim,
                          latent_dim=self.latent_dim,
                          architecture=self.architecture,
                          mode=self.mode)
        return encoder
    
    def reset_network(self):
        # reset encoder
        self.encoder = self.create_network()

    def create_actor(self, input_dim, output_dim):
        if self.action_type == 'Continuous':
            actor = SimpleActor(input_dim=input_dim, 
                                output_dim=output_dim,
                                action_type=self.action_type,
                                hidden_depth=1,
                                hidden_dim=64
                                )
        elif self.action_type == 'Discrete':
            actor = SimpleActor(input_dim=input_dim,
                                output_dim=output_dim,
                                action_type=self.action_type,
                                hidden_depth=0,
                                hidden_dim=0
                                )
        else:
            raise NotImplementedError
        return actor
    
    def reset_inverse_model(self):
        # reset inverse model
        if self.action_type == 'Continuous':
            self.inverse_model = self.create_actor(input_dim=self.obs_dim,
                                                   output_dim=self.action_dim)
        elif self.action_type == 'Discrete':
            self.inverse_model = self.create_actor(input_dim=self.encoder.embedding_size,
                                                   output_dim=self.action_dim)
    
    def reset_transition_model(self):
        # reset transition model
        self.transition_model = self.create_actor(input_dim=self.encoder.embedding_size+self.action_embedding_dim,
                                                  output_dim=self.encoder.embedding_size)

    def save_model(self, model_dir, step):
        torch.save(
            self.encoder.state_dict(), '%s/network_%s.pt' % (model_dir, step)
        )
        torch.save(
            self.inverse_model.state_dict(), '%s/inverse_model_%s.pt' % (model_dir, step)
        )
        torch.save(
            self.transition_model.state_dict(), '%s/transition_model_%s.pt' % (model_dir, step)
        )

    def train(self, training=True):
        self.training = training
        self.encoder.train(training)
        self.inverse_model.train(training)
        self.transition_model.train(training)
    
    def forward(self, obs, action, next_obs, memory = None, next_memory = None, next_memory_mask = None):
        a_t = self.action_embedding(action)
        z_t, memory = self.encoder(obs, memory)
        z_t1, next_memory = self.encoder(next_obs, memory*next_memory_mask) # memory*next_memory_mask can also work?
        
        i = torch.cat((z_t, z_t1), dim=1)
        t = torch.cat((z_t, a_t), dim=1)
        
        i_logits = self.inverse_model(i)
        z_hat_t1 = self.transition_model(t)

        return i_logits, z_hat_t1, memory, next_memory, z_t, z_t1

    def log(self, logger, step):
        self.encoder.log(logger,step)
        self.inverse_model.log(logger,step)