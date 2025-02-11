import torch
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

        self.network = self.create_network()
        self.action_model = self.create_actor()

    def create_network(self):
        # CNN, MLP, LSTM
        network = Encoder(obs_shape=self.obs_dim,
                          latent_dim=self.latent_dim,
                          architecture=self.architecture,
                          mode=self.mode)
        return network
    
    def create_actor(self):
        if self.action_type == 'Continuous':
            actor = SimpleActor(input_dim=self.obs_dim*2, 
                                output_dim=self.action_dim,
                                action_type=self.action_type,
                                hidden_depth=1,
                                hidden_dim=64
                                )
        elif self.action_type == 'Discrete':
            actor = SimpleActor(input_dim=self.network.embedding_size*2,
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
        self.action_model = self.create_actor()
    
    def reset_network(self):
        # reset network
        self.network = self.create_network()

    def train(self, training=True):
        self.training = training
        self.network.train(training)
        self.action_model.train(training)
    
    def forward(self, obs, next_obs, memory = None, next_memory_mask = None):
        if self.action_type == 'Continuous':
            x_t = obs
            x_t1 = next_obs
        elif self.action_type == 'Discrete':
            x_t, memory = self.network(obs, memory)
            x_t1, next_memory = self.network(next_obs, memory*next_memory_mask)
        x = torch.cat((x_t, x_t1), dim=1)
        logits = self.action_model(x)
        return logits, memory, next_memory

    def log(self, logger, step):
        self.network.log(logger,step)
        self.action_model.log(logger,step)