import torch
from torch import nn
import torch.nn.functional as F
from agent.common.actor_critic.critic_only import SimpleCritic
from agent.common.actor_critic.actor_only import SimpleActor
from agent.common.feature_extraction.encoder import Encoder
from geomloss import SamplesLoss

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
            self.inverse_model = SimpleActor(input_dim=self.obs_dim*2, 
                                                output_dim=self.action_dim,
                                                action_type=self.action_type,
                                                hidden_depth=2,
                                                hidden_dim=64
                                            )
        elif self.action_type == 'Discrete':
            self.action_embedding = nn.Embedding(action_dim,self.action_embedding_dim)
            self.inverse_model = SimpleActor(input_dim=self.encoder.embedding_size*2,
                                                output_dim=self.action_dim,
                                                action_type=self.action_type,
                                                hidden_depth=1,
                                                hidden_dim=64
                                            )

        self.transition_model = self.create_actor(input_dim=self.encoder.embedding_size+self.action_embedding_dim,
                                                  output_dim=self.encoder.embedding_size)

        self.mse_loss = nn.MSELoss()
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.eval_cross_entropy = nn.CrossEntropyLoss(reduction='none')
        self.wasserstein_loss = SamplesLoss(loss='sinkhorn', p=2, blur=0.05)
        self.l1_loss = nn.L1Loss()

    def contrastive_loss(self, states, embeddings, temporal_window=1, temperature = 0.1):
        # Encode states
        embeddings = F.normalize(embeddings, p=2, dim=1)  # Normalize embeddings

        # Positive pairs: states within temporal_window
        pos_pairs = []
        for t in range(len(states) - temporal_window):
            pos_pairs.append((embeddings[t], embeddings[t + 1]))  # Consecutive states

        # Negative pairs: states outside temporal_window
        neg_pairs = []
        for t in range(len(states)):
            neg_indices = [i for i in range(len(states)) if abs(i - t) > temporal_window]
            neg_pairs.extend([(embeddings[t], embeddings[i]) for i in neg_indices])

        # Compute similarity for positive and negative pairs
        pos_sim = torch.stack([torch.dot(z1, z2) for z1, z2 in pos_pairs]) / temperature
        neg_sim = torch.stack([torch.dot(z1, z2) for z1, z2 in neg_pairs]) / temperature

        # Contrastive loss (InfoNCE)
        numerator = torch.exp(pos_sim)
        denominator = numerator + torch.sum(torch.exp(neg_sim))
        loss = -torch.log(numerator / denominator).mean()

        return loss

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
    
    def forward(self, obs, action, next_obs, memory = None, next_memory_mask = None):
        a_t = self.action_embedding(action).squeeze(1)
        z_t, memory = self.encoder(obs, memory)
        next_memory = memory if next_memory_mask is None else memory * next_memory_mask
        z_t1, next_memory = self.encoder(next_obs, next_memory)
        
        i = torch.cat((z_t, z_t1), dim=1)
        t = torch.cat((z_t, a_t), dim=1)
        
        i_logits = self.inverse_model(i)
        z_hat_t1 = self.transition_model(t)

        return i_logits, z_hat_t1, memory, next_memory, z_t, z_t1

    def log(self, logger, step):
        self.encoder.log(logger,step)
        self.inverse_model.log(logger,step)
        self.transition_model.log(logger,step)