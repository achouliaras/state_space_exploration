import torch
import torch.nn as nn
from agent.common.feature_extraction.encoder import Encoder, Decoder

class AutoEncoder(nn.Module):
    def __init__(self, obs_dim, action_dim, action_type, latent_dim, architecture, mode):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.action_type = action_type
        self.latent_dim = latent_dim
        self.architecture = architecture
        self.mode = mode

        self.network = Encoder(obs_shape=self.obs_dim,
                          latent_dim=self.latent_dim,
                          architecture=self.architecture,
                          mode=self.mode)
        
        self.decoder = Decoder(obs_shape=self.obs_dim,
                          latent_dim=self.latent_dim,
                          architecture=self.architecture,
                          mode=mode)
    
    def train(self, training=True):
        self.training = training
        self.network.train(training)
        self.decoder.train(training)

    def save_model(self, model_dir, step):
        torch.save(
            self.network.state_dict(), '%s/network_%s.pt' % (model_dir, step)
        )
        torch.save(
            self.decoder.state_dict(), '%s/decoder_%s.pt' % (model_dir, step)
        )
     
    def forward(self, obs, memory = None):
        x, memory = self.network(obs, memory)
        prediction_obs = self.decoder(x)
        return prediction_obs, x, memory
    
    def log(self, logger, step):
        self.network.log(logger,step)
        self.decoder.log(logger,step)