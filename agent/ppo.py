import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from agent import Agent
from agent.pretraining import pebble
from agent.common.feature_extraction.autoencoder import AutoEncoder
from agent.common.transition_model import LatentMDPModel
from agent.common.actor_critic.actor_critic import ACModel, ACNModel

class PPO(Agent):
    """PPO algorithm."""
    def __init__(self, obs_dim, action_type, device, latent_dim, architecture, state_type,
                 agent_cfg, cfg, import_protocol, deploy_mode, mode=1, normalize_state_entropy=True):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_type = action_type
        self.device = device
        self.latent_dim = latent_dim
        self.architecture = architecture
        self.import_protocol = import_protocol
        self.deploy_mode = deploy_mode
        self.mode = mode

        self.state_type = state_type
        self.action_dim = agent_cfg.action_dim
        self.action_scale = (agent_cfg.action_range[1] - agent_cfg.action_range[0]) /2.0
        self.action_bias = (agent_cfg.action_range[1] + agent_cfg.action_range[0]) /2.0
        
        self.num_update_steps = cfg.agent.action_cfg.batch_size
        self.lr = cfg.agent.action_cfg.lr
        self.anneal_lr = cfg.agent.action_cfg.anneal_lr
        self.discount = cfg.agent.action_cfg.discount
        self.gae_lambda = cfg.agent.action_cfg.gae_lambda
        self.num_minibatches = cfg.agent.action_cfg.num_minibatches
        self.update_epochs = cfg.agent.action_cfg.update_epochs
        self.clip_coef = cfg.agent.action_cfg.clip_coef
        self.norm_adv = cfg.agent.action_cfg.norm_adv
        self.clip_vloss = cfg.agent.action_cfg.clip_vloss
        self.ent_coef = cfg.agent.action_cfg.ent_coef
        self.vf_coef = cfg.agent.action_cfg.vf_coef
        self.max_grad_norm = cfg.agent.action_cfg.max_grad_norm
        self.target_kl = None if cfg.agent.action_cfg.target_kl == 'None' else cfg.agent.action_cfg.target_kl
        
        self.s_ent_stats = pebble.TorchRunningMeanStd(shape=[1], device=self.device)
        self.normalize_state_entropy = normalize_state_entropy
        
        self.has_memory = False
        self.sequence_length = 1
        if 'LSTM' in self.architecture or 'GRU' in self.architecture:
            self.has_memory = True
            self.sequence_length = cfg.agent.action_cfg.sequence_length # CHECK

        self.minibatch_size =  int(self.num_update_steps // self.num_minibatches)
        # self.batch_size_of_sequences = int(self.minibatch_size // self.sequence_length)
        
        # online training
        if deploy_mode:
            self.acmodel = self._create_ACNModel()
            self.optimizer = torch.optim.Adam(self.acmodel.parameters(), lr=self.lr, eps=1e-08) # CHECK
        else:
            # self.autoencoder = self._create_AutoEncoder()
            # self.autoencoder_optimizer = torch.optim.Adam(self.autoencoder.parameters(), lr=self.lr) # CHECK     
            # self.autoencoder_loss_fn = nn.MSELoss()

            # pre-training
            self.latentMDP = self._create_LatentMDPModel()
            self.latentMDP_optimizer = torch.optim.Adam(self.latentMDP.parameters(), lr=self.lr) # CHECK
            
            if 'encoder_update_epochs' in cfg:
                self.encoder_update_epochs = cfg.encoder_update_epochs
                # online pretraining
                # AC model for policy learning
                self.acmodel = self._create_ACModel()
                self.optimizer = torch.optim.Adam(self.acmodel.parameters(), lr=self.lr, eps=1e-08) # CHECK

        # change mode
        self.train()
    
    def _create_ACModel(self):
        acnmodel = ACModel(obs_dim=self.latentMDP.embedding_size,
                          action_dim=self.action_dim,
                          action_type=self.action_type)
        acnmodel.double()
        acnmodel.to(self.device)
        return acnmodel

    def _create_ACNModel(self):
        acmodel = ACNModel(obs_dim=self.obs_dim,
                          action_dim=self.action_dim,
                          action_type=self.action_type,
                          latent_dim = self.latent_dim,
                          architecture=self.architecture,
                          mode=self.mode)
        acmodel.double()
        acmodel.to(self.device)
        return acmodel
    
    def _create_LatentMDPModel(self):
        model = LatentMDPModel(obs_dim=self.obs_dim,
                                    action_dim=self.action_dim,
                                    action_type=self.action_type,
                                    latent_dim = self.latent_dim,
                                    architecture=self.architecture,
                                    mode=self.mode)
        model.double()
        model.to(self.device)
        return model
    
    def _create_AutoEncoder(self):
        autoencoder = AutoEncoder(obs_dim=self.obs_dim,
                                action_dim=self.action_dim,
                                action_type=self.action_type,
                                latent_dim = self.latent_dim,
                                architecture=self.architecture,
                                mode=self.mode)
        autoencoder.double()
        autoencoder.to(self.device)
        return autoencoder
    
    @property
    def memory_size(self):
        if self.deploy_mode:
           return (self.acmodel.network.memory_size,)
        else:  
            # return (self.autoencoder.network.memory_size,)
            return (self.latentMDP.encoder.memory_size,)
        
    def reset_actor(self):
        # reset actor and critic
        self.acmodel.reset_actor()
        self.acmodel.double()
        self.acmodel.to(self.device)
    
    def reset_critic(self):
        # reset actor and critic
        self.acmodel.reset_critic()
        self.acmodel.double()
        self.acmodel.to(self.device)

    def reset_network(self):
        # reset network
        self.acmodel.reset_network()
        self.acmodel.double()
        self.acmodel.to(self.device)
    
    def reset_lmdp_network(self, reset_lr = True):  
        # reset lmdp network
        # pre-training
        self.latentMDP = self._create_LatentMDPModel()
        if reset_lr:
            self.latentMDP_optimizer = torch.optim.Adam(self.latentMDP.parameters(), lr=self.lr)
                
    def train(self, training=True):
        self.training = training
        if self.deploy_mode:
            self.acmodel.train(training=training)
        else:
            # self.autoencoder.train(training=training)
            self.latentMDP.train(training=training)

    def get_pretrain_action(self, obs, action = None, memory = None):
            # This WILL update the ENCODER on PPO update
            if self.has_memory:
                z_t, memory = self.latentMDP.encoder.forward(obs, memory)
            else:
                z_t, _ = self.latentMDP.encoder.forward(obs)

            if self.action_type == 'Continuous':
                (mean, log_std), state_value, _ = self.acmodel.forward(z_t)
                std = torch.exp(log_std)
                dist = torch.distributions.Normal(mean, std)
                if action == None:
                    action = dist.sample()
                return action, dist.log_prob(action).sum(1), dist.entropy().sum(1), state_value, memory
            
            elif self.action_type == 'Discrete':
                logits, state_value, _ = self.acmodel.forward(z_t)
                # Action is a flat integer
                dist = Categorical(logits=logits)
                if action == None:
                    action = dist.sample()

                return action, dist.log_prob(action), dist.entropy(), state_value, memory
        
    def get_action(self, obs, action = None, memory = None):
        if self.action_type == 'Continuous':
            (mean, log_std), state_value, memory = self.acmodel.forward(obs, memory)
            std = torch.exp(log_std)
            dist = torch.distributions.Normal(mean, std)
            if action == None:
                action = dist.sample()
            return action, dist.log_prob(action).sum(1), dist.entropy().sum(1), state_value, memory
        elif self.action_type == 'Discrete':
            logits, state_value, memory = self.acmodel.forward(obs, memory)
            # Action is a flat integer
            dist = Categorical(logits=logits)
            if action == None:
                action = dist.sample()

            return action, dist.log_prob(action), dist.entropy(), state_value, memory
    
    def save(self, model_dir, step, mode = "NORMAL"):
        if "OFFLINE" in mode:
            # self.autoencoder.save_model(model_dir, step)
            self.latentMDP.save_model(model_dir, step)
        elif "ONLINE" in mode:
            self.latentMDP.save_model(model_dir, step)
            self.acmodel.save_model(model_dir, step)
        else:
            self.acmodel.save_model(model_dir, step)
            
    def load(self, model_dir, step, mode = "NORMAL"):
        self.acmodel.load_encoder(model_dir, step)
        if "OFFLINE" not in mode:
            self.acmodel.load_actor_critic(model_dir, step)
        
    def freeze_models(self, mode = "NO"):
        print(mode)
        # Freeze feature extractor from encoder (cnn/mlp)
        if 'CNN' in mode or 'ResNet' in mode:
            for name, p in self.acmodel.network.cnn.named_parameters():
                p.requires_grad = False
                if 'PART' in mode and '7' in name:
                    p.requires_grad = True
                    print('Freeze ', name, p.requires_grad)
        # Freeze the entire encoder (cnn/mlp + lstm)
        elif 'ALL' in mode:
            for name, p in self.acmodel.network.named_parameters():
                p.requires_grad = False
        # Freeze specified module actor, critic etc.
        elif mode != "NO":
            model_name=getattr(self.acmodel, mode, None)
            if model_name is None:
                print(f"No model named '{model_name}' found.")
                return
            for p in model_name.parameters():
                p.requires_grad = False

    # Add Imitation Learning Options
    def offline_update(self, trajectory, logger, steps):
        batch_size = self.num_update_steps * 4 
        obs, actions, rewards, dones = trajectory
        not_dones_t = 1 - dones
        sequence_ids = list(range(0, steps-self.sequence_length,self.sequence_length))
        no_batches = int(len(sequence_ids) // batch_size)

        for _ in range(no_batches):
            sample_ids = np.random.choice(sequence_ids, batch_size, replace=False)
            batch_ids = np.repeat(sample_ids, self.sequence_length+1) + np.tile(np.arange(self.sequence_length+1), len(sample_ids))
            
            obs_t = obs[batch_ids]
            action_t = actions[batch_ids]
            # reward_t = rewards[batch_ids]
            not_done_t = not_dones_t[batch_ids]

            obs_t = torch.DoubleTensor(obs_t.reshape((self.sequence_length+1, batch_size) + tuple(self.obs_dim))).to(self.device)
            if self.action_type == 'Continuous':
                action_t = torch.DoubleTensor(action_t.reshape((self.sequence_length+1, batch_size)+ tuple(self.action_dim))).to(self.device)
            elif self.action_type == 'Discrete':
                action_t = torch.LongTensor(action_t.reshape((self.sequence_length+1, batch_size,1))).to(self.device)
            not_done_t = torch.DoubleTensor(not_done_t.reshape((self.sequence_length+1, batch_size, 1))).to(self.device)
            memories = torch.zeros((self.sequence_length+1, batch_size, self.memory_size[0]),dtype=torch.float64).to(self.device)

            batch_loss = 0
            for i in range(0, self.sequence_length):
                obs_tensor = obs_t[i]
                action_tensor = action_t[i].squeeze(1)
                next_obs_tensor = obs_t[i+1]
                
                if self.has_memory:
                    memory_tensor = memories[i]
                    mask_tensor = not_done_t[i]
                    next_memory_tensor = memories[i+1]
                    next_mask_tensor = not_done_t[i+1]
                    # prediction_obs, _, memory  = self.autoencoder(obs_tensor, memory_tensor * mask_tensor)
                    pred_action_logprobs, z_hat_t1, memory, next_memory, z_t, z_t1  = self.latentMDP(obs_tensor, action_tensor, next_obs_tensor, 
                                                                              memory_tensor * mask_tensor, next_mask_tensor)
                else:
                    # prediction_obs, _, memory,  = self.autoencoder(obs_tensor)
                    pred_action_logprobs, z_hat_t1, _, _, z_t, z_t1  = self.latentMDP(obs_tensor, action_tensor, next_obs_tensor)

                if self.has_memory and i < self.sequence_length-1:
                    memories[i + 1] = memory.detach()
                    # memories[i + 2] = next_memory.detach() # for MDP modelling

                # Best so far.
                action_loss = self.latentMDP.cross_entropy_loss(pred_action_logprobs, action_tensor)

                # contrastive_loss = -1 * self.latentMDP.wasserstein_loss(z_a, z_b) # Bad result
                # contrastive_loss = self.latentMDP.contrastive_loss(obs_tensor, z_t, temporal_window=2)

                # transition_loss = self.latentMDP.mse_loss(z_hat_t1, z_t1) # Decent alone, not good with action loss
                # transition_loss = self.latentMDP.wasserstein_loss(z_hat_t1, z_t1) # Not good enough
                
                # locality_loss = self.latentMDP.l1_loss(z_t, z_t1) # Big performance drop
                # locality_loss = self.latentMDP.wasserstein_loss(z_t, z_t1) # Performance drop

                # Bisimilarity loss
                loss = 1.0 * action_loss #+ 0.2 * contrastive_loss
                
                # loss = self.autoencoder_loss_fn(prediction_obs, obs_tensor)

                batch_loss += loss

            batch_loss = batch_loss/self.sequence_length
            
            # self.autoencoder_optimizer.zero_grad()
            # batch_loss.backward()
            # self.autoencoder_optimizer.step()

            self.latentMDP_optimizer.zero_grad()
            batch_loss.backward()
            self.latentMDP_optimizer.step()

        # consider only the last batch for reporting the final loss for each epoch
        epoch_loss = batch_loss.item()
        
        epoch_loss /= no_batches
        # logger.log('train_autoencoder/loss', epoch_loss, steps)
        logger.log('train_lmdp/loss', epoch_loss, steps)
        return epoch_loss

    def get_lmdp_loss(self, trajectory, next, diff=False):   
        obs, actions, _, _, _, dones, memories, next_obs, next_done, next_memory = self.to_tensor(trajectory, next)

        # extend obs, dones, memories with next_obs, next_done, next_memory
        obs = torch.cat((obs,next_obs.unsqueeze(0)), dim=0)
        dones = torch.cat((dones,next_done), dim=0)
        memories = torch.cat((memories,next_memory.unsqueeze(0)), dim=0)
        
        batch_size = self.num_update_steps
        if diff:
            effective_sequence_length = self.sequence_length + 2
            sequence_ids = list(range(0, batch_size-effective_sequence_length))
        else:
            effective_sequence_length = self.sequence_length + 1
            sequence_ids = list(range(0, batch_size-effective_sequence_length))
        # Reshape data by sequence size first, batch size second
        batch_ids = np.repeat(sequence_ids, effective_sequence_length) + np.tile(np.arange(effective_sequence_length), len(sequence_ids))

        # Create the padding patterns
        i = np.arange(effective_sequence_length)
        mask = np.arange(effective_sequence_length) >= (effective_sequence_length - i-1)[:, None]
        init_pad = np.where(mask, np.arange(effective_sequence_length) - (effective_sequence_length - i-1)[:, None], 0)

        # Combine padding patterns with the original batch (excluding the overlapping parts)
        batch_ids = np.concatenate([np.array(init_pad).flatten(), batch_ids])
        # [print(batch_ids[i:i+effective_sequence_length]) for i in range(0, len(batch_ids), effective_sequence_length)]

        obs_t = obs[batch_ids]
        actions_t = actions[batch_ids]
        dones_t = dones[batch_ids]
        memories_t = memories[batch_ids]

        # Flatten and reshape the data
        b_obs, b_actions, _, _, b_dones, b_memories, _, _ = self.reshape_batch(obs_t, actions_t, None, None, dones_t, memories_t, 
                                                                               sequence_length=effective_sequence_length,
                                                                               minibatch_size=batch_size)

        prev_batch_loss = 0 if diff else None
        batch_loss = 0
        for i in range(0, self.sequence_length):
            if diff:
                prev_obs_tensor = b_obs[i]
                prev_action_tensor = b_actions[i].squeeze(1)
            obs_tensor = b_obs[i+1] if diff else b_obs[i]
            action_tensor = b_actions[i+1].squeeze(1) if diff else b_actions[i].squeeze(1)
            next_obs_tensor = b_obs[i+2] if diff else b_obs[i+1]

            with torch.no_grad():
                if self.has_memory:
                    memory_tensor = b_memories[i+1] if diff else b_memories[i]
                    mask_tensor = (1-b_dones[i+1]).to(self.device) if diff else (1-b_dones[i]).to(self.device)
                    next_memory_tensor = b_memories[i+2] if diff else b_memories[i+1]
                    next_mask_tensor = (1-b_dones[i+2]).to(self.device) if diff else (1-b_dones[i+1]).to(self.device)

                    if diff:
                        prev_memory_tensor = b_memories[i]
                        prev_mask_tensor = (1-b_dones[i]).to(self.device)
                        prev_pred_action_logprobs, prev_z_hat_t1, prev_memory, _, prev_z_t, prev_z_t1  = self.latentMDP(
                                                                            prev_obs_tensor, prev_action_tensor, obs_tensor, 
                                                                            prev_memory_tensor * prev_mask_tensor, mask_tensor)

                    pred_action_logprobs, z_hat_t1, memory, next_memory, z_t, z_t1  = self.latentMDP(obs_tensor, action_tensor, next_obs_tensor, 
                                                                            memory_tensor * mask_tensor, next_mask_tensor)
                else:
                    if diff:
                        prev_pred_action_logprobs, prev_z_hat_t1, _, _, prev_z_t, prev_z_t1  = self.latentMDP(
                                                                            prev_obs_tensor, prev_action_tensor, obs_tensor)
                    pred_action_logprobs, z_hat_t1, _, _, z_t, z_t1  = self.latentMDP(obs_tensor, action_tensor, next_obs_tensor)

            if self.has_memory and i < self.sequence_length-1:
                b_memories[i + 1] = memory.detach()
                b_memories[i + 2] = next_memory.detach() # for MDP modelling
            
            if diff:
                prev_loss = self.lmdp_loss(prev_pred_action_logprobs, prev_action_tensor, prev_obs_tensor, prev_z_t, prev_z_t1, eval=True)
                prev_batch_loss += prev_loss
            loss = self.lmdp_loss(pred_action_logprobs, action_tensor, obs_tensor, z_t, z_t1)
            batch_loss += loss
        batch_loss = batch_loss/self.sequence_length
        if diff:
            prev_batch_loss = prev_batch_loss/self.sequence_length
            return (batch_loss.detach().cpu(), prev_batch_loss.detach().cpu())
        return batch_loss.detach().cpu()

    def lmdp_loss(self, pred_action_logprobs, action_tensor, obs_tensor, z_t, z_t1, eval=False):
        # Best so far.
        if eval:
            action_loss = self.latentMDP.eval_cross_entropy(pred_action_logprobs, action_tensor)
        else:
            action_loss = self.latentMDP.cross_entropy_loss(pred_action_logprobs, action_tensor)
        
        # z_a = z_t1[:self.minibatch_size//2]
        # z_b = z_t1[self.minibatch_size//2:]
        
        contrastive_loss = self.latentMDP.contrastive_loss(obs_tensor, z_t, temporal_window=2)

        # contrastive_loss = self.latentMDP.hinge_loss(z_a, z_b) # not any effect, at least no harm
        # contrastive_loss = -1 * self.latentMDP.wasserstein_loss(z_a, z_b) # Bad result

        # transition_loss = self.latentMDP.mse_loss(z_hat_t1, z_t1) # Decent alone, not good with action loss
        # transition_loss = self.latentMDP.wasserstein_loss(z_hat_t1, z_t1) # Not good enough

        # locality_loss = self.latentMDP.l1_loss(z_t, z_t1) # Big performance drop
        # locality_loss = self.latentMDP.wasserstein_loss(z_t, z_t1) # Performance drop
        # locality_loss = self.latentMDP.hinge_loss(z_t, z_t1)
        
        # Bisimilarity loss
        loss = 1.0 * action_loss + 0.05 * contrastive_loss
        return loss
    
    def update_encoder(self, trajectory, next, logger, step):    
        obs, actions, _, _, rewards, dones, memories, next_obs, next_done, next_memory = self.to_tensor(trajectory, next)

        # extend obs, dones, memories with next_obs, next_done, next_memory
        obs = torch.cat((obs,next_obs.unsqueeze(0)), dim=0)
        dones = torch.cat((dones,next_done), dim=0)
        memories = torch.cat((memories,next_memory.unsqueeze(0)), dim=0)
        
        batch_size = self.num_update_steps
        effective_sequence_length = self.sequence_length + 1
        sequence_ids = list(range(0, batch_size-effective_sequence_length))
        no_minibatches = max(int(len(sequence_ids) // self.minibatch_size),1)

        for epoch in range(self.encoder_update_epochs):
            # print('Epoch ', epoch)
            print_flag = False
            if epoch == self.encoder_update_epochs -1:
                print_flag = True
            for _ in range(no_minibatches):
                # Suffle sequence ids
                sample_ids = np.random.choice(sequence_ids, self.minibatch_size, replace=False)
                
                # Reshape data by sequence size first, batch size second
                batch_ids = np.repeat(sample_ids, effective_sequence_length) + np.tile(np.arange(effective_sequence_length), len(sample_ids))

                obs_t = obs[batch_ids]
                actions_t = actions[batch_ids]
                dones_t = dones[batch_ids]
                memories_t = memories[batch_ids]

                # Flatten and reshape the data
                b_obs, b_actions, _, _, b_dones, b_memories, _, _ = self.reshape_batch(obs_t, actions_t, None, None, dones_t, memories_t, 
                                                                                       sequence_length=effective_sequence_length, 
                                                                                       minibatch_size=self.minibatch_size)

                batch_loss = 0
                for i in range(0, self.sequence_length):
                    obs_tensor = b_obs[i]
                    action_tensor = b_actions[i].squeeze(1)
                    next_obs_tensor = b_obs[i+1]

                    if self.has_memory:
                        memory_tensor = b_memories[i]
                        mask_tensor = (1-b_dones[i]).to(self.device)
                        next_memory_tensor = b_memories[i+1]
                        next_mask_tensor = (1-b_dones[i+1]).to(self.device)

                        pred_action_logprobs, z_hat_t1, memory, next_memory, z_t, z_t1  = self.latentMDP(obs_tensor, action_tensor, next_obs_tensor, 
                                                                                memory_tensor * mask_tensor, next_mask_tensor)
                    else:
                        pred_action_logprobs, z_hat_t1, _, _, z_t, z_t1  = self.latentMDP(obs_tensor, action_tensor, next_obs_tensor)

                    if self.has_memory and i < self.sequence_length-1:
                        b_memories[i + 1] = memory.detach()
                        b_memories[i + 2] = next_memory.detach() # for MDP modelling

                    loss = self.lmdp_loss(pred_action_logprobs, action_tensor, obs_tensor, z_t, z_t1)                    
                    
                    batch_loss += loss
                batch_loss = batch_loss/self.sequence_length

                self.latentMDP_optimizer.zero_grad()
                batch_loss.backward()
                self.latentMDP_optimizer.step()

                # Use action to take the suitable Q value
                if print_flag:
                    logger.log('train_encoder/loss', batch_loss, step)

                self.latentMDP.log(logger, step)
                
        logger.log('train_encoder/learning_rate', self.optimizer.param_groups[0]["lr"], step)
        return self.latentMDP.state_dict()
    
    # An Intrinsic Reward is fed to the PPO update
    # Consider Auxilliery Losses
    # Add support for multiple actors
    def pretrain_update(self, trajectory, next, logger, step):
        obs, actions, logprobs, values, rewards, dones, memories, next_obs, next_done, next_memory = self.to_tensor(trajectory, next)
        
        with torch.no_grad():
            if self.has_memory:
                next_mask = 1-next_done
                z_next, _ = self.latentMDP.encoder.forward(next_obs, next_memory*next_mask)
            else:
                z_next, _ = self.latentMDP.encoder.forward(next_obs)

        # Calculate Advantages and Expected Returns        
        advantages, returns = self.calculate_advantage_and_returns(values, dones, rewards, z_next, next_memory, next_done, next_mask)

        batch_size = self.num_update_steps
        effective_sequence_length = self.sequence_length + 1
        sequence_ids = list(range(0, batch_size-effective_sequence_length))
        no_minibatches = max(int(len(sequence_ids) // self.minibatch_size), 1)

        self.clipfracs = []
        for epoch in range(self.update_epochs):
            # print('Epoch ', epoch)
            print_flag = False
            if epoch == self.update_epochs -1:
                logger.log('train/batch_reward', rewards.mean(), step)
                print_flag = True
            
            for _ in range(no_minibatches):
                # Suffle sequence ids
                sample_ids = np.random.choice(sequence_ids, self.minibatch_size, replace=False)
                
                # Reshape data by sequence size first, batch size second
                batch_ids = np.repeat(sample_ids, effective_sequence_length) + np.tile(np.arange(effective_sequence_length), len(sample_ids))

                obs_t = obs[batch_ids]
                actions_t = actions[batch_ids]
                logprobs_t = logprobs[batch_ids]
                values_t = values[batch_ids]
                dones_t = dones[batch_ids]
                memories_t = memories[batch_ids]
                advantages_t = advantages[batch_ids]
                returns_t = returns[batch_ids]
                
                # Flatten and reshape the data
                b_obs, b_actions, b_logprobs, b_values, b_dones, b_memories, b_advantages, b_returns = self.reshape_batch(obs_t, actions_t, 
                                                                        logprobs_t, values_t, dones_t, memories_t, advantages_t, returns_t, 
                                                                        sequence_length=effective_sequence_length, 
                                                                        minibatch_size=self.minibatch_size)
                
                batch_entropy = 0
                batch_value = 0
                batch_actor_loss = 0
                batch_critic_loss = 0
                batch_loss = 0

                for i in range(0, self.sequence_length):
                    obs_tensor = b_obs[i]
                    if self.has_memory:
                        memory_tensor = b_memories[i]
                        mask_tensor = (1-b_dones[i]).to(self.device)
                        
                        _, newlogprob, entropy, newvalue, memory = self.get_pretrain_action(obs_tensor, b_actions[i], memory_tensor * mask_tensor)
                    else:
                        obs_tensor, _ = self.latentMDP.encoder.forward(obs_tensor, next_memory*next_mask)
                        _, newlogprob, entropy, newvalue, _ = self.get_pretrain_action(obs_tensor, b_actions[i])
                    logratio = newlogprob - b_logprobs[i]
                    ratio = logratio.exp()

                    with torch.no_grad():
                        # calculate approx_kl http://joschu.net/blog/kl-approx.html
                        old_approx_kl = (-logratio).mean()
                        approx_kl = ((ratio - 1) - logratio).mean()
                        self.clipfracs += [((ratio - 1.0).abs() > self.clip_coef).float().mean().item()]
                    
                    actor_loss, critic_loss = self.policy_and_value_update(b_advantages[i], b_values[i], b_returns[i], ratio, newvalue)

                    entropy_loss = entropy.mean()
                    loss = actor_loss - self.ent_coef * entropy_loss + critic_loss * self.vf_coef

                    # print(f'  i: {i}, Loss: {loss}, Actor Loss: {actor_loss}, Critic Loss: {critic_loss}, Entropy Loss: {entropy_loss}')
                    # Update batch values
                    batch_entropy += entropy_loss.item()
                    batch_value += newvalue.mean().item()
                    batch_actor_loss += actor_loss.item()
                    batch_critic_loss += critic_loss.item()
                    batch_loss += loss

                    if self.has_memory and i < self.sequence_length-1:
                        b_memories[i + 1] = memory.detach()

                # print(f'Total Loss: {batch_loss}')

                # Update batch values
                batch_entropy /= self.sequence_length
                batch_value /= self.sequence_length
                batch_actor_loss /= self.sequence_length
                batch_critic_loss /= self.sequence_length
                batch_loss /= self.sequence_length
                
                # print(f'Batch Loss: {batch_loss}')
                # Update actor-critic
                self.optimizer.zero_grad()
                batch_loss.backward()
                grad_norm = sum(p.grad.data.norm(2).item() ** 2 for p in self.acmodel.parameters() if p.requires_grad == True) ** 0.5
                torch.nn.utils.clip_grad_norm_(self.acmodel.parameters(), self.max_grad_norm)
                self.optimizer.step()

                # Use action to take the suitable Q value
                if print_flag:
                    logger.log('train_critic/loss', batch_critic_loss, step)
                    logger.log('train_actor/loss', batch_actor_loss, step)
                    logger.log('train_actor/entropy', batch_entropy, step)
                    logger.log('train/gradnorm', grad_norm, step)
                    logger.log('train/batchvalue', batch_value, step)

                self.acmodel.log(logger, step)
                
            if self.target_kl is not None and approx_kl > self.target_kl:
                break
        
        y_pred, y_true = b_values.detach().cpu().numpy(), b_returns.detach().cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        logger.log('train/explained_variance', explained_var, step)
        logger.log('train/learning_rate', self.optimizer.param_groups[0]["lr"], step)
        logger.log('train/approx_kl', approx_kl, step)
        logger.log('train/old_approx_kl', old_approx_kl, step)
        logger.log('train/clipfrac', np.mean(self.clipfracs), step)
 
    def update(self, trajectory, next, logger, step):
        obs, actions, logprobs, values, rewards, dones, memories, next_obs, next_done, next_memory = self.to_tensor(trajectory, next)

        if self.has_memory:
                next_mask = 1-next_done

        # Calculate Advantages and Expected Returns        
        advantages, returns = self.calculate_advantage_and_returns(values, dones, rewards, next_obs, next_memory, next_done, next_mask)

        batch_size = self.num_update_steps
        effective_sequence_length = self.sequence_length + 1
        sequence_ids = list(range(0, batch_size-effective_sequence_length))
        no_minibatches = max(int(len(sequence_ids) // self.minibatch_size), 1)

        self.clipfracs = []
        for epoch in range(self.update_epochs):
            # print('Epoch ', epoch)
            print_flag = False
            if epoch == self.update_epochs -1:
                logger.log('train/batch_reward', rewards.mean(), step)
                print_flag = True
            
            for _ in range(no_minibatches):
                # Suffle sequence ids
                sample_ids = np.random.choice(sequence_ids, self.minibatch_size, replace=False)
                
                # Reshape data by sequence size first, batch size second
                batch_ids = np.repeat(sample_ids, effective_sequence_length) + np.tile(np.arange(effective_sequence_length), len(sample_ids))

                obs_t = obs[batch_ids]
                actions_t = actions[batch_ids]
                logprobs_t = logprobs[batch_ids]
                values_t = values[batch_ids]
                dones_t = dones[batch_ids]
                memories_t = memories[batch_ids]
                advantages_t = advantages[batch_ids]
                returns_t = returns[batch_ids]
                
                # Flatten and reshape the data
                b_obs, b_actions, b_logprobs, b_values, b_dones, b_memories, b_advantages, b_returns = self.reshape_batch(obs_t, actions_t, 
                                                                        logprobs_t, values_t, dones_t, memories_t, advantages_t, returns_t, 
                                                                        sequence_length=effective_sequence_length, 
                                                                        minibatch_size=self.minibatch_size)
                
                batch_entropy = 0
                batch_value = 0
                batch_actor_loss = 0
                batch_critic_loss = 0
                batch_loss = 0
        
                for i in range(0, self.sequence_length):
                    obs_tensor = b_obs[i]
                    if self.has_memory:
                        memory_tensor = b_memories[i]
                        mask_tensor = (1-b_dones[i]).to(self.device)

                        _, newlogprob, entropy, newvalue, memory = self.get_action(obs_tensor, b_actions.long()[i], memory_tensor * mask_tensor)
                    else:
                        _, newlogprob, entropy, newvalue, _ = self.get_action(obs_tensor, b_actions.long()[i])
                    logratio = newlogprob - b_logprobs[i]
                    ratio = logratio.exp()

                    with torch.no_grad():
                        # calculate approx_kl http://joschu.net/blog/kl-approx.html
                        old_approx_kl = (-logratio).mean()
                        approx_kl = ((ratio - 1) - logratio).mean()
                        self.clipfracs += [((ratio - 1.0).abs() > self.clip_coef).float().mean().item()]

                    actor_loss, critic_loss = self.policy_and_value_update(b_advantages[i], b_values[i], b_returns[i], ratio, newvalue)

                    
                    entropy_loss = entropy.mean()
                    loss = actor_loss - self.ent_coef * entropy_loss + critic_loss * self.vf_coef

                    # print(f'  i: {i}, Loss: {loss}, Actor Loss: {actor_loss}, Critic Loss: {critic_loss}, Entropy Loss: {entropy_loss}')
                    # Update batch values
                    batch_entropy += entropy_loss.item()
                    batch_value += newvalue.mean().item()
                    batch_actor_loss += actor_loss.item()
                    batch_critic_loss += critic_loss.item()
                    batch_loss += loss

                    if self.has_memory and i < self.sequence_length-1:
                        b_memories[i + 1] = memory.detach()

                # Update batch values
                batch_entropy /= self.sequence_length
                batch_value /= self.sequence_length
                batch_actor_loss /= self.sequence_length
                batch_critic_loss /= self.sequence_length
                batch_loss /= self.sequence_length
                
                # print(f'Batch Loss: {batch_loss}')
                # Update actor-critic
                self.optimizer.zero_grad()
                batch_loss.backward()
                grad_norm = sum(p.grad.data.norm(2).item() ** 2 for p in self.acmodel.parameters() if p.requires_grad == True) ** 0.5
                torch.nn.utils.clip_grad_norm_(self.acmodel.parameters(), self.max_grad_norm)
                self.optimizer.step()

                # Use action to take the suitable Q value
                if print_flag:
                    logger.log('train_critic/loss', critic_loss, step)
                    logger.log('train_actor/loss', actor_loss, step)
                    logger.log('train_actor/entropy', entropy_loss, step)
                    logger.log('train/gradnorm', grad_norm, step)
                    logger.log('train/batchvalue', batch_value, step)

                self.acmodel.log(logger, step)
                
            if self.target_kl is not None and approx_kl > self.target_kl:
                break
        
        y_pred, y_true = b_values.detach().cpu().numpy(), b_returns.detach().cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        logger.log('train/explained_variance', explained_var, step)
        logger.log('train/learning_rate', self.optimizer.param_groups[0]["lr"], step)
        logger.log('train/approx_kl', approx_kl, step)
        logger.log('train/old_approx_kl', old_approx_kl, step)
        logger.log('train/clipfrac', np.mean(self.clipfracs), step)

    def calculate_advantage_and_returns(self, values, dones, rewards, next_obs, next_memory, next_done, next_mask):
        # Calculate Advantages and Expected Returns
        with torch.no_grad():
            if self.has_memory:
                _, next_value, _ = self.acmodel(next_obs, next_memory * next_mask)
            else:
                _, next_value, _ = self.acmodel(next_obs)
                
            advantages = torch.zeros_like(rewards).to(self.device)
            lastgaelam = 0

            for t in reversed(range(self.num_update_steps)):
                if t == self.num_update_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]

                delta = rewards[t] + self.discount * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + self.discount * self.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values
        return advantages, returns
    
    def policy_and_value_update(self, b_advantages, b_values, b_returns, ratio, newvalue):
        mb_advantages = b_advantages
        if self.norm_adv: # Use on vector envs
            mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

        # Policy loss
        pg_loss1 = -mb_advantages * ratio
        pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef)
        pg_loss = torch.max(pg_loss1, pg_loss2).mean()

        # Value loss
        # newvalue = newvalue.view(-1) $ Not needed when processing batches
        if self.clip_vloss:
            v_loss_unclipped = (newvalue - b_returns) ** 2
            v_clipped = b_values + torch.clamp(newvalue - b_values, -self.clip_coef, self.clip_coef)
            v_loss_clipped = (v_clipped - b_returns) ** 2
            v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
            value_loss = 0.5 * v_loss_max.mean()
        else:
            value_loss = 0.5 * ((newvalue - b_returns) ** 2).mean()

        return pg_loss, value_loss

    def to_tensor(self, trajectory, next):
        obs, actions, logprobs, values, rewards, dones, memories = trajectory
        next_obs, next_done, next_memory = next 

        obs = torch.DoubleTensor(obs).to(self.device)
        if self.action_type == 'Continuous':
            actions = torch.DoubleTensor(actions).to(self.device)
        elif self.action_type == 'Discrete':
            actions = torch.LongTensor(actions).to(self.device)
        logprobs = torch.DoubleTensor(logprobs).to(self.device)
        values = torch.DoubleTensor(values).to(self.device)
        rewards = torch.DoubleTensor(rewards).to(self.device)
        dones = torch.DoubleTensor(dones).to(self.device)
        if self.has_memory: 
            memories = torch.DoubleTensor(memories).to(self.device)
        
        next_obs = torch.DoubleTensor(next_obs).to(self.device).unsqueeze(0)      
        
        if self.has_memory:
            next_memory = torch.DoubleTensor(next_memory).to(self.device).unsqueeze(0)
            next_done = torch.DoubleTensor([next_done]).to(self.device).unsqueeze(0)

        return obs, actions, logprobs, values, rewards, dones, memories, next_obs, next_done, next_memory
    
    def flatten_batch(self, obs, actions, logprobs, values, dones, memories=None, advantages=None, returns=None):
        # flatten the batch
        b_obs = obs.reshape((-1,) + tuple(self.obs_dim))
        b_logprobs = logprobs.reshape(-1)
        if self.action_type == 'Continuous':
            b_actions = actions.reshape((-1,) + tuple(self.action_dim))
        elif self.action_type == 'Discrete':
            b_actions = actions.reshape((-1,1))
        b_advantages = advantages.reshape(-1) if advantages !=None else None
        b_returns = returns.reshape(-1) if returns !=None else None
        b_dones = dones.reshape(-1)
        b_values = values.reshape(-1)
        b_memories = memories.reshape((-1,) + tuple(self.memory_size))  if memories !=None else None
        
        return b_obs, b_actions, b_logprobs, b_values, b_dones, b_memories, b_advantages, b_returns
    
    def reshape_batch(self, obs, actions, logprobs, values, dones, memories=None, advantages=None, returns=None, sequence_length=None, minibatch_size=None):
        # reshape the batch with sequence length first
        b_obs = obs.reshape((sequence_length, minibatch_size) + tuple(self.obs_dim))
        b_logprobs = logprobs.reshape((sequence_length, minibatch_size, -1)) if logprobs !=None else None
        if self.action_type == 'Continuous':
            b_actions = actions.reshape((sequence_length, minibatch_size) + tuple(self.action_dim))
        elif self.action_type == 'Discrete':
            b_actions = actions.reshape((sequence_length, minibatch_size,1))
        b_advantages = advantages.reshape((sequence_length, minibatch_size, -1)) if advantages !=None else None
        b_returns = returns.reshape((sequence_length, minibatch_size, -1)) if returns !=None else None
        b_dones = dones.reshape((sequence_length, minibatch_size, -1))
        b_values = values.reshape((sequence_length, minibatch_size, -1)) if values !=None else None
        b_memories = memories.reshape((sequence_length, minibatch_size, self.memory_size[0]))  if memories !=None else None
        
        return b_obs, b_actions, b_logprobs, b_values, b_dones, b_memories, b_advantages, b_returns