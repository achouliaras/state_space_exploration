import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from agent import Agent
from agent.pretraining import pebble
from agent.common.encoder import Encoder, Decoder
from agent.common.critic import DoubleQCritic, SimpleCritic
from agent.common.actor import DiagGaussianActor, CategoricalActor, SimpleActor

class ACModel(nn.Module):
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
    def reset_actor(self):
        # reset actor and critic
        self.actor = self.create_actor()
    
    def reset_critic(self):
        # reset actor and critic
    
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

class Autoencoder(nn.Module):
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

    def forward(self, obs, memory = None):
        x, memory = self.network(obs, memory)
        prediction_obs = self.decoder(x)
        return prediction_obs, x, memory
    
    def log(self, logger, step):
        self.network.log(logger,step)
        self.decoder.log(logger,step)

class PPO(Agent):
    """PPO algorithm."""
    def __init__(self, obs_dim, action_type, device, latent_dim, architecture, state_type,
                 agent_cfg, action_cfg, test, mode=1, normalize_state_entropy=True):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_type = action_type
        self.device = device
        self.latent_dim = latent_dim
        self.architecture = architecture
        self.test = test

        self.state_type = state_type
        self.action_dim = agent_cfg.action_dim
        self.action_scale = (agent_cfg.action_range[1] - agent_cfg.action_range[0]) /2.0
        self.action_bias = (agent_cfg.action_range[1] + agent_cfg.action_range[0]) /2.0
        
        self.num_update_steps = action_cfg.num_update_steps
        self.batch_size = action_cfg.batch_size
        self.lr = action_cfg.lr
        self.anneal_lr = action_cfg.anneal_lr
        self.discount = action_cfg.discount
        self.gae_lambda = action_cfg.gae_lambda
        self.num_minibatches = action_cfg.num_minibatches
        self.update_epochs = action_cfg.update_epochs
        self.clip_coef = action_cfg.clip_coef
        self.norm_adv = action_cfg.norm_adv
        self.clip_vloss = action_cfg.clip_vloss
        self.ent_coef = action_cfg.ent_coef
        self.vf_coef = action_cfg.vf_coef
        self.max_grad_norm = action_cfg.max_grad_norm
        self.target_kl = None if action_cfg.target_kl == 'None' else action_cfg.target_kl
        
        self.s_ent_stats = pebble.TorchRunningMeanStd(shape=[1], device=self.device)
        self.normalize_state_entropy = normalize_state_entropy
        
        self.has_memory = False
        self.sequence_length = 1
        if 'LSTM' in architecture: 
            self.has_memory = True
            self.sequence_length = action_cfg.sequence_length # CHECK

        self.minibatch_size =  int(self.batch_size // self.num_minibatches)
        self.batch_size_of_sequences = int(self.minibatch_size // self.sequence_length)
        
        if self.test == 'OFFLINE':
            self.offline_model = Autoencoder(obs_dim=self.obs_dim,
                                action_dim=self.action_dim,
                                action_type=self.action_type,
                                latent_dim = self.latent_dim,
                                architecture=self.architecture,
                                mode=mode)
            self.offline_model.to(self.device)
            self.offline_optimizer = torch.optim.Adam(self.offline_model.parameters(), lr=self.lr) # CHECK
            self.offline_loss_fn = nn.MSELoss()
        else:
            self.acmodel = ACModel(obs_dim=self.obs_dim,
                               action_dim=self.action_dim,
                               action_type=self.action_type,
                               latent_dim = self.latent_dim,
                               architecture=self.architecture,
                               mode=mode)
            self.acmodel.to(self.device)
            self.optimizer = torch.optim.Adam(self.acmodel.parameters(), lr=self.lr, eps=1e-08) # CHECK

        # change mode
        self.train()
    
    @property
    def memory_size(self):
        if self.test == 'OFFLINE':
            return (self.offline_model.network.memory_size,)
        else:
            return (self.acmodel.network.memory_size,)
        
    def reset_actor(self):
    def reset_actor(self):
        # reset actor and critic
        self.acmodel.reset_actor()
    
    def reset_critic(self):
        # reset actor and critic
        self.acmodel.reset_critic()
        self.acmodel.reset_actor()
    
    def reset_critic(self):
        # reset actor and critic
        self.acmodel.reset_critic()

    def reset_network(self):
        # reset network
        self.acmodel.reset_network()
        
    def train(self, training=True):
        self.training = training
        if self.test == 'OFFLINE':
            self.offline_model.train(training=training)
        else:
            self.acmodel.train(training=training)

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
        
    # UPDATE
    def update_critic_state_ent(
        self, obs, full_obs, action, next_obs, not_done, logger,
        step, K=5, print_flag=True):
        if self.action_type == 'Continuous':
            with torch.no_grad():
                next_action, log_prob, _ = self.get_action(torch.Tensor(next_obs).to(self.device))
                target_Q1, target_Q2 = self.critic_target(torch.Tensor(next_obs).to(self.device), action = next_action)
                target_V = torch.min(target_Q1, target_Q2) - self.alpha.detach() * log_prob
                # get current Q estimates
            current_Q1, current_Q2 = self.critic(torch.Tensor(obs).to(self.device), action = action)
        elif self.action_type == 'Discrete':
            with torch.no_grad():
                _, log_prob, action_probs = self.get_action(torch.Tensor(next_obs).to(self.device))
                target_Q1, target_Q2 = self.critic_target(torch.Tensor(next_obs).to(self.device))
                target_V = action_probs * (torch.min(target_Q1, target_Q2) - self.alpha.detach() * log_prob)
                target_V = target_V.sum(1).unsqueeze(-1)
                # get current Q estimates
            current_Q1, current_Q2 = self.critic(torch.Tensor(obs).to(self.device))
            current_Q1 = current_Q1.gather(1, action.long())
            current_Q2 = current_Q2.gather(1, action.long())
        
        # compute state entropy
        state_entropy = pebble.compute_state_entropy(obs, full_obs, k=K, state_type=self.state_type)
        if print_flag:
            logger.log("train_critic/entropy", state_entropy.mean(), step)
            logger.log("train_critic/entropy_max", state_entropy.max(), step)
            logger.log("train_critic/entropy_min", state_entropy.min(), step)
        self.s_ent_stats.update(state_entropy)
        if self.normalize_state_entropy:
            norm_state_entropy = state_entropy / self.s_ent_stats.std
            if print_flag:
                logger.log("train_critic/norm_entropy", norm_state_entropy.mean(), step)
                logger.log("train_critic/norm_entropy_max", norm_state_entropy.max(), step)
                logger.log("train_critic/norm_entropy_min", norm_state_entropy.min(), step)
            state_entropy = norm_state_entropy
        target_Q = state_entropy + (not_done * self.discount * target_V)
        target_Q = target_Q.detach()     
        qf1_loss = F.mse_loss(current_Q1, target_Q)
        qf2_loss = F.mse_loss(current_Q2, target_Q)
        critic_loss =  qf1_loss + qf2_loss
        if print_flag:
            logger.log('train_critic/loss', critic_loss, step)

        # Optimize the critic
        self.optimizer.zero_grad()
        critic_loss.backward()
        self.optimizer.step()
        self.critic.log(logger, step)
    
    def save(self, model_dir, step):
        if self.test == 'OFFLINE':
            torch.save(
                self.offline_model.network.state_dict(), '%s/encoder_%s.pt' % (model_dir, step)
            )
            torch.save(
                self.offline_model.decoder.state_dict(), '%s/decoder_%s.pt' % (model_dir, step)
            )
        else:
            torch.save(
                self.acmodel.actor.state_dict(), '%s/actor_%s.pt' % (model_dir, step)
            )
            torch.save(
                self.acmodel.critic.state_dict(), '%s/critic_%s.pt' % (model_dir, step)
            )
            torch.save(
                self.acmodel.network.state_dict(), '%s/encoder_%s.pt' % (model_dir, step)
            )
        
    def load(self, model_dir, step, mode = None):
        self.acmodel.network.load_state_dict(
            torch.load('%s/encoder_%s.pt' % (model_dir, step))
        )
        
        if mode != 'OFFLINE':
            self.acmodel.actor.load_state_dict(
                torch.load('%s/actor_%s.pt' % (model_dir, step))
            )
            self.acmodel.critic.load_state_dict(
                torch.load('%s/critic_%s.pt' % (model_dir, step))
            )
        
    def freeze_models(self, mode = None):
        if mode == 'OFFLINE':
            self.acmodel.network.eval()
            # for p in self.acmodel.network.parameters():
            #     p.requires_grad = False
            self.acmodel.network.eval()
            # for p in self.acmodel.network.parameters():
            #     p.requires_grad = False
        elif mode == None:
            return
        else:
            model_name=getattr(self.acmodel, mode, None)
            if model_name is None:
                print(f"No model named '{model_name}' found.")
                return
            for p in model_name.parameters():
                p.requires_grad = False

    def offline_update(self, trajectory, logger, step):
        batch_size = self.batch_size * 2
        epoch_loss = 0
        no_batches = int(len(trajectory) // batch_size)

        sequence_ids = list(range(0, len(trajectory)-self.sequence_length, self.sequence_length))
        
        for _ in range(no_batches):
            batch_ids = random.sample(sequence_ids, batch_size)
            
            # Consider using numpy arrays
            batch = []
            for i in batch_ids:
                for j in range(self.sequence_length):
                    batch.append(trajectory[i+j]) 
            # Consider using numpy arrays

            obs_t, action_t, reward_t, done_t, obs_t1 = zip(*batch)
            
            obs_t = torch.tensor(np.stack(obs_t), dtype=torch.float32)
            done_t = torch.Tensor(done_t).to(self.device)
            obs_t1 = torch.tensor(np.stack(obs_t1), dtype=torch.float32)
            # print('Obs Shape=',obs_t.shape)
            # print('Done Shape=',done_t.shape)

            memories = np.zeros([batch_size*self.sequence_length,self.memory_size[0]])
            # print('Memory Shape=',memories.shape)

            for sequence in range(batch_size): 
                # print(f'    Sequence: {sequence}')
                batch_loss = 0

                for i in range(sequence, sequence+self.sequence_length):
                    obs = obs_t[i].to(self.device).unsqueeze(0)
                    memory_tensor = torch.FloatTensor(memories[i]).to(self.device).unsqueeze(0)
                    mask_tensor = (1-done_t[i]).to(self.device).unsqueeze(0)
                    
                    prediction_obs, embedding, memory,  = self.offline_model(obs, memory_tensor * mask_tensor)
                    memory = memory.detach().cpu().numpy()[0]

                    loss = self.offline_loss_fn(prediction_obs, obs)
                    batch_loss += loss

                    if self.has_memory and i < sequence + self.sequence_length-1:
                        memories[i + 1] = memory
                    
                batch_loss /= self.sequence_length
            
            self.offline_optimizer.zero_grad()
            batch_loss.backward()
            self.offline_optimizer.step()

            epoch_loss += batch_loss.item()
        
        epoch_loss /= no_batches
        logger.log('train_autoencoder/loss', epoch_loss, steps)
        logger.log('train_autoencoder/loss', epoch_loss, steps)
        return epoch_loss
           
    # UPDATE
    # Add intrinsic reward options
    # Possibly disregard true rewards
    # Add support for multiple actors
    def pretrain_update(self, trajectory, next, logger, step):
        obs, actions, logprobs, values, rewards, dones, memories = trajectory
        next_obs, next_done, next_memory = next  

        obs, actions = torch.Tensor(obs).to(self.device), torch.Tensor(actions).to(self.device)
        logprobs, values = torch.Tensor(logprobs).to(self.device), torch.Tensor(values).to(self.device)
        rewards, dones = torch.Tensor(rewards).to(self.device), torch.Tensor(dones).to(self.device)
        if self.has_memory: 
            memories = torch.Tensor(memories).to(self.device)
        
        next_obs = torch.Tensor(next_obs).to(self.device).unsqueeze(0)      
        
        if self.has_memory:
            next_memory, next_mask = torch.Tensor(next_memory).to(self.device).unsqueeze(0), torch.Tensor([1-next_done]).to(self.device).unsqueeze(0)

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
        
        # flatten the batch
        b_obs = obs.reshape((-1,) + tuple(self.obs_dim))
        b_logprobs = logprobs.reshape(-1)
        if self.action_type == 'Continuous':
            b_actions = actions.reshape((-1,) + tuple(self.action_dim))
        elif self.action_type == 'Discrete':
            b_actions = actions.reshape((-1,1))
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_dones = dones.reshape(-1)
        b_values = values.reshape(-1)
        if self.has_memory:
            b_memories = memories.reshape((-1,) + tuple(self.memory_size))
        
        self.clipfracs = []

        for epoch in range(self.update_epochs):
            # print('Epoch ', epoch)
            print_flag = False
            if epoch == self.update_epochs -1:
                logger.log('train/batch_reward', rewards.mean(), step)
                print_flag = True
            
            suffled_minibatches = np.arange(0, self.batch_size, self.minibatch_size)
            # np.random.shuffle(suffled_minibatches)
            for minibatch in suffled_minibatches:
                start = minibatch
                end = minibatch + self.minibatch_size
                # print(f'  Minibatch {start} - {end}')

                suffled_sequences = np.arange(start, end, self.sequence_length)
                np.random.shuffle(suffled_sequences)

                for sequence in suffled_sequences: 
                    # print(f'    Sequence: {sequence}')

                    batch_entropy = 0
                    batch_value = 0
                    batch_actor_loss = 0
                    batch_critic_loss = 0
                    batch_loss = 0

                    for i in range(sequence, sequence+self.sequence_length):                  
                        # print(f'      i: {i}')

                        obs_tensor = b_obs[i].unsqueeze(0)
                        if self.has_memory:
                            memory_tensor = b_memories[i].unsqueeze(0)
                            mask_tensor = (1-b_dones[i]).to(self.device).unsqueeze(0)

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

                        mb_advantages = b_advantages[i]
                        if self.norm_adv: # Use on vector envs
                            mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                        # Policy loss
                        pg_loss1 = -mb_advantages * ratio
                        pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef)
                        actor_loss = torch.max(pg_loss1, pg_loss2).mean()

                        # Value loss
                        newvalue = newvalue.view(-1)
                        if self.clip_vloss:
                            v_loss_unclipped = (newvalue - b_returns[i]) ** 2
                            v_clipped = b_values[i] + torch.clamp(newvalue - b_values[i], -self.clip_coef, self.clip_coef)
                            v_loss_clipped = (v_clipped - b_returns[i]) ** 2
                            v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                            critic_loss = 0.5 * v_loss_max.mean()
                        else:
                            critic_loss = 0.5 * ((newvalue - b_returns[i]) ** 2).mean()

                        entropy_loss = entropy.mean()
                        loss = actor_loss - self.ent_coef * entropy_loss + critic_loss * self.vf_coef

                        # print(f'  i: {i}, Loss: {loss}, Actor Loss: {actor_loss}, Critic Loss: {critic_loss}, Entropy Loss: {entropy_loss}')
                        # Update batch values
                        batch_entropy += entropy.item()
                        batch_value += newvalue.mean().item()
                        batch_actor_loss += actor_loss.item()
                        batch_critic_loss += critic_loss.item()
                        batch_loss += loss

                        if self.has_memory and i < sequence + self.sequence_length-1:
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
 
    def update(self, trajectory, next, logger, step):
        obs, actions, logprobs, values, rewards, dones, memories = trajectory
        next_obs, next_done, next_memory = next  

        obs, actions = torch.Tensor(obs).to(self.device), torch.Tensor(actions).to(self.device)
        logprobs, values = torch.Tensor(logprobs).to(self.device), torch.Tensor(values).to(self.device)
        rewards, dones = torch.Tensor(rewards).to(self.device), torch.Tensor(dones).to(self.device)
        if self.has_memory: 
            memories = torch.Tensor(memories).to(self.device)
        
        next_obs = torch.Tensor(next_obs).to(self.device).unsqueeze(0)      
        
        if self.has_memory:
            next_memory, next_mask = torch.Tensor(next_memory).to(self.device).unsqueeze(0), torch.Tensor([1-next_done]).to(self.device).unsqueeze(0)

        # Calculate Advantages and Expected Returns
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
        
        # flatten the batch
        b_obs = obs.reshape((-1,) + tuple(self.obs_dim))
        b_logprobs = logprobs.reshape(-1)
        if self.action_type == 'Continuous':
            b_actions = actions.reshape((-1,) + tuple(self.action_dim))
        elif self.action_type == 'Discrete':
            b_actions = actions.reshape((-1,1))
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_dones = dones.reshape(-1)
        b_values = values.reshape(-1)
        if self.has_memory:
            b_memories = memories.reshape((-1,) + tuple(self.memory_size))
        
        self.clipfracs = []

        for epoch in range(self.update_epochs):
            # print('Epoch ', epoch)
            print_flag = False
            if epoch == self.update_epochs -1:
                logger.log('train/batch_reward', rewards.mean(), step)
                print_flag = True
            
            suffled_minibatches = np.arange(0, self.batch_size, self.minibatch_size)
            # np.random.shuffle(suffled_minibatches)
            for minibatch in suffled_minibatches:
                start = minibatch
                end = minibatch + self.minibatch_size
                # print(f'  Minibatch {start} - {end}')

                suffled_sequences = np.arange(start, end, self.sequence_length)
                np.random.shuffle(suffled_sequences)

                for sequence in suffled_sequences: 
                    # print(f'    Sequence: {sequence}')

                    batch_entropy = 0
                    batch_value = 0
                    batch_actor_loss = 0
                    batch_critic_loss = 0
                    batch_loss = 0

                    for i in range(sequence, sequence+self.sequence_length):                  
                        # print(f'      i: {i}')

                        obs_tensor = b_obs[i].unsqueeze(0)
                        if self.has_memory:
                            memory_tensor = b_memories[i].unsqueeze(0)
                            mask_tensor = (1-b_dones[i]).to(self.device).unsqueeze(0)

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

                        mb_advantages = b_advantages[i]
                        if self.norm_adv: # Use on vector envs
                            mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                        # Policy loss
                        pg_loss1 = -mb_advantages * ratio
                        pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef)
                        actor_loss = torch.max(pg_loss1, pg_loss2).mean()

                        # Value loss
                        newvalue = newvalue.view(-1)
                        if self.clip_vloss:
                            v_loss_unclipped = (newvalue - b_returns[i]) ** 2
                            v_clipped = b_values[i] + torch.clamp(newvalue - b_values[i], -self.clip_coef, self.clip_coef)
                            v_loss_clipped = (v_clipped - b_returns[i]) ** 2
                            v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                            critic_loss = 0.5 * v_loss_max.mean()
                        else:
                            critic_loss = 0.5 * ((newvalue - b_returns[i]) ** 2).mean()

                        entropy_loss = entropy.mean()
                        loss = actor_loss - self.ent_coef * entropy_loss + critic_loss * self.vf_coef

                        # print(f'  i: {i}, Loss: {loss}, Actor Loss: {actor_loss}, Critic Loss: {critic_loss}, Entropy Loss: {entropy_loss}')
                        # Update batch values
                        batch_entropy += entropy.item()
                        batch_value += newvalue.mean().item()
                        batch_actor_loss += actor_loss.item()
                        batch_critic_loss += critic_loss.item()
                        batch_loss += loss

                        if self.has_memory and i < sequence + self.sequence_length-1:
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
 
    def update(self, trajectory, next, logger, step):
        obs, actions, logprobs, values, rewards, dones, memories = trajectory
        next_obs, next_done, next_memory = next  

        obs, actions = torch.Tensor(obs).to(self.device), torch.Tensor(actions).to(self.device)
        logprobs, values = torch.Tensor(logprobs).to(self.device), torch.Tensor(values).to(self.device)
        rewards, dones = torch.Tensor(rewards).to(self.device), torch.Tensor(dones).to(self.device)
        if self.has_memory: 
            memories = torch.Tensor(memories).to(self.device)
        
        next_obs = torch.Tensor(next_obs).to(self.device).unsqueeze(0)      
        
        if self.has_memory:
            next_memory, next_mask = torch.Tensor(next_memory).to(self.device).unsqueeze(0), torch.Tensor([1-next_done]).to(self.device).unsqueeze(0)

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

        # flatten the batch
        b_obs = obs.reshape((-1,) + tuple(self.obs_dim))
        b_logprobs = logprobs.reshape(-1)
        if self.action_type == 'Continuous':
            b_actions = actions.reshape((-1,) + tuple(self.action_dim))
        elif self.action_type == 'Discrete':
            b_actions = actions.reshape((-1,1))
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_dones = dones.reshape(-1)
        b_values = values.reshape(-1)
        if self.has_memory:
            b_memories = memories.reshape((-1,) + tuple(self.memory_size))
        
        self.clipfracs = []

        for epoch in range(self.update_epochs):
            # print('Epoch ', epoch)
            print_flag = False
            if epoch == self.update_epochs -1:
                logger.log('train/batch_reward', rewards.mean(), step)
                print_flag = True
            
            suffled_minibatches = np.arange(0, self.batch_size, self.minibatch_size)
            # np.random.shuffle(suffled_minibatches)
            for minibatch in suffled_minibatches:
                start = minibatch
                end = minibatch + self.minibatch_size
                # print(f'  Minibatch {start} - {end}')

                suffled_sequences = np.arange(start, end, self.sequence_length)
                np.random.shuffle(suffled_sequences)

                for sequence in suffled_sequences: 
                    # print(f'    Sequence: {sequence}')

                    batch_entropy = 0
                    batch_value = 0
                    batch_actor_loss = 0
                    batch_critic_loss = 0
                    batch_loss = 0

                    for i in range(sequence, sequence+self.sequence_length):                  
                        # print(f'      i: {i}')

                        obs_tensor = b_obs[i].unsqueeze(0)
                        obs_tensor = b_obs[i].unsqueeze(0)
                        if self.has_memory:
                            memory_tensor = b_memories[i].unsqueeze(0)
                            mask_tensor = (1-b_dones[i]).to(self.device).unsqueeze(0)
                            memory_tensor = b_memories[i].unsqueeze(0)
                            mask_tensor = (1-b_dones[i]).to(self.device).unsqueeze(0)

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

                        mb_advantages = b_advantages[i]
                        if self.norm_adv: # Use on vector envs
                            mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                        # Policy loss
                        pg_loss1 = -mb_advantages * ratio
                        pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef)
                        actor_loss = torch.max(pg_loss1, pg_loss2).mean()

                        # Value loss
                        newvalue = newvalue.view(-1)
                        if self.clip_vloss:
                            v_loss_unclipped = (newvalue - b_returns[i]) ** 2
                            v_clipped = b_values[i] + torch.clamp(newvalue - b_values[i], -self.clip_coef, self.clip_coef)
                            v_loss_clipped = (v_clipped - b_returns[i]) ** 2
                            v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                            critic_loss = 0.5 * v_loss_max.mean()
                        else:
                            critic_loss = 0.5 * ((newvalue - b_returns[i]) ** 2).mean()

                        entropy_loss = entropy.mean()
                        loss = actor_loss - self.ent_coef * entropy_loss + critic_loss * self.vf_coef

                        # print(f'  i: {i}, Loss: {loss}, Actor Loss: {actor_loss}, Critic Loss: {critic_loss}, Entropy Loss: {entropy_loss}')
                        # Update batch values
                        batch_entropy += entropy.item()
                        batch_value += newvalue.mean().item()
                        batch_actor_loss += actor_loss.item()
                        batch_critic_loss += critic_loss.item()
                        batch_loss += loss

                        if self.has_memory and i < sequence + self.sequence_length-1:
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