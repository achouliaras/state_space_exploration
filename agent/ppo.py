import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from agent import Agent
from agent.pretraining import pebble
from agent.common.feature_extraction.autoencoder import AutoEncoder
from agent.common.transition_model import InverseTransitionModel
from agent.common.actor_critic.actor_critic import ACModel

class PPO(Agent):
    """PPO algorithm."""
    def __init__(self, obs_dim, action_type, device, latent_dim, architecture, state_type,
                 agent_cfg, action_cfg, import_protocol, deploy_mode, mode=1, normalize_state_entropy=True):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_type = action_type
        self.device = device
        self.latent_dim = latent_dim
        self.architecture = architecture
        self.import_protocol = import_protocol
        self.deploy_mode = deploy_mode

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
        
        if deploy_mode:
            self.acmodel = ACModel(obs_dim=self.obs_dim,
                               action_dim=self.action_dim,
                               action_type=self.action_type,
                               latent_dim = self.latent_dim,
                               architecture=self.architecture,
                               mode=mode)
            self.acmodel.double()
            self.acmodel.to(self.device)
            self.optimizer = torch.optim.Adam(self.acmodel.parameters(), lr=self.lr, eps=1e-08) # CHECK
        else:
            # self.autoencoder = AutoEncoder(obs_dim=self.obs_dim,
            #                     action_dim=self.action_dim,
            #                     action_type=self.action_type,
            #                     latent_dim = self.latent_dim,
            #                     architecture=self.architecture,
            #                     mode=mode)
            # self.autoencoder.double()
            # self.autoencoder.to(self.device)
            # self.autoencoder_optimizer = torch.optim.Adam(self.autoencoder.parameters(), lr=self.lr) # CHECK     
            # self.autoencoder_loss_fn = nn.MSELoss()

            self.inverseMDP = InverseTransitionModel(obs_dim=self.obs_dim,
                                                        action_dim=self.action_dim,
                                                        action_type=self.action_type,
                                                        latent_dim = self.latent_dim,
                                                        architecture=self.architecture,
                                                        mode=mode)
            self.inverseMDP.double()
            self.inverseMDP.to(self.device)
            self.inverseMDP_optimizer = torch.optim.Adam(self.inverseMDP.parameters(), lr=self.lr) # CHECK
            if self.action_type == 'Continuous': 
                self.inverseMDP_loss_fn = nn.MSELoss()
            elif self.action_type == 'Discrete': 
                self.inverseMDP_loss_fn = nn.CrossEntropyLoss()

        # change mode
        self.train()
    
    @property
    def memory_size(self):
        if self.deploy_mode:
           return (self.acmodel.network.memory_size,)
        else:  
            # return (self.autoencoder.network.memory_size,)
            return (self.inverseMDP.network.memory_size,)
        
    def reset_actor(self):
        # reset actor and critic
        self.acmodel.reset_actor()
    
    def reset_critic(self):
        # reset actor and critic
        self.acmodel.reset_critic()

    def reset_network(self):
        # reset network
        self.acmodel.reset_network()
        
    def train(self, training=True):
        self.training = training
        if self.deploy_mode:
            self.acmodel.train(training=training)
        else:
            # self.autoencoder.train(training=training)
            self.inverseMDP.train(training=training)

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
                next_action, log_prob, _ = self.get_action(torch.tensor(next_obs).to(self.device))
                target_Q1, target_Q2 = self.critic_target(torch.tensor(next_obs).to(self.device), action = next_action)
                target_V = torch.min(target_Q1, target_Q2) - self.alpha.detach() * log_prob
                # get current Q estimates
            current_Q1, current_Q2 = self.critic(torch.tensor(obs).to(self.device), action = action)
        elif self.action_type == 'Discrete':
            with torch.no_grad():
                _, log_prob, action_probs = self.get_action(torch.tensor(next_obs).to(self.device))
                target_Q1, target_Q2 = self.critic_target(torch.tensor(next_obs).to(self.device))
                target_V = action_probs * (torch.min(target_Q1, target_Q2) - self.alpha.detach() * log_prob)
                target_V = target_V.sum(1).unsqueeze(-1)
                # get current Q estimates
            current_Q1, current_Q2 = self.critic(torch.tensor(obs).to(self.device))
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
    
    def save(self, model_dir, step, mode=None):
        if "OFFLINE" in mode:
            # torch.save(
            #     self.autoencoder.network.state_dict(), '%s/encoder_%s.pt' % (model_dir, step)
            # )
            # torch.save(
            #     self.autoencoder.decoder.state_dict(), '%s/decoder_%s.pt' % (model_dir, step)
            # )
            torch.save(
                self.inverseMDP.network.state_dict(), '%s/encoder_%s.pt' % (model_dir, step)
            )
            torch.save(
                self.inverseMDP.action_model.state_dict(), '%s/action_model_%s.pt' % (model_dir, step)
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
        
        if "OFFLINE" not in mode:
            self.acmodel.actor.load_state_dict(
                torch.load('%s/actor_%s.pt' % (model_dir, step))
            )
            self.acmodel.critic.load_state_dict(
                torch.load('%s/critic_%s.pt' % (model_dir, step))
            )
        
    def freeze_models(self, mode = None):
        # Freeze feature extractor from encoder (cnn/mlp)
        if 'PART' in mode:
            for name, p in self.acmodel.network.cnn.named_parameters():
                p.requires_grad = False
                if '7' in name:
                    print('HERE ', name, p.requires_grad)
                    p.requires_grad = True
        # Freeze the entire encoder (cnn/mlp + lstm)
        if 'ALL' in mode:
            for name, p in self.acmodel.network.named_parameters():
                p.requires_grad = False
        # Freeze specified module actor, critic etc.
        if mode is not None:
            model_name=getattr(self.acmodel, mode, None)
            if model_name is None:
                print(f"No model named '{model_name}' found.")
                return
            for p in model_name.parameters():
                p.requires_grad = False

    # Add Imitation Learning Options
    def offline_update(self, trajectory, logger, steps):
        batch_size = self.batch_size * 4
        epoch_loss = 0
                
        obs, actions, rewards, dones = trajectory
        not_dones_t = 1 - dones

        sequence_ids = list(range(0, steps-self.sequence_length-1,self.sequence_length))
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
                memory_tensor = memories[i]
                mask_tensor = not_done_t[i]
                action_tensor = action_t[i].squeeze(1)
                next_obs_tensor = obs_t[i+1]
                next_mask_tensor = not_done_t[i+1]
                
                if self.has_memory:
                    # prediction_obs, _, memory  = self.autoencoder(obs_tensor, memory_tensor * mask_tensor)
                    pred_action_logprobs, memory, next_memory  = self.inverseMDP(obs_tensor, next_obs_tensor, 
                                                                              memory_tensor * mask_tensor,
                                                                              next_mask_tensor)
                else:
                    # prediction_obs, _, memory,  = self.autoencoder(obs_tensor)
                    pred_action_logprobs, memory, next_memory  = self.inverseMDP(obs_tensor, next_obs_tensor)

                if self.has_memory and i < self.sequence_length-1:
                    memories[i + 1] = memory.detach()
                    memories[i + 2] = next_memory.detach()

                # loss = self.autoencoder_loss_fn(prediction_obs, obs_tensor)

                loss = self.inverseMDP_loss_fn(pred_action_logprobs, action_tensor)
                
                batch_loss += loss

            batch_loss = batch_loss/self.sequence_length
            # self.autoencoder_optimizer.zero_grad()
            # batch_loss.backward()
            # self.autoencoder_optimizer.step()

            self.inverseMDP_optimizer.zero_grad()
            batch_loss.backward()
            self.inverseMDP_optimizer.step()

            epoch_loss += batch_loss.item()
        
        epoch_loss /= no_batches
        logger.log('train_autoencoder/loss', epoch_loss, steps)
        return epoch_loss
           
    # UPDATE
    # Add intrinsic reward options
    # Possibly disregard true rewards
    # Add support for multiple actors
    def pretrain_update(self, trajectory, next, logger, step):
        obs, actions, logprobs, values, rewards, dones, memories = trajectory
        next_obs, next_done, next_memory = next  

        obs = torch.DoubleTensor(obs).to(self.device)
        actions = torch.DoubleTensor(actions).to(self.device)
        logprobs = torch.DoubleTensor(logprobs).to(self.device)
        values = torch.DoubleTensor(values).to(self.device)
        rewards = torch.DoubleTensor(rewards).to(self.device)
        dones = torch.DoubleTensor(dones).to(self.device)
        if self.has_memory: 
            memories = torch.DoubleTensor(memories).to(self.device)
        
        next_obs = torch.DoubleTensor(next_obs).to(self.device).unsqueeze(0)      
        
        if self.has_memory:
            next_memory = torch.DoubleTensor(next_memory).to(self.device).unsqueeze(0)
            next_mask = torch.DoubleTensor([1-next_done]).to(self.device).unsqueeze(0)

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

        obs = torch.DoubleTensor(obs).to(self.device)
        actions = torch.DoubleTensor(actions).to(self.device)
        logprobs = torch.DoubleTensor(logprobs).to(self.device)
        values = torch.DoubleTensor(values).to(self.device)
        rewards = torch.DoubleTensor(rewards).to(self.device)
        dones = torch.DoubleTensor(dones).to(self.device)
        if self.has_memory: 
            memories = torch.DoubleTensor(memories).to(self.device)
        
        next_obs = torch.DoubleTensor(next_obs).to(self.device).unsqueeze(0)      
        
        if self.has_memory:
            next_memory = torch.DoubleTensor(next_memory).to(self.device).unsqueeze(0)
            next_mask = torch.DoubleTensor([1-next_done]).to(self.device).unsqueeze(0)

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