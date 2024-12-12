import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from agent import Agent
from agent.pretraining import pebble
from agent.common.encoder import Encoder
from agent.common.critic import DoubleQCritic, SimpleCritic
from agent.common.actor import DiagGaussianActor, CategoricalActor, SimpleActor


class ACModel(nn.Module):
    def __init__(self,obs_space, obs_dim, action_dim, action_type, architecture, mode):
        super().__init__()
        
        self.obs_space = obs_space
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.action_type = action_type
        self.architecture = architecture
        self.mode = mode

        self.network = self.create_network()
        self.actor = self.create_actor()
        self.critic = self.create_critic()

    def create_network(self):
        # CNN, MLP, LSTM
        network = Encoder(obs_space=self.obs_space,
                          obs_dim=self.obs_dim,
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
                                  hidden_depth=0
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
                                hidden_depth=0
                                )
        else:
            raise NotImplementedError
        return actor
    
    def reset_actor_critic(self):
        # reset actor and critic
        self.actor = self.create_actor()
        self.critic = self.create_critic()

    def reset_network(self):
        # reset network
        self.network = self.create_network()

    def train(self, training=True):
        self.training = training
        self.network.train(training)
        self.actor.train(training)
        self.critic.train(training)
    
    def forward(self, obs, memory):
        if self.action_type == 'Continuous':
            x, memory = obs, None
        elif self.action_type == 'Discrete':
            x, memory = self.network(obs, memory)
        logits = self.actor(x)
        state_value = self.critic(x)

        return logits, state_value, memory

    def log(self, logger, step):
        self.network.log(logger,step)
        self.actor.log(logger,step)
        self.critic.log(logger,step)
class PPO(Agent):
    """PPO algorithm."""
    def __init__(self, obs_space, obs_dim, action_range, device,
                 agent_cfg, action_cfg, mode=1, normalize_state_entropy=True):
        super().__init__()

        self.obs_space = obs_space
        self.obs_dim = obs_dim
        self.action_scale = (action_range[1] - action_range[0]) /2.0
        self.action_bias = (action_range[1] + action_range[0]) /2.0
        self.device = device
        
        self.state_type = agent_cfg.state_type
        self.action_type = agent_cfg.action_type
        self.architecture = agent_cfg.architecture
        self.action_dim = agent_cfg.action_dim

        self.num_update_steps = action_cfg.num_update_steps
        self.batch_size = action_cfg.batch_size
        self.lr = action_cfg.actor_lr
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
        self.target_kl = action_cfg.target_kl

        self.minibatch_size = int(self.batch_size // self.num_minibatches)
        
        self.s_ent_stats = pebble.TorchRunningMeanStd(shape=[1], device=self.device)
        self.normalize_state_entropy = normalize_state_entropy
        
        self.acmodel = ACModel(obs_space=self.obs_space,
                               obs_dim=self.obs_dim,
                               action_dim=self.action_dim,
                               architecture=self.architecture,
                               mode=mode)
        
        # optimizers
        self.optimizer = torch.optim.Adam(self.acmodel.parameters(), lr=self.lr, eps=1e-05) # CHECK
        
        # change mode
        self.train()
        
    def reset_actor_critic(self):
        # reset actor and critic
        self.acmodel.reset_actor_critic()

    def reset_network(self):
        # reset network
        self.acmodel.reset_network()
        
    def train(self, training=True):
        self.acmodel.train(training=training)

    def get_action(self, obs, action = None):
        if self.action_type == 'Continuous':
            (mean, log_std), state_value, memory = self.acmodel.forward(obs)
            std = torch.exp(log_std)
            dist = torch.distributions.Normal(mean, std)
            if action == None:
                action = dist.sample()
            return action, dist.log_prob(action).sum(1), dist.entropy().sum(1), state_value  
        elif self.action_type == 'Discrete':
            logits, state_value, memory = self.acmodel.forward(obs)
            # Action is a flat integer
            dist = Categorical(logits=logits)
            if action == None:
                action = dist.sample()
            return action, dist.log_prob(action), dist.entropy(), state_value

    def update_actor_and_critic(self, obs, actions, logprobs, values, returns, advantages,
                      logger, step, print_flag=True):
        _, newlogprob, entropy, newvalue = self.get_action(obs, actions)
        logratio = newlogprob - logprobs
        ratio = logratio.exp()

        with torch.no_grad():
            # calculate approx_kl http://joschu.net/blog/kl-approx.html
            old_approx_kl = (-logratio).mean()
            approx_kl = ((ratio - 1) - logratio).mean()
            self.clipfracs += [((ratio - 1.0).abs() > self.clip_coef).float().mean().item()]

        mb_advantages = advantages
        if self.norm_adv:
            mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

        # Policy loss
        pg_loss1 = -mb_advantages * ratio
        pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef)
        actor_loss = torch.max(pg_loss1, pg_loss2).mean()

        # Value loss
        newvalue = newvalue.view(-1)
        if self.clip_vloss:
            v_loss_unclipped = (newvalue - returns) ** 2
            v_clipped = values + torch.clamp(newvalue - values, -self.clip_coef, self.clip_coef)
            v_loss_clipped = (v_clipped - returns) ** 2
            v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
            critic_loss = 0.5 * v_loss_max.mean()
        else:
            critic_loss = 0.5 * ((newvalue - returns) ** 2).mean()

        entropy_loss = entropy.mean()
        loss = actor_loss - self.ent_coef * entropy_loss + critic_loss * self.vf_coef

        # Use action to take the suitable Q value
        if print_flag:
            logger.log('train_critic/loss', critic_loss, step)
            logger.log('train_actor/loss', actor_loss, step)
            logger.log('train_actor/entropy', entropy_loss, step)
            
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.acmodel.parameters(), self.max_grad_norm)
        self.optimizer.step()

        self.acmodel.log(logger, step)
        return approx_kl, old_approx_kl
        
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
        torch.save(
            self.acmodel.actor.state_dict(), '%s/actor_%s.pt' % (model_dir, step)
        )
        torch.save(
            self.acmodel.critic.state_dict(), '%s/critic_%s.pt' % (model_dir, step)
        )
        torch.save(
            self.acmodel.network.state_dict(), '%s/encoder%s.pt' % (model_dir, step)
        )
        
    def load(self, model_dir, step):
        self.acmodel.actor.load_state_dict(
            torch.load('%s/actor_%s.pt' % (model_dir, step))
        )
        self.acmodel.critic.load_state_dict(
            torch.load('%s/critic_%s.pt' % (model_dir, step))
        )
        self.acmodel.network.load_state_dict(
            torch.load('%s/encoder%s.pt' % (model_dir, step))
        )
            
    # UPDATE
    def pretrain_update(self, replay_buffer, logger, step, total_timesteps, gradient_update=1, K=5):
        for index in range(gradient_update):
            obs, full_obs, action, reward, next_obs, not_done, not_done_no_max = replay_buffer.sample_state_ent(self.batch_size)

            print_flag = False
            if index == gradient_update -1:
                logger.log('train/batch_reward', reward.mean(), step)
                print_flag = True
                
            if step % self.critic_update_frequency == 0:
                self.update_critic_state_ent(obs, full_obs, action, next_obs, not_done_no_max,
                                            logger, step, K=K, print_flag=print_flag)

            if step % self.actor_update_frequency == 0:
                for _ in range(self.actor_update_frequency):
                    self.update_actor_and_alpha(obs, logger, step, print_flag)
                    if self.action_type == 'Discrete': break # Do not update actor as in TD3 for Discrete environments
    
    # UPDATE
    def update_after_reset(self, replay_buffer, logger, step, total_timesteps, gradient_update=1, policy_update=True):
        for index in range(gradient_update):
            obs, action, reward, next_obs, not_done, not_done_no_max = replay_buffer.sample_tensors(self.batch_size)

            print_flag = False
            if index == gradient_update -1:
                logger.log('train/batch_reward', reward.mean(), step)
                print_flag = True
                
            if step % self.critic_update_frequency == 0:
                self.update_critic(obs, action, reward, next_obs, not_done_no_max, logger, step, print_flag)

            if index % self.actor_update_frequency == 0 and policy_update:
                for _ in range(self.actor_update_frequency):
                    self.update_actor_and_alpha(obs, logger, step, print_flag)
                    if self.action_type == 'Discrete': break # Do not update actor as in TD3 for Discrete environments
 
    def update(self, trajectory, next, logger, step):
        obs, actions, logprobs, values, rewards, dones = trajectory
        next_obs, next_done = next  

        obs, actions = torch.Tensor(obs).to(self.device), torch.Tensor(actions).to(self.device)
        logprobs, values = torch.Tensor(logprobs).to(self.device), torch.Tensor(values).to(self.device)
        rewards, dones = torch.Tensor(rewards).to(self.device), torch.Tensor(dones).to(self.device)
        next_obs, next_done = torch.Tensor(next_obs).to(self.device), torch.Tensor(next_done).to(self.device)       

        with torch.no_grad():
            next_value = self.acmodel.critic(next_obs).reshape(1,-1)
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
        b_obs = obs.reshape((-1,) + self.obs_dim)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + self.actor_cfg.action_dim)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)
        
        b_inds = np.arange(self.batch_size)
        self.clipfracs = []
        for epoch in range(self.update_epochs):
            print_flag = False
            if epoch == self.update_epochs -1:
                logger.log('train/batch_reward', rewards.mean(), step)
                print_flag = True
            
            np.random.shuffle(b_inds)
            for start in range(0, self.batch_size, self.minibatch_size):
                end = start+self.minibatch_size
                mb_inds = b_inds[start:end]
                approx_kl, old_approx_kl = self.update_actor_and_critic(b_obs[mb_inds], b_actions.long()[mb_inds], b_logprobs[mb_inds], b_values[mb_inds], 
                                             b_returns[mb_inds], b_advantages[mb_inds], logger, step, print_flag)
                
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