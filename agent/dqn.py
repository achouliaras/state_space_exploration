import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import lib.utils as utils
import hydra

from agent import Agent
from agent.common.actor_critic.critic_only import DoubleQCritic
from stable_baselines3.common.distributions import CategoricalDistribution
from stable_baselines3.common.utils import get_linear_fn

def compute_state_entropy(obs, full_obs, k, action_type):
    batch_size = 100
    with torch.no_grad():
        dists = []
        for idx in range(len(full_obs) // batch_size + 1):
            start = idx * batch_size
            end = (idx + 1) * batch_size
            if action_type == 'Continuous':
                dist = torch.norm(obs[:, None, :] - full_obs[None, start:end, :], dim=-1, p=2)
            else:
                #print(full_obs[None, start:end, :].shape)
                #print(obs[:, None, :].shape)
                dist = torch.norm(obs[:, None, :] - full_obs[None, start:end, :], dim=(-1,-2,-3), p=2)
            dists.append(dist)

        dists = torch.cat(dists, dim=1)
        knn_dists = torch.kthvalue(dists, k=k + 1, dim=1).values
        state_entropy = knn_dists

    return state_entropy.unsqueeze(1)

class DQNAgent(Agent):
    """DQN algorithm."""
    def __init__(self, obs_space, obs_dim, action_space, action_range, device, q_net_cfg,
                 discount, exploration_initial_eps, exploration_final_eps, exploration_fraction, q_net_lr,
                 q_net_betas, q_net_tau, q_net_target_update_frequency, batch_size, policy, q_net_max_grad=10, mode=0, 
                 normalize_state_entropy=True):
        super().__init__()

        self.obs_space = obs_space
        self.obs_dim = obs_dim
        self.action_space = action_space
        self.action_range = action_range
        self.device = torch.device(device)
        self.discount = discount
        self.q_net_tau = q_net_tau

        self.q_net_target_update_frequency = q_net_target_update_frequency
        self.batch_size = batch_size
        self.q_net_cfg = q_net_cfg
        self.q_net_lr = q_net_lr
        self.q_net_betas = q_net_betas
        self.max_grad_norm = q_net_max_grad
        self.s_ent_stats = utils.TorchRunningMeanStd(shape=[1], device=device)
        self.normalize_state_entropy = normalize_state_entropy
        self.policy = policy
        self.action_type = self.actor_cfg.action_type
        self.mode = mode

        self.exploration_initial_eps = exploration_initial_eps
        self.exploration_final_eps = exploration_final_eps
        self.exploration_fraction = exploration_fraction
        self.exploration_rate = 0.0

        #self.q_net = hydra.utils.instantiate(q_net_cfg, _convert_="all").to(self.device)
        self.q_net = self.create_q_net()
        #self.q_net_target = hydra.utils.instantiate(q_net_cfg, _convert_="all").to(self.device)
        self.q_net_target = self.create_q_net()
        
        self.q_net_target.load_state_dict(self.q_net.state_dict())

        # optimizers
        self.q_net_optimizer = torch.optim.Adam(
            self.q_net.parameters(),
            lr=q_net_lr,
            betas=q_net_betas)
        
        # change mode
        self.train()

        self.exploration_schedule = get_linear_fn(
            self.exploration_initial_eps,
            self.exploration_final_eps,
            self.exploration_fraction,
        )
    
    def create_q_net(self):
        q_net = DoubleQCritic(obs_space = self.obs_space, 
            obs_dim= self.obs_dim,
            action_dim= self.q_net_cfg.action_dim,
            action_type= self.q_net_cfg.action_type,
            policy= self.q_net_cfg.policy,
            hidden_dim= self.q_net_cfg.hidden_dim,
            hidden_depth= self.q_net_cfg.hidden_depth,
            mode= self.mode).to(self.device)
        return q_net
    
    def reset_q_net(self):
        # The CNN feature shouldn't reset...
        # copy them and paste on top of the reseted q_nets
        self.q_net = self.create_q_net()
        self.q_net_target = self.create_q_net()
        self.q_net_target.load_state_dict(self.q_net.state_dict())
        self.q_net_optimizer = torch.optim.Adam(
            self.q_net.parameters(),
            lr=self.q_net_lr,
            betas=self.q_net_betas)
    
    def _update_current_progress_remaining(self, num_timesteps: int, total_timesteps: int) -> None:
        """
        Compute current progress remaining (starts from 1 and ends to 0)

        :param num_timesteps: current number of timesteps
        :param total_timesteps:
        """
        self._current_progress_remaining = 1.0 - float(num_timesteps) / float(total_timesteps)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def act(self, obs, determ=False):
        if self.action_type == 'Continuous':
            raise Exception("DQN doesn't support continuous action spaces")
        elif self.action_type == 'Discrete':

            if not determ & np.random.rand() < self.exploration_rate:
                return self.action_space.sample()
            else:
                obs = torch.FloatTensor(obs).to(self.device)
                obs = obs.unsqueeze(0)
                dist = self.q_net.forward(obs)

                # Action is a flat integer
                dist = self.categorical.proba_distribution(action_logits=dist)
                action = dist.get_actions(deterministic=False)
                return utils.to_np(action[0])

    def update_q_net(self, obs, action, reward, next_obs, 
                      not_done, logger, step, print_flag=True):
                
        with torch.no_grad():
            # TO CHECK MAY BE DIFFERENT FOR DQN THAN SAC
            target_Q1, target_Q2 = self.q_net_target(next_obs, next_action=None) # No action needed for Discrete
            target_V = torch.min(target_Q1, target_Q2)
            target_V = target_V.max(1).unsqueeze(-1)    #??? Possible reshape needed
            target_Q = reward + (not_done * self.discount * target_V) # OLO TO ZOUMI EINAI EDW
            target_Q = target_Q.detach()

        # get current Q estimates
        current_Q1, current_Q2 = self.q_net(obs, action=None) # No action needed for Discrete
        current_Q1 = current_Q1.gather(1, action.long())
        current_Q2 = current_Q2.gather(1, action.long())
        
        qf1_loss = F.smooth_l1_loss(current_Q1, target_Q)
        qf2_loss = F.smooth_l1_loss(current_Q2, target_Q)
        
        q_net_loss =  qf1_loss + qf2_loss
        # Use action to take the suitable Q value
        
        if print_flag:
            logger.log('train_q_net/loss', q_net_loss, step)

        # Optimize the q_net
        self.q_net_optimizer.zero_grad()
        qf1_loss.backward(retain_graph=True)
        qf2_loss.backward(retain_graph=True)
        # Clip gradient norm
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), self.max_grad_norm)
        self.q_net_optimizer.step()
        self.q_net.log(logger, step)
        
    def update_q_net_state_ent(
        self, obs, full_obs, action, next_obs, not_done, logger,
        step, K=5, print_flag=True):
        
        output = self.actor.forward(next_obs)
        if self.action_type == 'Continuous':
            dist = output
            next_action = dist.rsample()
            log_prob = dist.log_prob(next_action).sum(-1, keepdim=True)
        elif self.action_type == 'Discrete':
            action_probs = output
            next_action = self.categorical.actions_from_params(action_logits=output)
            z = action_probs == 0.0
            z = z.float() * 1e-8
            log_prob = torch.log(action_probs + z)
        
        target_Q1, target_Q2 = self.q_net_target(next_obs, next_action)
        target_V = torch.min(target_Q1, target_Q2) - self.alpha.detach() * log_prob

        # compute state entropy
        state_entropy = compute_state_entropy(obs, full_obs, k=K, action_type=self.action_type)
        #print('State entropy = ', state_entropy.shape)
        if print_flag:
            logger.log("train_q_net/entropy", state_entropy.mean(), step)
            logger.log("train_q_net/entropy_max", state_entropy.max(), step)
            logger.log("train_q_net/entropy_min", state_entropy.min(), step)
        
        self.s_ent_stats.update(state_entropy)
        norm_state_entropy = state_entropy / self.s_ent_stats.std
        
        if print_flag:
            logger.log("train_q_net/norm_entropy", norm_state_entropy.mean(), step)
            logger.log("train_q_net/norm_entropy_max", norm_state_entropy.max(), step)
            logger.log("train_q_net/norm_entropy_min", norm_state_entropy.min(), step)
        
        if self.normalize_state_entropy:
            state_entropy = norm_state_entropy
        
        target_Q = state_entropy + (not_done * self.discount * target_V)
        target_Q = target_Q.detach()

        # get current Q estimates
        # get current Q estimates
        current_Q1, current_Q2 = self.q_net(obs, action)
        current_Q1 = current_Q1.gather(1, action.long())
        current_Q2 = current_Q2.gather(1, action.long())
        
        qf1_loss = F.mse_loss(current_Q1, target_Q)
        qf2_loss = F.mse_loss(current_Q2, target_Q)
        q_net_loss =  qf1_loss + qf2_loss
        
        if print_flag:
            logger.log('train_q_net/loss', q_net_loss, step)

        # Optimize the q_net
        self.q_net_optimizer.zero_grad()
        qf1_loss.backward(retain_graph=True)
        qf2_loss.backward(retain_graph=True)
        self.q_net_optimizer.step()
        self.q_net.log(logger, step)
    
    def save(self, model_dir, step):
        torch.save(
            self.actor.state_dict(), '%s/actor_%s.pt' % (model_dir, step)
        )
        torch.save(
            self.q_net.state_dict(), '%s/q_net_%s.pt' % (model_dir, step)
        )
        torch.save(
            self.q_net_target.state_dict(), '%s/q_net_target_%s.pt' % (model_dir, step)
        )
        
    def load(self, model_dir, step):
        self.actor.load_state_dict(
            torch.load('%s/actor_%s.pt' % (model_dir, step))
        )
        self.q_net.load_state_dict(
            torch.load('%s/q_net_%s.pt' % (model_dir, step))
        )
        self.q_net_target.load_state_dict(
            torch.load('%s/q_net_target_%s.pt' % (model_dir, step))
        )
    
    def update_actor_and_alpha(self, obs, logger, step, print_flag=False):
        output = self.actor.forward(obs)
        if self.action_type == 'Continuous':
            dist = output
            action = dist.rsample()
            log_prob = dist.log_prob(action).sum(-1, keepdim=True)

            actor_Q1, actor_Q2 = self.q_net(obs, action)
            actor_Q = torch.min(actor_Q1, actor_Q2)
            actor_loss = (self.alpha.detach() * log_prob - actor_Q).mean()
        elif self.action_type == 'Discrete':
            action_probs = output
            action = self.categorical.actions_from_params(action_logits=output)
            z = action_probs == 0.0
            z = z.float() * 1e-8
            log_prob = torch.log(action_probs + z)
            
            actor_Q1, actor_Q2 = self.q_net(obs, action)
            actor_Q = torch.min(actor_Q1, actor_Q2)
            inside_term = (self.alpha.detach() * log_prob) - actor_Q
            actor_loss = (action_probs*inside_term).sum(dim=1).mean()
            log_prob = torch.sum(log_prob * action_probs, dim=1)        # CHECK AGAIN
            #print('actor_loss', actor_loss)
            
        if print_flag:
            logger.log('train_actor/loss', actor_loss, step)
            logger.log('train_actor/target_entropy', self.target_entropy, step)
            logger.log('train_actor/entropy', -log_prob.mean(), step)

        # optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.actor.log(logger, step)

        if self.learnable_temperature:
            self.log_alpha_optimizer.zero_grad()
            alpha_loss = -(self.alpha * 
                          (log_prob + self.target_entropy).detach()).mean()
            if print_flag:
                logger.log('train_alpha/loss', alpha_loss, step)
                logger.log('train_alpha/value', self.alpha, step)
            alpha_loss.backward()
            self.log_alpha_optimizer.step()
            
    def update(self, replay_buffer, logger, step, total_timesteps, gradient_update=1):
        for index in range(gradient_update):
            obs, action, reward, next_obs, not_done, not_done_no_max = replay_buffer.sample(
                self.batch_size)

            print_flag = False
            if index == gradient_update -1:
                logger.log('train/batch_reward', reward.mean(), step)
                print_flag = True
                
            self.update_q_net(obs, action, reward, next_obs, not_done_no_max,
                               logger, step, print_flag)
            
            self._update_current_progress_remaining(step, total_timesteps)

            self.exploration_rate = self.exploration_schedule(self._current_progress_remaining) 

        if step % self.q_net_target_update_frequency == 0:
            utils.soft_update_params(self.q_net, self.q_net_target,
                                     self.q_net_tau)
        
        
            
    def update_after_reset(self, replay_buffer, logger, step, total_timesteps, gradient_update=1, policy_update=True):
        for index in range(gradient_update):
            obs, action, reward, next_obs, not_done, not_done_no_max = replay_buffer.sample(
                self.batch_size)

            print_flag = False
            if index == gradient_update -1:
                logger.log('train/batch_reward', reward.mean(), step)
                print_flag = True
                
            self.update_q_net(obs, action, reward, next_obs, not_done_no_max,
                               logger, step, print_flag)

            self._update_current_progress_remaining(step, total_timesteps)

            self.exploration_rate = self.exploration_schedule(self._current_progress_remaining) 

            if index % self.q_net_target_update_frequency == 0:
                utils.soft_update_params(self.q_net, self.q_net_target,
                                         self.q_net_tau)
                
            
            
    def update_state_ent(self, replay_buffer, logger, step, total_timesteps, gradient_update=1, K=5):
        for index in range(gradient_update):
            obs, full_obs, action, reward, next_obs, not_done, not_done_no_max = replay_buffer.sample_state_ent(self.batch_size)

            print_flag = False
            if index == gradient_update -1:
                logger.log('train/batch_reward', reward.mean(), step)
                print_flag = True
            
            self.update_q_net_state_ent(
                obs, full_obs, action, next_obs, not_done_no_max,
                logger, step, K=K, print_flag=print_flag)

            if step % self.actor_update_frequency == 0:
                self.update_actor_and_alpha(obs, logger, step, print_flag)

        if step % self.q_net_target_update_frequency == 0:
            utils.soft_update_params(self.q_net, self.q_net_target,
                                     self.q_net_tau)