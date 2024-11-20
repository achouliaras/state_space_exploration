import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from agent import Agent
from agent.pretraining import pebble
from agent.common.critic import DoubleQCritic
from agent.common.actor import DiagGaussianActor, CategoricalActor

class SAC(Agent):
    """SAC algorithm."""
    def __init__(self, obs_space, obs_dim, action_range, device, 
                 actor_cfg, critic_cfg, action_cfg, discount, init_temperature,
                 learnable_temperature, mode=0, normalize_state_entropy=True):
        super().__init__()

        self.obs_space = obs_space
        self.obs_dim = obs_dim
        self.action_scale = (action_range[1] - action_range[0]) /2.0
        self.action_bias = (action_range[1] + action_range[0]) /2.0
        self.device = device
        self.discount = discount
        self.critic_tau = action_cfg.critic_tau
        self.actor_update_frequency = action_cfg.actor_update_frequency
        self.critic_update_frequency = action_cfg.critic_update_frequency
        self.critic_target_update_frequency = action_cfg.critic_target_update_frequency
        self.batch_size = action_cfg.batch_size
        self.learnable_temperature = learnable_temperature
        self.critic_cfg = critic_cfg
        self.critic_lr = action_cfg.critic_lr
        self.s_ent_stats = pebble.TorchRunningMeanStd(shape=[1], device=self.device)
        self.normalize_state_entropy = normalize_state_entropy
        self.init_temperature = init_temperature
        self.alpha_lr = action_cfg.alpha_lr
        self.actor_cfg = actor_cfg
        self.actor_lr = action_cfg.actor_lr
        self.action_type = self.actor_cfg.action_type
        self.state_type = self.critic_cfg.state_type
        self.mode = mode

        self.critic = self.create_critic()
        self.critic_target = self.create_critic()
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.actor = self.create_actor()
        
        self.log_alpha = torch.tensor(np.log(init_temperature), requires_grad=True, dtype=torch.float32, device=self.device)
        # set target entropy 
        self.target_entropy =  - action_cfg.target_entropy_scale * np.log((1.0 / self.actor_cfg.action_dim)) 

        # optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.critic_lr)
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.alpha_lr, eps=1e-4)
        
        # change mode
        self.train()
    
    def create_critic(self):
        critic = DoubleQCritic(obs_space = self.obs_space, 
            obs_dim= self.obs_dim,
            action_dim= self.critic_cfg.action_dim,
            action_type= self.critic_cfg.action_type,
            architecture= self.critic_cfg.architecture,
            hidden_dim= self.critic_cfg.hidden_dim,
            hidden_depth= self.critic_cfg.hidden_depth,
            mode= self.mode).to(self.device)
        return critic
    
    def create_actor(self):
        if self.action_type == 'Continuous':
            actor = DiagGaussianActor(obs_dim = self.actor_cfg.obs_dim, 
                action_dim = self.actor_cfg.action_dim,
                #action_type = self.actor_cfg.action_type,
                architecture = self.actor_cfg.architecture,
                hidden_dim = self.actor_cfg.hidden_dim, 
                hidden_depth = self.actor_cfg.hidden_depth,
                log_std_bounds = self.actor_cfg.log_std_bounds).to(self.device)
        elif self.action_type == 'Discrete':
            actor = CategoricalActor(obs_space = self.obs_space, 
                obs_dim = self.actor_cfg.obs_dim, 
                action_dim = self.actor_cfg.action_dim,
                #action_type = self.actor_cfg.action_type, 
                architecture = self.actor_cfg.architecture, 
                hidden_dim = self.actor_cfg.hidden_dim, 
                hidden_depth = self.actor_cfg.hidden_depth,
                mode= self.mode).to(self.device)
        return actor
    
    def reset_critic(self):
        # The CNN feature shouldn't reset...
        # copy them and paste on top of the reseted critics
        self.critic = self.create_critic()
        self.critic_target = self.create_critic()
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),lr=self.critic_lr)
    
    def reset_actor(self):
        # reset log_alpha
        self.log_alpha = torch.tensor(np.log(self.init_temperature),
                                        requires_grad=True, 
                                        dtype=torch.float32
                                      ).to(self.device)
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha],lr=self.alpha_lr, eps=1e-4)
        
        # reset actor
        self.actor = self.create_actor()
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        
    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def get_action(self, obs):
        if self.action_type == 'Continuous':
            mean, log_std = self.actor.forward(obs)
            std = log_std.exp()
            normal = torch.distributions.Normal(mean, std)
            x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
            y_t = torch.tanh(x_t)
            action = y_t * self.action_scale + self.action_bias
            log_prob = normal.log_prob(x_t)

            # Enforcing Action Bound
            log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
            log_prob = log_prob.sum(1, keepdim=True)
            mean = torch.tanh(mean) * self.action_scale + self.action_bias
            return action, log_prob, mean       
        elif self.action_type == 'Discrete':
            logits = self.actor.forward(obs)
            # Action is a flat integer
            dist = Categorical(logits=logits)
            action = dist.sample()
            # Action probabilities for calculating the adapted soft-Q loss
            action_probs = dist.probs
            log_prob = F.log_softmax(logits, dim=1)
            return action, log_prob, action_probs

    def update_critic(self, obs, action, reward, next_obs, 
                      not_done, logger, step, print_flag=True):
        if self.action_type == 'Continuous':
            with torch.no_grad():
                next_action, log_prob, _ = self.get_action(torch.Tensor(next_obs).to(self.device))
                target_Q1, target_Q2 = self.critic_target(torch.Tensor(next_obs).to(self.device), next_action)
                target_V = torch.min(target_Q1, target_Q2) - self.alpha.detach() * log_prob
                target_Q = reward.flatten() + (not_done.flatten() * self.discount * target_V)
                target_Q = target_Q.detach()
            current_Q1, current_Q2 = self.critic(torch.Tensor(obs).to(self.device), action)
        elif self.action_type == 'Discrete':
            with torch.no_grad():
                _, log_prob, action_probs = self.get_action(torch.Tensor(next_obs).to(self.device))
                target_Q1, target_Q2 = self.critic_target(torch.Tensor(next_obs).to(self.device))
                target_V = action_probs * (torch.min(target_Q1, target_Q2) - self.alpha.detach() * log_prob)
                target_V = target_V.sum(1).unsqueeze(-1)
                target_Q = reward.flatten() + (not_done.flatten() * self.discount * target_V)
                target_Q = target_Q.detach()
            current_Q1, current_Q2 = self.critic(torch.Tensor(obs).to(self.device))
        # get current Q estimates
        current_Q1 = current_Q1.gather(1, action.long())
        current_Q2 = current_Q2.gather(1, action.long())
        qf1_loss = F.mse_loss(current_Q1, target_Q)
        qf2_loss = F.mse_loss(current_Q2, target_Q)
        critic_loss =  qf1_loss + qf2_loss
        # Use action to take the suitable Q value
        if print_flag:
            logger.log('train_critic/loss', critic_loss, step)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        # qf1_loss.backward(retain_graph=True)
        # qf2_loss.backward(retain_graph=True)
        critic_loss.backward()
        self.critic_optimizer.step()
        self.critic.log(logger, step)
        
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
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        self.critic.log(logger, step)
    
    def save(self, model_dir, step):
        torch.save(
            self.actor.state_dict(), '%s/actor_%s.pt' % (model_dir, step)
        )
        torch.save(
            self.critic.state_dict(), '%s/critic_%s.pt' % (model_dir, step)
        )
        torch.save(
            self.critic_target.state_dict(), '%s/critic_target_%s.pt' % (model_dir, step)
        )
        
    def load(self, model_dir, step):
        self.actor.load_state_dict(
            torch.load('%s/actor_%s.pt' % (model_dir, step))
        )
        self.critic.load_state_dict(
            torch.load('%s/critic_%s.pt' % (model_dir, step))
        )
        self.critic_target.load_state_dict(
            torch.load('%s/critic_target_%s.pt' % (model_dir, step))
        )
    
    def update_actor_and_alpha(self, obs, logger, step, print_flag=False):   
        if self.action_type == 'Continuous':
            action, log_prob, _ = self.get_action(obs)
            actor_Q1, actor_Q2 = self.critic(torch.Tensor(obs).to(self.device), action)
            actor_Q = torch.min(actor_Q1, actor_Q2)
            actor_loss = (self.alpha.detach() * log_prob - actor_Q).mean()
        elif self.action_type == 'Discrete':
            _, log_prob, action_probs = self.get_action(obs)
            with torch.no_grad():
                actor_Q1, actor_Q2 = self.critic(torch.Tensor(obs).to(self.device))
                actor_Q = torch.min(actor_Q1, actor_Q2)
            inside_term = (self.alpha.detach() * log_prob) - actor_Q
            actor_loss = (action_probs*inside_term).sum(dim=1).mean()
            
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
            if self.action_type == 'Continuous':
                with torch.no_grad():
                    _, log_prob, _ = self.get_action(obs)
                alpha_loss = (-self.log_alpha.exp() * (log_prob + self.target_entropy).detach()).mean()
            elif self.action_type == 'Discrete':
                alpha_loss = (action_probs.detach() * (-self.log_alpha.exp() * (log_prob + self.target_entropy).detach())).mean()
            
            if print_flag:
                logger.log('train_alpha/loss', alpha_loss, step)
                logger.log('train_alpha/value', self.alpha, step)
            self.log_alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.log_alpha_optimizer.step()
            
    def update(self, replay_buffer, logger, step, total_timesteps, gradient_update=1):
        for index in range(gradient_update):
            obs, action, reward, next_obs, not_done, not_done_no_max = replay_buffer.sample(self.batch_size)

            print_flag = False
            if index == gradient_update -1:
                logger.log('train/batch_reward', reward.mean(), step)
                print_flag = True
                
            if step % self.critic_update_frequency == 0:
                self.update_critic(obs, action, reward, next_obs, not_done_no_max, logger, step, print_flag)

            if step % self.actor_update_frequency == 0:
                for _ in range(self.actor_update_frequency):
                    self.update_actor_and_alpha(obs, logger, step, print_flag)
                    if self.action_type == 'Discrete': break # Do not update actor as in TD3 for Discrete environments

        if step % self.critic_target_update_frequency == 0:
            self.soft_update_params(self.critic, self.critic_target, self.critic_tau)
            
    def update_state_ent(self, replay_buffer, logger, step, total_timesteps, gradient_update=1, K=5):
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

        if step % self.critic_target_update_frequency == 0:
            soft_update_params(self.critic, self.critic_target, self.critic_tau)
    
    def update_after_reset(self, replay_buffer, logger, step, total_timesteps, gradient_update=1, policy_update=True):
        for index in range(gradient_update):
            obs, action, reward, next_obs, not_done, not_done_no_max = replay_buffer.sample(
                self.batch_size)

            print_flag = False
            if index == gradient_update -1:
                logger.log('train/batch_reward', reward.mean(), step)
                print_flag = True
                
            self.update_critic(obs, action, reward, next_obs, not_done_no_max,
                               logger, step, print_flag)

            if index % self.actor_update_frequency == 0 and policy_update:
                self.update_actor_and_alpha(obs, logger, step, print_flag)

            if index % self.critic_target_update_frequency == 0:
                self.soft_update_params(self.critic, self.critic_target,
                                         self.critic_tau)
    
def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
