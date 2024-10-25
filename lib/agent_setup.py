import torch
from agent.sac import SAC
from agent.common.replay_buffer import ReplayBuffer


def config_agent(cfg):
    # Setup Agent
    if 'Control' in cfg.domain:
        if cfg.action_type == 'Discrete':
            actor_cfg = cfg.categorical_actor
        elif cfg.action_type == 'Continuous':
            actor_cfg = cfg.diag_gaussian_actor
        else:
            raise NotImplementedError
        critic_cfg = cfg.double_q_critic
        
        critic_cfg.action_type = cfg.action_type
        critic_cfg.architecture = cfg.architecture
        
        actor_cfg.action_type = cfg.action_type
        actor_cfg.architecture = cfg.architecture
    elif 'MuJoCo' in cfg.domain:
        actor_cfg = cfg.diag_gaussian_actor
        critic_cfg = cfg.double_q_critic
        
        critic_cfg.action_type = cfg.action_type
        critic_cfg.architecture = cfg.architecture
        
        actor_cfg.action_type = cfg.action_type
        actor_cfg.architecture = cfg.architecture
    elif 'MiniGrid' in cfg.domain or 'BabyAI' in cfg.domain:
        actor_cfg = cfg.categorical_actor
        critic_cfg = cfg.double_q_critic

        actor_cfg.action_type = cfg.action_type
        actor_cfg.architecture = cfg.architecture
        critic_cfg.action_type = cfg.action_type
        critic_cfg.architecture = cfg.architecture
    elif 'ALE' in cfg.domain:
        actor_cfg = cfg.categorical_actor
        critic_cfg = cfg.double_q_critic
        
        actor_cfg.action_type = cfg.action_type
        actor_cfg.architecture = cfg.architecture
        critic_cfg.action_type = cfg.action_type
        critic_cfg.architecture = cfg.architecture
    elif 'Box2D' in cfg.domain:
        if 'LunarLander' in cfg.env:
            actor_cfg = cfg.categorical_actor
        elif 'BipedalWalker' in cfg.env:
            actor_cfg = cfg.diag_gaussian_actor
        elif 'CarRacing' in cfg.env:
            actor_cfg = cfg.diag_gaussian_actor
        else:
            raise NotImplementedError  
        critic_cfg = cfg.double_q_critic

        actor_cfg.action_type = cfg.action_type
        actor_cfg.architecture = cfg.architecture
        critic_cfg.action_type = cfg.action_type
        critic_cfg.architecture = cfg.architecture
    elif 'highway-env' in cfg.domain:
        actor_cfg = cfg.categorical_actor
        critic_cfg = cfg.double_q_critic

        actor_cfg.action_type = cfg.action_type
        actor_cfg.architecture = cfg.architecture
        critic_cfg.action_type = cfg.action_type
        critic_cfg.architecture = cfg.architecture
    else:
        raise NotImplementedError
    
    return actor_cfg, critic_cfg

def create_agent(cfg, actor_cfg, critic_cfg, action_cfg, obs_space):

    agent = SAC(obs_space = obs_space,
            obs_dim = cfg.agent.obs_dim, 
            action_dim = cfg.agent.action_dim, 
            action_range = cfg.agent.action_range, 
            device = torch.device(cfg.device), 
            actor_cfg = actor_cfg,
            critic_cfg = critic_cfg,
            action_cfg = action_cfg, 
            discount = cfg.agent.discount, 
            init_temperature = cfg.agent.init_temperature,
            mode= cfg.mode, 
            learnable_temperature = cfg.agent.learnable_temperature,
            normalize_state_entropy = True)
    
    replay_buffer = ReplayBuffer(
            obs_space,
            cfg.agent.obs_dim,
            cfg.action_space,
            cfg.action_type,
            int(cfg.replay_buffer_capacity), 
            torch.device(cfg.device))
    
    return agent, replay_buffer

class eval_mode(object):
    def __init__(self, *models):
        self.models = models

    def __enter__(self):
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.train(False)

    def __exit__(self, *args):
        for model, state in zip(self.models, self.prev_states):
            model.train(state)
        return False

class train_mode(object):
    def __init__(self, *models):
        self.models = models

    def __enter__(self):
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.train(True)

    def __exit__(self, *args):
        for model, state in zip(self.models, self.prev_states):
            model.train(state)
        return False