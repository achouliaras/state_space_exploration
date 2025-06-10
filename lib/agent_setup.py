import torch
import os
from agent.sac import SAC
from agent.ppo import PPO
from agent.common.replay_buffers import ReplayBuffer

def config_agent(cfg):
    # Setup Agent
    if 'Control' in cfg.domain:
        if cfg.action_type == 'Discrete':
            actor_cfg = cfg.agent.categorical_actor
            critic_cfg = cfg.agent.double_q_critic
        elif cfg.action_type == 'Continuous':
            actor_cfg = cfg.agent.diag_gaussian_actor
            critic_cfg = cfg.agent.double_q_critic
            actor_cfg.action_dim = actor_cfg.action_dim[0]
            critic_cfg.action_dim = critic_cfg.action_dim[0]
        else:
            raise NotImplementedError
        
        critic_cfg.action_type = cfg.action_type
        critic_cfg.state_type = cfg.state_type
        critic_cfg.architecture = cfg.architecture

        actor_cfg.action_type = cfg.action_type
        actor_cfg.architecture = cfg.architecture
    elif 'MuJoCo' in cfg.domain:
        actor_cfg = cfg.agent.diag_gaussian_actor
        critic_cfg = cfg.agent.double_q_critic
        
        critic_cfg.action_type = cfg.action_type
        critic_cfg.architecture = cfg.architecture
        critic_cfg.action_dim = critic_cfg.action_dim[0]
        
        actor_cfg.action_type = cfg.action_type
        critic_cfg.state_type = cfg.state_type
        actor_cfg.architecture = cfg.architecture
        actor_cfg.action_dim = actor_cfg.action_dim[0]
    elif 'MiniGrid' in cfg.domain or 'BabyAI' in cfg.domain:
        actor_cfg = cfg.agent.categorical_actor
        critic_cfg = cfg.agent.double_q_critic

        actor_cfg.action_type = cfg.action_type
        actor_cfg.architecture = cfg.architecture
        critic_cfg.action_type = cfg.action_type
        critic_cfg.state_type = cfg.state_type
        critic_cfg.architecture = cfg.architecture
    elif 'ALE' in cfg.domain:
        actor_cfg = cfg.agent.categorical_actor
        critic_cfg = cfg.agent.double_q_critic
        
        actor_cfg.action_type = cfg.action_type
        actor_cfg.architecture = cfg.architecture
        critic_cfg.action_type = cfg.action_type
        critic_cfg.state_type = cfg.state_type
        critic_cfg.architecture = cfg.architecture
    elif 'Box2D' in cfg.domain:
        if 'LunarLander' in cfg.env:
            actor_cfg = cfg.agent.categorical_actor
            critic_cfg = cfg.agent.double_q_critic
        elif 'BipedalWalker' in cfg.env:
            actor_cfg = cfg.agent.diag_gaussian_actor
            critic_cfg = cfg.agent.double_q_critic
            actor_cfg.action_dim = actor_cfg.action_dim[0]
            critic_cfg.action_dim = critic_cfg.action_dim[0]
        elif 'CarRacing' in cfg.env:
            actor_cfg = cfg.agent.diag_gaussian_actor
            critic_cfg = cfg.agent.double_q_critic
        else:
            raise NotImplementedError  

        actor_cfg.action_type = cfg.action_type
        actor_cfg.architecture = cfg.architecture
        critic_cfg.action_type = cfg.action_type
        critic_cfg.state_type = cfg.state_type
        critic_cfg.architecture = cfg.architecture
    elif 'highway-env' in cfg.domain:
        actor_cfg = cfg.agent.categorical_actor
        critic_cfg = cfg.agent.double_q_critic

        actor_cfg.action_type = cfg.action_type
        actor_cfg.architecture = cfg.architecture
        critic_cfg.action_type = cfg.action_type
        critic_cfg.state_type = cfg.state_type
        critic_cfg.architecture = cfg.architecture
    else:
        raise NotImplementedError
    
    return actor_cfg, critic_cfg

def create_agent(cfg, obs_space=None):

    if cfg.agent.name == 'SAC':
        actor_cfg, critic_cfg = config_agent(cfg)

        agent = SAC(obs_space = obs_space,
                    obs_dim = cfg.agent.obs_dim,
                    action_range = cfg.agent.action_range, 
                    device = torch.device(cfg.device), 
                    actor_cfg = actor_cfg,
                    critic_cfg = critic_cfg,
                    action_cfg = cfg.agent.action_cfg, 
                    discount = cfg.agent.discount, 
                    init_temperature = cfg.agent.init_temperature,
                    mode= cfg.mode, 
                    learnable_temperature = cfg.agent.learnable_temperature,
                    normalize_state_entropy = True)
        if cfg.deploy_mode:
            replay_buffer = ReplayBuffer(
                    obs_space,
                    cfg.agent.obs_shape,
                    cfg.action_space,
                    cfg.action_type,
                    int(cfg.replay_buffer_capacity), 
                    torch.device(cfg.device))
        else:
            replay_buffer = None
        return agent, replay_buffer
    elif cfg.agent.name == 'PPO':
        agent = PPO(obs_dim = cfg.agent.obs_dim, 
                    action_type= cfg.action_type,
                    device = torch.device(cfg.device), 
                    latent_dim = cfg.latent_dim,
                    architecture=cfg.architecture,
                    state_type=cfg.state_type, 
                    agent_cfg = cfg.agent,
                    cfg = cfg, 
                    import_protocol = cfg.import_protocol,
                    deploy_mode = cfg.deploy_mode,
                    mode= cfg.mode,
                    normalize_state_entropy = True)
        return agent
    else:
        raise NotImplementedError   

def save_agent(agent, replay_buffer, payload, work_dir, cfg, global_frame, mode = 'NORMAL'):
    if 'OFFLINE' in mode:
        models_dir = work_dir / cfg.models_dir / 'offline_models'
        global_frame=cfg.offline_epochs
        print(f'Saving OFFLINE pretrained model to: {models_dir}')
    elif 'ONLINE' in mode:
        models_dir = work_dir / cfg.models_dir / 'online_models'
        global_frame=cfg.num_seed_steps + cfg.num_unsup_steps
        print(f'Saving ONLINE pretrained model to: {models_dir}')
    elif "CURRICULUM" in mode:
        models_dir = work_dir / cfg.models_dir / 'models'
        global_frame=cfg.num_train_steps
        print(f'Saving model for CURRICULUM LEARNING to: {models_dir}')
    else:
        models_dir = work_dir / cfg.models_dir / 'models'
    
    os.makedirs(models_dir, exist_ok=True)
    # Save agent's models & replay buffer
    agent.save(models_dir, global_frame, mode)
    if replay_buffer != None:
        replay_buffer.save(models_dir, global_frame)
    # Save experiment variables like step and episode
    snapshot = models_dir / f'snapshot_{global_frame}.pt'
    
    torch.save(payload, snapshot, pickle_protocol=4)

def load_agent(work_dir, cfg, agent, replay_buffer=None, mode = 'NORMAL'):
    if "CURRICULUM" in mode:
        models_dir = work_dir / cfg.curriculum_models_dir / 'models'
        if "OFFLINE" in mode:
            models_dir = work_dir / cfg.curriculum_models_dir / 'offline_models'
        if not models_dir.exists():
            print('NOT EXISTS:', models_dir)
            raise FileNotFoundError()
        if "OFFLINE" in mode:
            print(f'Loading OFFLINE pretrained model for from: {models_dir}')
            global_frame = cfg.offline_epochs
        else:
            print(f'Loading model for CURRICULUM LEARNING from: {models_dir}')
            global_frame = cfg.num_train_steps
        
        if replay_buffer != None:
            replay_buffer.load(models_dir, global_frame)

        agent.load(models_dir, global_frame, mode)
    elif "ONLINE" in mode:
        models_dir = work_dir / cfg.online_models_dir
        if not models_dir.exists():
            print('NOT EXISTS:', models_dir)
            raise FileNotFoundError()
        print(f'Loading ONLINE pretrained model from: {models_dir}')
        global_frame = cfg.num_seed_steps + cfg.num_unsup_steps

        # snapshot = models_dir / f'snapshot_{global_frame}.pt'
        if replay_buffer != None:
            replay_buffer.load(models_dir, global_frame)

        agent.load(models_dir, global_frame, mode)
    elif "OFFLINE" in mode: 
        models_dir = work_dir / cfg.offline_models_dir
        if not models_dir.exists():
            print('NOT EXISTS:', models_dir)
            raise FileNotFoundError()
        print(f'Loading OFFLINE pretrained model from: {models_dir}')
        global_frame=cfg.offline_epochs

        agent.load(models_dir, global_frame, mode)
    else:
        print('No model parameters were loaded')
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