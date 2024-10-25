import numpy as np
import torch
import torch.nn.functional as F
import gymnasium as gym
import os
import random
import math
from copy import deepcopy
from collections import deque

def record_video_wrap(env, cfg):
    cfg.video_location = f'clips/{cfg.domain}/{cfg.env}/'
    env = gym.wrappers.RecordVideo(env, cfg.video_location, 
                                   episode_trigger=lambda t: t % cfg.record_frequency == 0, 
                                   disable_logger=True
                                   )
    return env, cfg
        
def make_env(cfg, render_mode=None):
    #gym.logger.set_level(40) ???
    print('GYM LOCATION: ',gym.__file__)
    
    env = None
    env_info = {}
    if 'Control' in cfg.domain:
        envlistD = ['Acrobot-','CartPole-','MountainCar-']
        envlistC=['MountainCarContinuous-','Pendulum-']
        if any(i in cfg.env for i in envlistD):
            id = cfg.env
            env = gym.make(id=id, render_mode=render_mode)  
            env = gym.wrappers.RecordEpisodeStatistics(env)
            # env = gym.wrappers.TimeLimit(env, env._max_episode_steps)
            if cfg.save_video:
                env, cfg = record_video_wrap(env, cfg)

            cfg.action_type = 'Discrete'
            cfg.state_type = 'tabular'
            cfg.architecture = 'MLP'
            cfg.mode = 0
            cfg.action_space = [1]
            obs_space = env.observation_space
            cfg.action_dim = env.action_space.shape
            cfg.agent.obs_dim = obs_space.shape
            cfg.agent.action_dim = int(env.action_space.n)
            cfg.agent.action_cfg = cfg.discrete_action
            cfg.agent.action_cfg.batch_size = 256
            cfg.agent.action_range = [0,1]
        elif any(i in cfg.env for i in envlistC):
            id = cfg.env
            env = gym.make(id=id, render_mode=render_mode)  
            env = gym.wrappers.RecordEpisodeStatistics(env)
            # env = gym.wrappers.TimeLimit(env, env._max_episode_steps)
            if cfg.save_video:
                env, cfg = record_video_wrap(env, cfg)

            cfg.action_type = 'Continuous'
            cfg.state_type = 'tabular'
            cfg.architecture = 'MLP'
            cfg.mode = 0
            cfg.action_space = env.action_space.shape
            obs_space = env.observation_space
            cfg.action_dim = env.action_space.shape[0]
            cfg.agent.obs_dim = obs_space.shape[0]
            cfg.agent.action_dim = cfg.action_dim
            cfg.agent.action_cfg = cfg.continuous_action
            cfg.agent.action_cfg.batch_size = 256
            cfg.agent.action_range = [
                float(env.action_space.low.min()),
                float(env.action_space.high.max())
            ]
            #cfg.agent.actor_cfg = '${diag_gaussian_actor}' find another way to do this
        else:
            raise NotImplementedError
    elif 'MuJoCo' in cfg.domain:
        #Helper function to create MUJOCO environment
        id = cfg.env
        env = gym.make(id=id, render_mode=render_mode)  
        env = gym.wrappers.RecordEpisodeStatistics(env)
        # env = gym.wrappers.TimeLimit(env, env._max_episode_steps) #MuJoCo
        if cfg.save_video:
            env, cfg = record_video_wrap(env, cfg)

        cfg.action_type = 'Continuous'
        cfg.state_type = 'tabular'
        cfg.architecture = 'MLP'
        cfg.mode = 0
        cfg.action_space = env.action_space.shape
        obs_space = env.observation_space
        cfg.action_dim = env.action_space.shape[0]
        cfg.agent.obs_dim = obs_space.shape[0]
        cfg.agent.action_dim = cfg.action_dim
        cfg.agent.action_cfg = cfg.continuous_action
        cfg.agent.action_cfg.batch_size = 256
        cfg.agent.action_range = [
            float(env.action_space.low.min()),
            float(env.action_space.high.max())
        ]
        #cfg.agent.actor_cfg = '${diag_gaussian_actor}' find another way to do this
    elif 'MiniGrid' in cfg.domain or 'BabyAI' in cfg.domain:
        import minigrid

        minigrid.register_minigrid_envs()
        # gym.pprint_registry()
        #Helper function to create MiniGrid environment
        id = cfg.domain+'-'+cfg.env
        env = gym.make(id=id, render_mode=render_mode)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.TimeLimit(env, max_episode_steps = 100)
        
        # env = ResizeObservation(env,64)
        # env = FullyObsWrapper(env)
        # env = PositionBonus(env)
        # env = RGBImgObsWrapper(tile_size = 8)
        if cfg.save_video:
            env, cfg = record_video_wrap(env, cfg)

        cfg.action_type = 'Discrete'
        cfg.state_type = 'pixel-grid'
        cfg.architecture = 'MLP'
        cfg.mode = 1
        cfg.action_space = [1]
        obs_space = env.observation_space['image']
        sp = list(obs_space.shape) # Reorder first 2 dimensions to match state shape
        # obs_space_shape = sp[1], sp[0], sp[2] # Test
        cfg.action_dim = env.action_space.shape
        # cfg.agent.obs_dim = obs_space_shape # Test
        cfg.agent.obs_dim = sp[1], sp[0], sp[2] # Test
        cfg.agent.action_dim = int(env.action_space.n)
        cfg.agent.action_cfg = cfg.discrete_action
        cfg.agent.action_cfg.batch_size = 64
        cfg.agent.action_range = [0,1]
    elif 'ALE' in cfg.domain:
        import ale_py
        # from stable_baselines3.common.atari_wrappers import(
        #     ClipRewardEnv,
        #     FireResetEnv,
        # )
        # gym.register_envs(ale_py)
        #Helper function to create Atari environment
        id=cfg.domain+'/'+cfg.env
        env = gym.make(id=id,render_mode=render_mode)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.AtariPreprocessing(env, noop_max=cfg.noop_max, 
                                              frame_skip=1, 
                                              screen_size=84, 
                                              grayscale_obs=cfg.grayscale_obs, 
                                              scale_obs=cfg.scale_obs)
        env = gym.wrappers.MaxAndSkipObservation(env, skip=cfg.frameskip)
        # env = gym.wrappers.TimeLimit(env, max_episode_steps = 1000)
        # if "FIRE" in env.unwrapped.get_action_meanings():
        #     env = FireResetEnv(env)
        # env = ClipRewardEnv(env)
        env = gym.wrappers.FrameStackObservation(env, cfg.frame_stack)
        if cfg.save_video:
            env, cfg = record_video_wrap(env, cfg)

        cfg.action_type = 'Discrete'
        cfg.state_type = 'pixels'
        cfg.architecture = 'CNN'
        cfg.mode = 0
        cfg.action_space = [1]
        obs_space = env.observation_space
        cfg.action_dim = env.action_space.shape
        cfg.agent.obs_dim = obs_space.shape
        cfg.agent.action_dim = int(env.action_space.n)
        cfg.agent.action_cfg = cfg.discrete_action
        cfg.agent.action_cfg.batch_size = 64
        cfg.agent.action_range = [0,1]
    elif 'Box2D' in cfg.domain:
        if 'LunarLander' in cfg.env:
            id = cfg.env
            env = gym.make(id=id, render_mode=render_mode)
            env = gym.wrappers.RecordEpisodeStatistics(env)
            # env = TimeLimit(env, max_episode_steps = 100)
            if cfg.save_video:
                env, cfg = record_video_wrap(env, cfg)
            
            cfg.action_type = 'Discrete'
            cfg.state_type = 'tabular'
            cfg.architecture = 'MLP'
            cfg.mode = 0
            cfg.action_space = [1]
            obs_space = env.observation_space
            cfg.action_dim = env.action_space.shape
            cfg.agent.obs_dim = obs_space.shape
            cfg.agent.action_dim = int(env.action_space.n)
            cfg.agent.action_cfg = cfg.discrete_action
            cfg.agent.action_cfg.batch_size = 64
            cfg.agent.action_range = [0,1]
        elif 'BipedalWalker' in cfg.env:
            id = cfg.env
            env = gym.make(id=id, render_mode=render_mode)
            env = gym.wrappers.RecordEpisodeStatistics(env)
            # env = TimeLimit(env, max_episode_steps = 100)
            if cfg.save_video:
                env, cfg = record_video_wrap(env, cfg)

            cfg.action_type = 'Continuous'
            cfg.state_type = 'tabular'
            cfg.architecture = 'MLP'
            cfg.mode = 0
            cfg.action_space = env.action_space.shape
            obs_space = env.observation_space
            cfg.action_dim = env.action_space.shape[0]
            cfg.agent.obs_dim = obs_space.shape[0]
            cfg.agent.action_dim = cfg.action_dim
            cfg.agent.action_cfg = cfg.continuous_action
            cfg.agent.action_cfg.batch_size = 256
            cfg.agent.action_range = [
                float(env.action_space.low.min()),
                float(env.action_space.high.max())
            ]
        elif 'CarRacing' in cfg.env:
            id = cfg.env
            env = gym.make(id=id, render_mode=render_mode)
            env = gym.wrappers.RecordEpisodeStatistics(env)
            # env = TimeLimit(env, max_episode_steps = 100)
            env = gym.wrappers.GrayscaleObservation(env)
            env = gym.wrappers.ResizeObservation(env, (64, 64))
            env = gym.wrappers.MaxAndSkipObservation(env, skip=cfg.frameskip)
            env = gym.wrappers.FrameStackObservation(env, cfg.frame_stack)
            if cfg.save_video:
                env, cfg = record_video_wrap(env, cfg)

            cfg.action_type = 'Continuous'
            cfg.state_type = 'pixels'
            cfg.architecture = 'CNN'
            cfg.mode = 0
            cfg.action_space = env.action_space.shape
            obs_space = env.observation_space
            cfg.action_dim = env.action_space.shape[0]
            cfg.agent.obs_dim = obs_space.shape
            cfg.agent.action_dim = cfg.action_dim
            cfg.agent.action_cfg = cfg.continuous_action
            cfg.agent.action_cfg.batch_size = 256
            cfg.agent.action_range = [
                float(env.action_space.low.min()),
                float(env.action_space.high.max())
            ]
        else:
            raise NotImplementedError
    elif 'highway-env' in cfg.domain:
        import highway_env      # Wait for official update

        highway_env.register_highway_envs()
        # gym.pprint_registry()
        id = cfg.env
        env = gym.make(id=id, render_mode=None)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        # env = TimeLimit(env, max_episode_steps = 100)
        if cfg.save_video:
            env, cfg = record_video_wrap(env, cfg)

        cfg.action_type = 'Discrete'
        cfg.state_type = 'tabular'
        cfg.architecture = 'MLP'
        cfg.mode = 0
        cfg.action_space = [1]
        obs_space = env.observation_space
        cfg.action_dim = env.action_space.shape
        cfg.agent.obs_dim = obs_space.shape
        cfg.agent.action_dim = int(env.action_space.n)
        cfg.agent.action_cfg = cfg.discrete_action
        cfg.agent.action_cfg.batch_size = 256
        cfg.agent.action_range = [0,1]
    else:
        raise NotImplementedError
    return env, cfg, obs_space


# class RewindWrapper(gym.Wrapper):
#     def __init__(self, env, domain):
#         super().__init__(env)

#         self.domain = domain

#     def get_state(self):
#         if self.domain == 'ALE':
#             return self.env.ale.cloneState()
#         elif self.domain == 'MiniGrid':
#             return deepcopy(self.env)
#         elif self.domain == 'BabyAI':
#             return deepcopy(self.env)
#         elif self.domain == 'Control':
#             return self.env.get_data()
#         elif self.domain == 'Box2D':
#             return self.env.get_state()
#         else:
#             print('You need to provide the get snapshot functionality of your environment')
#             raise NotImplementedError

#     def set_state(self, snapshot):
#         if self.domain == 'ALE':
#             self.env.ale.restoreState(snapshot)
#         elif self.domain == 'MiniGrid':
#             self.env = deepcopy(snapshot)
#         elif self.domain == 'BabyAI':
#             self.env = deepcopy(snapshot)
#         elif self.domain == 'Control':
#             self.env.set_state()
#         elif self.domain == 'Box2D':
#             self.env.set_state()
#         else:
#             print('You need to provide the set snapshot functionality of your environment')
#             raise NotImplementedError