#!/usr/bin/env python3
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math
import os
import sys
from pathlib import Path
import time
import pickle as pkl
from tqdm import tqdm
import copy

from lib.logger import Logger
from collections import deque

import lib.env_setup as env_setup
import lib.agent_setup as agent_setup
import lib.utils as utils
from lib.trajectory_io import TrajectoryProcessor
import hydra
from omegaconf import DictConfig
from termcolor import colored

class Workspace(object):
    def __init__(self, cfg, work_dir):
        print(colored('EXECUTING GENERATE TRAJECTORIES','green'))
        self.work_dir = work_dir
        
        self.logger = Logger(
            cfg.models_dir,
            save_tb=cfg.log_save_tb,
            seed=cfg.seed,
            log_frequency=cfg.log_frequency,
            agent=cfg.agent.name)

        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)

        cfg.save_video=True
        self.env, cfg, self.obs_space = env_setup.make_env(cfg, cfg.render_mode)
        self.cfg = cfg

        actor_cfg, critic_cfg = agent_setup.config_agent(cfg)
        self.agent, _ = agent_setup.create_agent(cfg, actor_cfg, critic_cfg, cfg.agent.action_cfg, self.obs_space)

        # LOAD TRAINED AGENT
        self.agent, _ = agent_setup.load_agent(work_dir, cfg, self.agent)
        
        self.traj_proc = TrajectoryProcessor(work_dir, cfg)
        print('INIT COMPLETE')
        
    @property
    def global_step(self):
        return self.step

    @property
    def global_episode(self):
        return self.episode

    @property
    def global_frame(self):
        return self.step #* self.cfg.action_repeat

    def run(self):
        average_episode_reward = 0
        total_episode_reward = 0
        average_true_episode_reward = 0
        total_true_episode_reward = 0
        success_rate = 0
        self.step = 0
        total_time = 0
        start_time = time.time()
        
        for episode in tqdm(range(self.cfg.episodes_to_gen), desc="GENERATING TRAJECTORIES: "):
            # obs, _ = self.env.reset(seed = self.cfg.seed)
            obs, _ = self.env.reset()
            obs, _, _, _, _ = self.env.step(1) # FIRE action for breakout
            
            if self.cfg.action_type == 'Discrete' and self.cfg.state_type == 'grid':
                obs = obs['image']
            
            terminated = False
            truncated = False
            episode_reward = 0
            true_episode_reward = 0
            if self.cfg.log_success:
                episode_success = 0

            while not (terminated or truncated):
                with agent_setup.eval_mode(self.agent):
                    action, _, _ = self.agent.get_action(torch.FloatTensor(obs).to(self.device).unsqueeze(0))
                    action = action.detach().cpu().numpy()[0]

                next_obs, reward, terminated, truncated, info = self.env.step(action)
                if self.cfg.action_type == 'Discrete' and self.cfg.state_type == 'grid':
                    next_obs = next_obs['image']

                frame = self.env.render()
                
                # ADD SAVE STATE to files
                self.traj_proc.add(frame, obs, action, reward)

                episode_reward += reward
                true_episode_reward += reward

                if self.cfg.log_success:
                    episode_success = max(episode_success, terminated)
                
                obs = next_obs
                self.step += 1
            self.traj_proc.save(self.traj_proc.traj_dir,episode)

            total_episode_reward += episode_reward
            total_true_episode_reward += true_episode_reward
            if self.cfg.log_success:
                success_rate += episode_success
        print('TRAJECTORIES GENERATED')

        total_time = time.time() - start_time
        self.env.close()
        average_episode_reward = total_episode_reward / self.cfg.episodes_to_gen
        average_true_episode_reward = total_true_episode_reward / self.cfg.episodes_to_gen
        if self.cfg.log_success:
            success_rate /= self.cfg.episodes_to_gen
            success_rate *= 100.0    

        self.logger.log('eval/episode', episode+1, self.global_step)
        self.logger.log('eval/avg_episode_reward', average_episode_reward, self.global_step)
        self.logger.log('eval/avg_true_episode_reward', average_true_episode_reward, self.global_step)
        self.logger.log('eval/total_duration', total_time, self.global_step)
        if self.cfg.log_success:
            self.logger.log('eval/success_rate', success_rate, self.global_step)
        self.logger.dump(self.global_step, ty='eval')
        self.logger.close()
        
@hydra.main(version_base=None, config_path="config", config_name='themis_generate_trajectories')
def main(cfg : DictConfig):
    work_dir = Path.cwd()
    # cfg.output_dir = work_dir / cfg.output_dir  

    folder = work_dir / cfg.models_dir
    if not folder.exists():
        print(f"Experiment for {cfg.agent.name}_{cfg.test} with seed {cfg.seed} doesn't exist at {cfg.models_dir}")
        raise FileNotFoundError
    workspace = Workspace(cfg, work_dir)
    
    print(f'Workspace: {work_dir}\nSEED {cfg.seed}')
    workspace.run()
    print(f'Experiment with CPU ID {cfg.cpu_id} and SEED {cfg.seed} finished')

if __name__ == '__main__':
    main()