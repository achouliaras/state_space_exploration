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
import tqdm
import copy

from lib.logger import Logger
from collections import deque

import lib.env_setup as env_setup
import lib.agent_setup as agent_setup
import hydra
from omegaconf import DictConfig
from gymnasium.spaces import utils as gym_utils

class Workspace(object):
    def __init__(self, cfg, work_dir):
        self.work_dir = work_dir
        print(f'Workspace: {self.work_dir}')

        folder = work_dir / cfg.checkpoints_dir    
        folder.mkdir(exist_ok=True, parents=True)
        self.checkpoints_dir = cfg.checkpoints_dir

        self.logger = Logger(
            self.work_dir,
            save_tb=cfg.log_save_tb,
            log_frequency=cfg.log_frequency,
            agent=cfg.algorithm.name)

        # utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)
        self.env, cfg, self.obs_space = env_setup.make_env(cfg, cfg.render_mode)
        self.cfg = cfg
        
        actor_cfg, critic_cfg = agent_setup.config_agent(cfg)
        self.log_success = True

        # for logging
        self.total_feedback = 0
        self.labeled_feedback = 0
        self.step = 0
        self.episode=0
        self.interactions=0
        
        print('INIT COMPLETE')
        
    @property
    def global_step(self):
        return self.step

    @property
    def global_episode(self):
        return self.episode

    @property
    def global_frame(self):
        return self.step * self.cfg.action_repeat

    def run(self):
        self.episode, episode_reward, terminated, truncated = 0, 0, True, False
        if self.log_success:
            episode_success = 0
        true_episode_reward = 0
        
        # store train returns of recent 10 episodes
        avg_train_true_return = deque([], maxlen=10) 
        total_time=0
        start_time = time.time()

        interact_count = 0
        while self.step != (self.cfg.num_seed_steps + self.cfg.num_unsup_steps):
            if terminated or truncated:
                if self.step > 0:
                    episode_time = time.time() - start_time
                    self.logger.log('train/duration', episode_time, self.step)
                    total_time += episode_time
                    self.logger.log('train/total_duration', total_time, self.step)
                    start_time = time.time()
                    self.logger.dump(
                        self.step, save=(self.step > self.cfg.num_seed_steps))
                
                self.logger.log('train/episode_reward', episode_reward, self.step)
                self.logger.log('train/true_episode_reward', true_episode_reward, self.step)
                self.logger.log('train/total_feedback', self.total_feedback, self.step)
                self.logger.log('train/labeled_feedback', self.labeled_feedback, self.step)
                
                if self.log_success:
                    self.logger.log('train/episode_success', episode_success, self.step)
                    self.logger.log('train/true_episode_success', episode_success, self.step)
                
                obs, info = self.env.reset(seed = self.cfg.seed)

                if self.cfg.action_type == 'Discrete' and self.cfg.state_type == 'grid':
                    obs = obs['image']

                # self.agent.reset()
                terminated = False
                truncated = False
                episode_reward = 0
                avg_train_true_return.append(true_episode_reward)
                true_episode_reward = 0
                if self.log_success:
                    episode_success = 0
                episode_step = 0
                self.episode += 1

                self.logger.log('train/episode', self.episode, self.step)

            # # For Video generation
            # if self.cfg.state_type == 'grid' or self.cfg.state_type == 'tabular':
            #     env_snapshot = [] # Not yet supported
            # elif self.cfg.state_type == 'pixel-grid':
            #     env_snapshot = self.env.get_state() 
            # elif self.cfg.state_type == 'pixels':
            #     env_snapshot = self.env.get_state()
            # else:
            #     env_snapshot = [] # Not yet supported

            action = self.env.action_space.sample()
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            
            if self.cfg.action_type == 'Discrete':
                next_obs = next_obs['image'] if self.cfg.state_type == 'grid' else next_obs
                action = np.array([action], dtype=np.uint8)

            # obs_flat = gym_utils.flatten(self.obs_space, obs)

            # reward_hat = self.reward_model.r_hat(np.concatenate([obs_flat, action], axis=-1))

            # allow infinite bootstrap
            terminated = float(terminated)
            # episode_reward += reward_hat
            true_episode_reward += reward
            
            if self.log_success:
                episode_success = max(episode_success, terminated)

            obs = next_obs
            episode_step += 1
            self.step += 1
            interact_count += 1
        self.env.close()

    def save_snapshot(self):
        snapshot_dir = self.cfg.snapshot_dir        
        snapshot_dir.mkdir(exist_ok=True, parents=True)
        self.agent.save(snapshot_dir, self.global_frame)
        self.replay_buffer.save(snapshot_dir, self.global_frame)
        self.reward_model.save(snapshot_dir, self.global_frame)
        snapshot = snapshot_dir / f'snapshot_{self.global_frame}.pt'
        keys_to_save = ['step', 'episode']
        payload = {k: self.__dict__[k] for k in keys_to_save}
        torch.save(payload, snapshot, pickle_protocol=4)
        
@hydra.main(version_base=None, config_path="config", config_name='themis_pretrain')
def main(cfg : DictConfig):
    work_dir = Path.cwd()
    workspace = Workspace(cfg, work_dir)
    cfg.snapshot_dir = work_dir / cfg.snapshot_dir
    snapshot = cfg.snapshot_dir / f'snapshot_{cfg.num_seed_steps + cfg.num_unsup_steps}.pt'
    if snapshot.exists():
        print(f'Snapshot seems to already exist at {cfg.snapshot_dir}')
        print('Do you want to overwrite it?\n')
        answer = input('[y]/n \n')
        if answer in ['n','no','No']: exit()
    workspace.run()
    if snapshot.exists():
        print(f'Overwriting models at: {cfg.snapshot_dir}')
    else:
        print(f'Creating models at: {cfg.snapshot_dir}')
    # workspace.save_snapshot()

if __name__ == '__main__':
    main()