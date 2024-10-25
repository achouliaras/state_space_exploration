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
from lib.eval import evaluate_agent
import hydra
from omegaconf import DictConfig

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
            experiment_name=cfg.experiment + f'_{cfg.seed}',
            log_frequency=cfg.log_frequency,
            agent=cfg.algorithm.name)

        # utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)
        self.env, cfg, self.obs_space = env_setup.make_env(cfg, cfg.render_mode)
        self.cfg = cfg
        
        actor_cfg, critic_cfg = agent_setup.config_agent(cfg)

        self.agent, self.replay_buffer = agent_setup.create_agent(cfg, actor_cfg, critic_cfg, cfg.agent.action_cfg, self.obs_space)
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
        if self.cfg.log_success:
            episode_success = 0
        true_episode_reward = 0
        
        # store train returns of recent 10 episodes
        avg_train_true_return = deque([], maxlen=10) 
        total_time=0
        start_time = time.time()
        obs, info = self.env.reset(seed = self.cfg.seed)

        for global_step in range(self.cfg.num_seed_steps + self.cfg.num_unsup_steps):
            # sample action for data collection
            if global_step < self.cfg.num_seed_steps:
                action = self.env.action_space.sample()
                if global_step == self.cfg.num_seed_steps-1: print('PRETRAINING STARTS')
            else:
                #with utils.eval_mode(self.agent):
                action, _, _ = self.agent.get_action(torch.FloatTensor(obs).to(self.device).unsqueeze(0))
                action = action.detach().cpu().numpy()[0]

            next_obs, reward, terminated, truncated, info = self.env.step(action)
            
            # Push data to replay buffer
            self.replay_buffer.add(obs, action, reward, next_obs, terminated, truncated)

            obs = next_obs

            if global_step > self.cfg.num_seed_steps:
                self.agent.update_state_ent(self.replay_buffer, self.logger, 
                                            global_step, self.cfg.num_train_steps, 
                                            gradient_update=1, K=self.cfg.topK)

            if terminated or truncated:
                if global_step > 0:
                    episode_time = time.time() - start_time
                    self.logger.log('train/duration', episode_time, global_step)
                    total_time += episode_time
                    self.logger.log('train/total_duration', total_time, global_step)
                    start_time = time.time()
                    self.logger.dump(
                        global_step, save=(global_step > self.cfg.num_seed_steps))
                
                self.logger.log('train/episode_reward', episode_reward, global_step)
                self.logger.log('train/true_episode_reward', true_episode_reward, global_step)
                self.logger.log('train/total_feedback', self.total_feedback, global_step)
                self.logger.log('train/labeled_feedback', self.labeled_feedback, global_step)
                
                if self.cfg.log_success:
                    self.logger.log('train/episode_success', episode_success, global_step)
                    self.logger.log('train/true_episode_success', episode_success, global_step)
                
                obs, info = self.env.reset(seed = self.cfg.seed)

                if self.cfg.action_type == 'Discrete' and self.cfg.state_type == 'grid':
                    obs = obs['image']

                # self.agent.reset()
                terminated = False
                truncated = False
                episode_reward = 0
                avg_train_true_return.append(true_episode_reward)
                true_episode_reward = 0
                if self.cfg.log_success:
                    episode_success = 0
                self.episode += 1

                self.logger.log('train/episode', self.episode, global_step)
            
            # allow infinite bootstrap
            terminated = float(terminated)
            # episode_reward += reward_hat
            true_episode_reward += reward
            
            if self.cfg.log_success:
                episode_success = max(episode_success, terminated)
        self.logger.dump(global_step, save=(global_step > self.cfg.num_seed_steps))
        self.env.close()
        print('PRETRAINING FINISHED')
        self.logger = evaluate_agent(self.agent, self.cfg, self.logger)

    def save_snapshot(self):
        print('SAVING STARTS')
        snapshot_dir = self.cfg.snapshot_dir        
        snapshot_dir.mkdir(exist_ok=True, parents=True)
        self.agent.save(snapshot_dir, self.global_frame)
        self.replay_buffer.save(snapshot_dir, self.global_frame)
        snapshot = snapshot_dir / f'snapshot_{self.global_frame}.pt'
        keys_to_save = ['step', 'episode']
        payload = {k: self.__dict__[k] for k in keys_to_save}
        torch.save(payload, snapshot, pickle_protocol=4)
        print('SAVING COMPLETE')
        
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