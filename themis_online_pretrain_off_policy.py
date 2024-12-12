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
import lib.utils as utils
from lib.eval import evaluate_agent
import hydra
from omegaconf import DictConfig
from termcolor import colored

# Interact with environment for data generation
#  online Curiosity Driven Exploration
#  online Skill Discovery
#  online Data Coverage Maximization
# Save resulting agent

class Workspace(object):
    def __init__(self, cfg, work_dir):
        print(colored('EXECUTING PRETRAINING', 'green'))
        self.work_dir = work_dir
        
        self.logger = Logger(
            cfg.models_dir,
            save_tb=cfg.log_save_tb,
            seed=cfg.seed,
            log_frequency=cfg.log_frequency,
            agent=cfg.agent.name)

        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)
        self.env, cfg, self.obs_space = env_setup.make_env(cfg, cfg.render_mode)
        self.env.action_space.seed(cfg.seed)
        self.cfg = cfg
        
        self.agent, self.replay_buffer = agent_setup.create_agent(cfg, cfg.agent.action_cfg, self.obs_space)
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
        return self.step #* self.cfg.action_repeat

    def run(self):
        self.episode, episode_reward, terminated, truncated = 0, 0, False, False
        if self.cfg.log_success:
            episode_success = 0
        true_episode_reward = 0
        
        # store train returns of recent 10 episodes
        # avg_train_true_return = deque([], maxlen=10) 
        total_time=0
        start_time = time.time()

        obs, _ = self.env.reset(seed = self.cfg.seed)
        # obs, _ = self.env.reset()
        obs, _, _, _, _ = self.env.step(1) # FIRE action for breakout

        for global_step in range(int(self.cfg.num_seed_steps + self.cfg.num_unsup_steps+1)):
            # sample action for data collection
            if global_step < self.cfg.num_seed_steps:
                action = self.env.action_space.sample()
            else:
                #with utils.eval_mode(self.agent):
                action, _, _ = self.agent.get_action(torch.FloatTensor(obs).to(self.device).unsqueeze(0))
                action = action.detach().cpu().numpy()[0]
                # print(action)

            next_obs, reward, terminated, truncated, info = self.env.step(action)

            if terminated or truncated:
                episode_time = time.time() - start_time
                total_time += episode_time
                self.logger.log('train/episode', self.episode, global_step)
                self.logger.log('train/episode_reward', episode_reward, global_step)
                self.logger.log('train/true_episode_reward', true_episode_reward, global_step)
                self.logger.log('train/duration', episode_time, global_step)
                self.logger.log('train/total_duration', total_time, global_step)
                if self.cfg.log_success:
                    self.logger.log('train/episode_success', episode_success, global_step)
                    self.logger.log('train/true_episode_success', episode_success, global_step)

                self.logger.dump(global_step, save=(global_step > self.cfg.num_seed_steps), ty='train')
                start_time = time.time()

                # episode_reward = 0
                # avg_train_true_return.append(true_episode_reward)
                true_episode_reward = 0
                if self.cfg.log_success:
                    episode_success = 0
                self.episode += 1
                self.step = global_step
                next_obs, _ = self.env.reset()
                next_obs, _, _, _, _ = self.env.step(1) # FIRE action for breakout
                

            # Push data to replay buffer
            self.replay_buffer.add(obs, action, reward, next_obs, terminated, truncated)


            obs = next_obs
            true_episode_reward += reward

            # Pre-Training Update
            if global_step >= self.cfg.num_seed_steps:
                self.agent.pretrain_update(self.replay_buffer, self.logger, 
                                            global_step, self.cfg.num_train_steps, 
                                            gradient_update=1, K=self.cfg.topK)
            elif global_step == self.cfg.num_seed_steps-1: 
                obs, _ = self.env.reset()
                obs, _, _, _, _ = self.env.step(1) # FIRE action for breakout
                                
                self.episode = 0
                true_episode_reward = 0
                print('PRETRAINING STARTS')
            
            
            if self.cfg.log_success:
                episode_success = max(episode_success, terminated)

        episode_time = time.time() - start_time
        total_time += episode_time
        self.logger.log('train/episode', self.episode, global_step)
        self.logger.log('train/episode_reward', episode_reward, global_step)
        self.logger.log('train/true_episode_reward', true_episode_reward, global_step)
        self.logger.log('train/duration', episode_time, global_step)
        self.logger.log('train/total_duration', total_time, global_step)
        if self.cfg.log_success:
            self.logger.log('train/episode_success', episode_success, global_step)
            self.logger.log('train/true_episode_success', episode_success, global_step)
        
        self.step = global_step
        self.logger.dump(global_step, ty='train')
        self.env.close()
        print('PRETRAINING FINISHED')
        self.logger = evaluate_agent(self.agent, self.cfg, self.logger)
        self.logger.close()

    def save_results(self):
        print('SAVING STARTS')
        keys_to_save = ['step', 'episode']
        payload = {k: self.__dict__[k] for k in keys_to_save}

        agent_setup.save_agent(self.agent, self.replay_buffer, payload, self.work_dir, self.cfg, self.global_step)
        print('SAVING COMPLETED')
        
@hydra.main(version_base=None, config_path="config", config_name='themis_online_pretrain_off_policy')
def main(cfg : DictConfig):
    work_dir = Path.cwd()
    # cfg.output_dir = work_dir / cfg.output_dir  
    
    print(cfg.agent.name)
    print(cfg.agent.init_temperature)

    folder = work_dir / cfg.models_dir
    if folder.exists():
        print(f'Experiment for {cfg.agent.name}_{cfg.test} with seed {cfg.seed} seems to already exist at {cfg.models_dir}')
        # print('Pretraining abort...')
        # exit()
        print('\nDo you want to overwrite it?')
        answer = input('Answer: [y]/n \n')
        while answer not in ['', 'y', 'Y', 'yes', 'Yes','n', 'Y','no','No'] :  
            answer = input('Answer: [y]/n \n')
        if answer in ['n','no','No']: exit()
    os.makedirs(folder, exist_ok=True)
    # cfg.models_dir = work_dir / cfg.models_dir 
    workspace = Workspace(cfg, work_dir)
    
    print(f'Workspace: {work_dir}\nSEED {cfg.seed}')
    workspace.run()
    workspace.save_results()
    print(f'Experiment with SEED {cfg.seed} finished')

if __name__ == '__main__':
    main()