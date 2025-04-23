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
from omegaconf import DictConfig
from termcolor import colored
from collections import deque
from tqdm import tqdm
import hydra
from lib.logger import Logger
from lib import env_setup
from lib import agent_setup
from lib import utils
from lib.eval import evaluate_agent

# Interact with environment for data generation or Load data if generate trajectories works
# Representation Learning
# Imitation Learning
# Skill Extraction
# Save resulting agent

class Workspace(object):
    def __init__(self, cfg, work_dir):
        print(colored('EXECUTING OFFLINE PRETRAINING', 'green'))
        self.work_dir = work_dir
        
        self.logger = Logger(
            cfg.models_dir,
            save_tb=cfg.log_save_tb,
            log_frequency=cfg.log_frequency,
            agent=cfg.agent.name)

        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)
        cfg.save_video=False
        self.env, cfg, self.obs_space = env_setup.make_env(cfg, cfg.render_mode)
        self.env.action_space.seed(cfg.seed)
        self.cfg = cfg
        
        self.agent = agent_setup.create_agent(cfg)
        
        self.num_seed_steps = int(self.cfg.num_seed_steps)

        # for logging
        self.total_feedback = 0
        self.labeled_feedback = 0
        self.step = 0
        self.episode=0
        self.interactions=0
        
        self.obs = np.zeros((self.num_seed_steps+1, 1) + self.obs_space.shape)
        self.actions = np.zeros((self.num_seed_steps, 1) + self.cfg.action_space)
        self.logprobs = np.zeros((self.num_seed_steps, 1))
        self.rewards = np.zeros((self.num_seed_steps, 1))
        self.dones = np.zeros((self.num_seed_steps+1, 1))
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

    def generate_training_data(self):
        self.episode, terminated, truncated = 0, False, False
        self.total_time=0
        start_time = time.time()
        obs, _ = self.env.reset(seed = self.cfg.seed)
        # obs, _ = self.env.reset()
        # obs, _, _, _, _ = self.env.step(1) # FIRE action for breakout

        for global_step in tqdm(range(self.num_seed_steps), desc="GENERATING DATA: "):
            # sample action for data collection
            action = self.env.action_space.sample()
            
            next_obs, reward, terminated, truncated, info = self.env.step(action)

            done = terminated or truncated
            if terminated or truncated:
                episode_time = time.time() - start_time
                self.total_time += episode_time
                
                self.episode += 1
                self.step = global_step
                next_obs, _ = self.env.reset()

                # next_obs, _, _, _, _ = self.env.step(1) # FIRE action for breakout
            
            self.obs[global_step]=obs
            self.actions[global_step]=action
            self.rewards[global_step]=reward
            self.dones[global_step]=done
            obs = next_obs

        last_obs = next_obs
        self.obs[-1] = last_obs
        last_done = done
        self.dones[-1]=last_done
        print(f'DATA GENERATED: Steps:{global_step+1}, Episodes:{self.episode}, Time:{self.total_time} sec')

    def run(self):
        print('OFFLINE PRETRAINING STARTS')
        start_time = time.time()
        # Training Loop
        for epoch in tqdm(range(self.cfg.offline_epochs), desc="Training Offline Model: "):
            # Pre-Training Update 
            loss = self.agent.offline_update([self.obs, self.actions, self.rewards, self.dones], self.logger, self.num_seed_steps)
            
            if epoch % 5 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.6f}")
        print(f"Epoch {epoch}, Loss: {loss}")

        episode_time = time.time() - start_time
        self.total_time += episode_time
        self.logger.log('train/epoch', epoch, epoch)
        self.logger.log('train/duration', episode_time, epoch)
        self.logger.log('train/total_duration', self.total_time, epoch)
        self.logger.dump(epoch, ty='train')
        self.env.close()
        print('OFFLINE PRETRAINING FINISHED')
        self.logger.close()

    def save_results(self):
        print('SAVING STARTS')
        keys_to_save = ['step', 'episode']
        payload = {k: self.__dict__[k] for k in keys_to_save}
        
        agent_setup.save_agent(self.agent, None, payload, self.work_dir, 
                               self.cfg, 
                               self.cfg.offline_epochs,
                               mode=self.cfg.export_protocol)
        print('SAVING COMPLETED')
        
@hydra.main(version_base=None, config_path="../config", config_name='themis_offline_pretrain')
def main(cfg : DictConfig):
    print(f'Device used CUDA:{torch.cuda.is_available()}, MPS:{torch.backends.mps.is_available()}')
    work_dir = Path.cwd()
    # cfg.output_dir = work_dir / cfg.output_dir  
    
    folder = work_dir / cfg.models_dir
    if folder.exists():
        print(f'Experiment for {cfg.agent.name}_{cfg.test} with seed {cfg.seed} seems to already exist at {cfg.models_dir}')
        print('\nOverwriting...')
        # print('Pretraining abort...')
        # exit()
        # print('\nDo you want to overwrite it?')
        # answer = input('Answer: [y]/n \n')
        # while answer not in ['', 'y', 'Y', 'yes', 'Yes','n', 'Y','no','No'] :  
        #     answer = input('Answer: [y]/n \n')
        # if answer in ['n','no','No']: exit()
    os.makedirs(folder, exist_ok=True)
    # cfg.models_dir = work_dir / cfg.models_dir 
    workspace = Workspace(cfg, work_dir)
    
    print(f'Workspace: {work_dir}\nSEED {cfg.seed}')
    workspace.generate_training_data()
    workspace.run()
    workspace.save_results()
    print(f'Experiment with SEED {cfg.seed} finished')

if __name__ == '__main__':
    main()