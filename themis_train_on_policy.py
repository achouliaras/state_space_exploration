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

class Workspace(object):
    def __init__(self, cfg, work_dir):
        print(colored('EXECUTING TRAINING', 'green'))
        self.work_dir = work_dir
        
        self.logger = Logger(
            cfg.models_dir,
            save_tb=cfg.log_save_tb,
            seed=cfg.seed,
            log_frequency=cfg.log_frequency,
            agent=cfg.algorithm.name)

        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)
        self.env, cfg, self.obs_space = env_setup.make_env(cfg, cfg.render_mode)
        self.env.action_space.seed(cfg.seed)
        self.cfg = cfg
        
        self.num_update_steps = self.cfg.agent.action_cfg.num_update_steps
        self.batch_size =  int(self.num_update_steps) # x num_of_envs
        self.cfg.agent.action_cfg.batch_size = self.batch_size
        self.num_iterations = (self.cfg.num_train_steps+1) // self.batch_size

        actor_cfg, critic_cfg = agent_setup.config_agent(cfg)

        self.agent, self.replay_buffer = agent_setup.create_agent(cfg, actor_cfg, critic_cfg, cfg.agent.action_cfg, self.obs_space)
        
        # Load AGENT
        self.agent, _ = agent_setup.load_agent(self.work_dir, self.cfg, self.agent)
        self.agent.reset_critic()

        # If you add parallel envs adjust size
        self.obs = np.zeros(self.num_update_steps + self.obs_space.shape)
        self.actions = np.zeros(self.num_update_steps + self.actor_cfg.action_dim)
        self.logprobs = np.zeros(self.num_update_steps)
        self.rewards = np.zeros(self.num_update_steps)
        self.dones = np.zeros(self.num_update_steps)
        self.values = np.zeros(self.num_update_steps)

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
        
        global_step = 0

        # store train returns of recent 10 episodes
        # avg_train_true_return = deque([], maxlen=10) 
        total_time=0
        start_time = time.time()

        next_obs, _ = self.env.reset(seed = self.cfg.seed)
        # next_obs, _ = self.env.reset()
        # next_obs, _, _, _, _ = self.env.step(1) # FIRE action for breakout
        next_done = 0 

        print('TRAINING STARTS')
        for iteration in range(1, self.num_iterations+1):
            # Annealing the rate if instructed to do so.
            if self.cfg.agent.anneal_lr:
                frac = 1.0 - (iteration - 1.0) / self.num_iterations
                lrnow = frac * self.cfg.agent.action_cfg.actor_lr
                self.agent.actor_optimizer.param_groups[0]["lr"] = lrnow
                
            for step in range(self.num_update_steps):
                self.global_step += 1 # or num of envs
                self.obs[step] = next_obs
                self.dones[step] = next_done

                # Action logic
                with torch.no_grad():
                    action, logprob, _, value = self.agent.get_action(torch.FloatTensor(next_obs).to(self.device).unsqueeze(0))
                action = action.detach().cpu().numpy()[0]
                self.actions[step] = action
                self.logprobs[step] = logprob.detach().cpu().numpy()[0]
                self.values[step] = value.detach().cpu().numpy()[0]
                print(action)

                # execute step and log data
                next_obs, reward, terminated, truncated, info = self.env.step(action)
                next_done = terminated or truncated
                self.rewards[step] = reward
                
                if terminated or truncated:
                    episode_time = time.time() - start_time
                    total_time += episode_time
                    self.logger.log('train/episode', self.episode, global_step)
                    self.logger.log('train/episode_reward', episode_reward, global_step)
                    self.logger.log('train/true_episode_reward', info["episode"]["r"], global_step)
                    self.logger.log('train/episode_length', info["episode"]["l"], global_step)
                    self.logger.log('train/duration', episode_time, global_step)
                    self.logger.log('train/total_duration', total_time, global_step)
                    if self.cfg.log_success:
                        self.logger.log('train/episode_success', episode_success, global_step)
                        self.logger.log('train/true_episode_success', episode_success, global_step)

                    self.logger.dump(global_step, ty='train')
                    start_time = time.time()

                    # episode_reward = 0
                    if self.cfg.log_success:
                        episode_success = 0
                    self.step = step
                    self.episode += 1
                    next_obs, _ = self.env.reset()
                    # next_obs, _, _, _, _ = self.env.step(1) # FIRE action for breakout
                    next_done = 0 

            # Training Update 
            # if global_step % self.num_update_steps == 0:
            self.agent.update([self.obs, self.actions, self.logprobs, self.values, self.rewards, self.dones], 
                              [next_obs, next_done], self.logger, global_step)
            
            if self.cfg.log_success:
                episode_success = max(episode_success, terminated)

        episode_time = time.time() - start_time
        total_time += episode_time
        self.logger.log('train/episode', self.episode, global_step)
        self.logger.log('train/episode_reward', episode_reward, global_step)
        self.logger.log('train/true_episode_reward', info["episode"]["r"], global_step)
        self.logger.log('train/episode_length', info["episode"]["l"], global_step)
        self.logger.log('train/duration', episode_time, global_step)
        self.logger.log('train/total_duration', total_time, global_step)
        if self.cfg.log_success:
            self.logger.log('train/episode_success', episode_success, global_step)
            self.logger.log('train/true_episode_success', episode_success, global_step)
        self.logger.dump(global_step, ty='train')
        self.env.close()
        print('TRAINING FINISHED')
        self.logger = evaluate_agent(self.agent, self.cfg, self.logger)
        self.logger.close()

    def save_results(self):
        print('SAVING STARTS')
        keys_to_save = ['step', 'episode']
        payload = {k: self.__dict__[k] for k in keys_to_save}

        agent_setup.save_agent(self.agent, self.replay_buffer, payload, self.work_dir, self.cfg, self.global_step)
        print('SAVING COMPLETED')
        
@hydra.main(version_base=None, config_path="config", config_name='themis_train')
def main(cfg : DictConfig):
    work_dir = Path.cwd()
    # cfg.output_dir = work_dir / cfg.output_dir  
    
    folder = work_dir / cfg.models_dir
    # if folder.exists():
    #     print(f'Experiment for {cfg.algorithm.name}_{cfg.test} with seed {cfg.seed} seems to already exist at {cfg.models_dir}')
    #     # print('Pretraining abort...')
    #     # exit()
    #     print('\nDo you want to overwrite it?')
    #     answer = input('Answer: [y]/n \n')
    #     while answer not in ['', 'y', 'Y', 'yes', 'Yes','n', 'Y','no','No'] :  
    #         answer = input('Answer: [y]/n \n')
    #     if answer in ['n','no','No']: exit()
    os.makedirs(folder, exist_ok=True)
    # cfg.models_dir = work_dir / cfg.models_dir 
    workspace = Workspace(cfg, work_dir)
    
    print(f'Workspace: {work_dir}\nSEED {cfg.seed}')
    workspace.run()
    workspace.save_results()
    print(f'Experiment with SEED {cfg.seed} finished')

if __name__ == '__main__':
    main()