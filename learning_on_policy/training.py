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
from collections import deque
import copy
import hydra
from omegaconf import DictConfig
from termcolor import colored

from lib.logger import Logger
from lib import env_setup
from lib import agent_setup
from lib import utils
from lib.eval import evaluate_agent


class Workspace(object):
    def __init__(self, cfg, work_dir):
        print(colored('EXECUTING TRAINING', 'green'))
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
        
        self.num_update_steps = self.cfg.agent.action_cfg.num_update_steps
        self.batch_size =  int(self.num_update_steps) # x num_of_envs
        self.cfg.agent.action_cfg.batch_size = self.batch_size
        self.num_iterations = int((self.cfg.num_train_steps+1) // self.batch_size)
        
        self.agent = agent_setup.create_agent(cfg)
        
        # Load AGENT
        if cfg.import_model:
            self.agent, _ = agent_setup.load_agent(self.work_dir, self.cfg, self.agent, mode=cfg.import_protocol)
            # self.agent.freeze_models(mode=cfg.import_protocol) # Has error makes model output Nans
        # self.agent.reset_critic()

        # If you add parallel envs adjust size
        self.obs = np.zeros((self.num_update_steps, 1) + self.obs_space.shape)
        self.actions = np.zeros((self.num_update_steps, 1) + self.cfg.action_space)
        self.logprobs = np.zeros((self.num_update_steps, 1))
        self.rewards = np.zeros((self.num_update_steps, 1))
        self.dones = np.zeros((self.num_update_steps, 1))
        self.values = np.zeros((self.num_update_steps, 1))
        self.memories = None
        if self.agent.has_memory:
            self.memories = np.zeros((self.num_update_steps, 1) + self.agent.memory_size)

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
        true_episode_reward = 0
        episode_length = 0
        # store train returns of recent 10 episodes
        # avg_train_true_return = deque([], maxlen=10) 
        total_time=0
        start_time = time.time()

        obs, _ = self.env.reset(seed = self.cfg.seed)
        # obs, _ = self.env.reset()
        # obs, _, _, _, _ = self.env.step(1) # FIRE action for breakout
        done = 0 
        if self.agent.has_memory:
            memory = np.zeros(self.agent.memory_size)

        print('TRAINING STARTS')
        for iteration in range(1, self.num_iterations+1):
            # Annealing the rate if instructed to do so.
            if self.agent.anneal_lr:
                frac = 1.0 - (iteration - 1.0) / self.num_iterations
                lrnow = frac * self.agent.lr
                self.agent.optimizer.param_groups[0]["lr"] = lrnow
                
            for step in range(self.num_update_steps):
                global_step += 1 # or num of envs
                episode_length +=1
                self.obs[step] = obs
                self.dones[step] = done
                if self.agent.has_memory:
                    self.memories[step] = memory

                # Action logic
                with torch.no_grad():
                    if self.agent.has_memory:
                        obs_tensor = torch.DoubleTensor(obs).to(self.device).unsqueeze(0)
                        memory_tensor = torch.DoubleTensor(memory).to(self.device).unsqueeze(0)
                        mask_tensor = torch.DoubleTensor(1-done).to(self.device).unsqueeze(0)
                        # print(memory_tensor.shape)
                        # print(mask_tensor.shape)
                        action, logprob, _, value, memory = self.agent.get_action(obs=obs_tensor,
                                                                          action=None,
                                                                          memory=memory_tensor * mask_tensor)
                        memory = memory.detach().cpu().numpy()[0]
                    else:
                        action, logprob, _, value, _ = self.agent.get_action(torch.DoubleTensor(obs).to(self.device).unsqueeze(0))
                action = action.detach().cpu().numpy()[0]
                
                self.actions[step] = action
                self.logprobs[step] = logprob.detach().cpu().numpy()[0]
                self.values[step] = value.detach().cpu().numpy()[0]

                # execute step and log data
                next_obs, reward, terminated, truncated, info = self.env.step(action)
                next_done = terminated or truncated
                next_memory = None
                if self.agent.has_memory:
                    next_memory = memory
                self.rewards[step] = reward
                
                obs = next_obs
                done = next_done
                memory = next_memory
                true_episode_reward += reward
                
                if terminated or truncated:
                    episode_time = time.time() - start_time
                    total_time += episode_time
                    self.logger.log('train/episode', self.episode, global_step)
                    self.logger.log('train/episode_reward', episode_reward, global_step)
                    self.logger.log('train/true_episode_reward', true_episode_reward, global_step)
                    self.logger.log('train/episode_length', episode_length, global_step)
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
                    true_episode_reward = 0
                    episode_length = 0
                    self.step = global_step
                    self.episode += 1
                    obs, _ = self.env.reset()
                    # obs, _, _, _, _ = self.env.step(1) # FIRE action for breakout
                    done = 0 
                    if self.agent.has_memory:
                        memory = np.zeros(self.agent.memory_size)

            # Training Update 
            # if global_step % self.num_update_steps == 0:
            # print('Actions: ',[i[0][0] for i in self.actions])
            # print([i[0] for i in self.rewards])
            self.agent.update([self.obs, self.actions, self.logprobs, self.values, self.rewards, self.dones, self.memories], 
                              [next_obs, next_done, next_memory], self.logger, global_step)
            
            if self.cfg.log_success:
                episode_success = max(episode_success, terminated)

        episode_time = time.time() - start_time
        total_time += episode_time
        self.logger.log('train/episode', self.episode, global_step)
        self.logger.log('train/episode_reward', episode_reward, global_step)
        self.logger.log('train/true_episode_reward', true_episode_reward, global_step)
        self.logger.log('train/episode_length', episode_length, global_step)
        self.logger.log('train/duration', episode_time, global_step)
        self.logger.log('train/total_duration', total_time, global_step)
        if self.cfg.log_success:
            self.logger.log('train/episode_success', episode_success, global_step)
            self.logger.log('train/true_episode_success', episode_success, global_step)

        # self.logger.dump(global_step, ty='train')
        self.env.close()
        print('TRAINING FINISHED')
        self.logger = evaluate_agent(self.agent, self.cfg, self.logger)
        self.logger.close()

    def save_results(self):
        print('SAVING STARTS')
        keys_to_save = ['step', 'episode']
        payload = {k: self.__dict__[k] for k in keys_to_save}

        agent_setup.save_agent(self.agent, None, payload, self.work_dir, self.cfg, self.global_step)
        print('SAVING COMPLETED')
        
@hydra.main(version_base=None, config_path="../config", config_name='themis_train_on_policy')
def main(cfg : DictConfig):
    print(f'Device used CUDA:{torch.cuda.is_available()}, MPS:{torch.backends.mps.is_available()}')
    work_dir = Path.cwd()
    # cfg.output_dir = work_dir / cfg.output_dir  
    
    folder = work_dir / cfg.models_dir
    if folder.exists():
        print(f'Experiment for {cfg.agent.name}_{cfg.test} with seed {cfg.seed} seems to already exist at {cfg.models_dir}')
        print('\nOverwriting...')
        # print('\nDo you want to overwrite it?')
        # answer = input('Answer: [y]/n \n')
        # while answer not in ['', 'y', 'Y', 'yes', 'Yes','n', 'Y','no','No'] :  
        #     answer = input('Answer: [y]/n \n')
        # if answer in ['n','no','No']: exit()
    os.makedirs(folder, exist_ok=True)
    # cfg.models_dir = work_dir / cfg.models_dir 
    workspace = Workspace(cfg, work_dir)
    
    print(f'Workspace: {work_dir}\nSEED {cfg.seed}')
    workspace.run()
    workspace.save_results()
    print(f'Experiment with SEED {cfg.seed} finished')

if __name__ == '__main__':
    main()