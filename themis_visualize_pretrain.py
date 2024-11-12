#!/usr/bin/env python3
import numpy as np
import pandas as pd
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
import random

from lib.logger import Logger
from collections import deque

import lib.env_setup as env_setup
import lib.agent_setup as agent_setup
import lib.utils as utils
from lib.trajectory_io import TrajectoryProcessor
import hydra
from omegaconf import DictConfig

class Workspace(object):
    def __init__(self, cfg, work_dir):
        self.work_dir = work_dir
        self.cfg = cfg

        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)

        self.traj_proc = TrajectoryProcessor(work_dir, cfg)
        print('INIT COMPLETE')
        
    def run(self):
        percentage = self.cfg.sample_size
        rate = self.cfg.sampling_rate

        # Check if the percentage is valid
        if not (0 < percentage <= 100):
            raise ValueError("Percentage must be between 1\% - 100\%.")

        combined_df = pd.DataFrame()
        for traj_dir, label in zip(self.traj_proc.traj_dir_list,self.traj_proc.labels):
            filenames = self.traj_proc.get_trajectory_filenames(traj_dir)
            # Calculate the number of trajectories to sample
            rough_sample_size = len(filenames) * (percentage / 100)
            num_traj = max(1, int(round(rough_sample_size)))  # Ensure at least one trajectory is sampled
            print(f"Sample rate for {label} is {percentage}% of {len(filenames)} = {rough_sample_size}")
            print(f"Sampling size for {label} was chosen as {num_traj}")
            # Sample num_elements unique elements from the list
            sampled_traj_names = random.sample(filenames, num_traj)
            sampled_traj_names.sort()
            
            for name in sampled_traj_names:
                self.traj_proc.load(traj_dir, name)
                self.traj_proc.resample(rate)
                filename = os.path.splitext(name)[0] # remove .pt suffix from the name
                if self.cfg.export_clips :self.traj_proc.export_to_video(label, filename=filename,fps=self.cfg.fps)

                parts = filename.split("_")
                episode = f"{parts[-2]}_{parts[-1]}"
                df_episode = self.traj_proc.create_dataframe(episode=episode, label=label)
                combined_df = pd.concat([combined_df, df_episode], ignore_index=True) # Concat all episodes from the same label
            print(f'Trajectory dataset size is: {combined_df.shape}')
        
        combined_df.to_csv(f'{self.traj_proc.vis_dir}/data.csv', index=False)
        print(f'Dataset created at: {self.traj_proc.vis_dir}')
        
@hydra.main(version_base=None, config_path="config", config_name='themis_visualise_trajectories')
def main(cfg : DictConfig):
    work_dir = Path.cwd()
    # cfg.output_dir = work_dir / cfg.output_dir  

    folder = work_dir / cfg.output_dir
    if not folder.exists():
        print(f"Experiment collection doesn't exist at {cfg.output_dir}")
        raise FileNotFoundError
    print(f'Workspace: {work_dir}\nSEED {cfg.seed}')
    workspace = Workspace(cfg, work_dir)    
    workspace.run()
    print(f'Experiment with SEED {cfg.seed} finished')

if __name__ == '__main__':
    main()