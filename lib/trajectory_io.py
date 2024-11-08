import os
import re
import shutil
import cv2
import torch
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.preprocessing import QuantileTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

import umap  # pip install umap-learn

def clear_folder_contents(folder_path):
    # Loop through each entry in the folder and remove it
    for entry in os.listdir(folder_path):
        entry_path = os.path.join(folder_path, entry)
        if os.path.isdir(entry_path):
            shutil.rmtree(entry_path)  # Delete subdirectory and its contents
        else:
            os.remove(entry_path)  # Delete individual file

class TrajectoryProcessor(object):
    def __init__(self, work_dir, cfg) -> None:
        self.file_index = 0
        self.cpu_id = cfg.cpu_id
        self.experiments_dir = work_dir / cfg.output_dir
        

        traj_dir_list = [
                name for name in os.listdir(self.experiments_dir) 
                if os.path.isdir(os.path.join(self.experiments_dir, name)) and name != cfg.vis_dir_name
            ]
        labels = traj_dir_list.copy()
        print('Experiment names considered:', labels)
        traj_dir_list = [f'{cfg.output_dir}/{path}/seed-{cfg.seed}/trajectories' for path in traj_dir_list]
        print('Experiment locations:\n\t', "\n\t".join(traj_dir_list))
        
        vis_dir = self.experiments_dir / cfg.vis_dir_name
        print('Visualisations output location:',vis_dir)
        vis_clips_dir = vis_dir / 'Clips'
        print('Sampled trajectory clips location:', vis_clips_dir)
        
        # Used only when generating trajectories
        if 'models_dir' in cfg:
            self.traj_dir = cfg.models_dir + "/trajectories"
        else:
            self.traj_dir = None

        if self.traj_dir is not None:
            if os.path.exists(self.traj_dir) and cfg.overwrite_trajectories == True:
                try:
                    clear_folder_contents(self.traj_dir)
                except:
                    print("logger.py warning: Unable to remove tb directory")
                    pass    
            elif os.path.exists(self.traj_dir) and cfg.overwrite_trajectories == False:
                largest_idx = 0
                pattern = re.compile(rf"Episode_{self.cpu_id}_(\d+)\.pt")
                for filename in os.listdir(self.traj_dir):
                    match = pattern.match(filename)
                    if match:
                        number = int(match.group(1))
                        largest_idx = max(largest_idx, number)
                self.file_index = largest_idx+1
            else:
                os.makedirs(self.traj_dir, exist_ok=True)
        
        # Used when visualising trajectories
        if os.path.exists(vis_dir):
            try:
                clear_folder_contents(vis_dir)
                os.makedirs(vis_clips_dir, exist_ok=True)
            except:
                print("logger.py warning: Unable to remove tb directory")
                pass   
        else:
            os.makedirs(vis_dir, exist_ok=True)
            os.makedirs(vis_clips_dir, exist_ok=True) 
        
        self.labels = labels
        self.traj_dir_list = traj_dir_list
        self.vis_dir = vis_dir
        self.vis_clips_dir = vis_clips_dir
        self.frames = []
        self.observations = []
        self.actions = []
        self.rewards = []
        self.dataset = None
        self.keys_for_payload = ['frames','observations', 'actions', 'rewards']

    def add(self, frame, obs, action, reward):
        self.frames.append(frame)
        self.observations.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)

    def save(self, traj_dir, episode):
        payload = {k: self.__dict__[k] for k in self.keys_for_payload}
        torch.save(payload, f'{traj_dir}/Episode_{self.cpu_id}_{self.file_index+episode}.pt')
        # Reset saved elements for the next trajectory
        self.frames = []
        self.observations = []
        self.actions = []
        self.rewards =[]
    
    def load(self, traj_dir, filename):
        payload = torch.load(f'{traj_dir}/{filename}')
        self.frames, self.observations, self.actions, self.rewards = [payload[k] for k in self.keys_for_payload]

    def get_trajectory_filenames(self, traj_dir):
        files = []
        for filename in os.listdir(traj_dir):
            if filename.endswith('.pt'):
                files.append(filename)
        return files

    def resample(self, rate):
        if rate <= 0:
            raise ValueError("Sampling rate must be a positive integer.")
        if rate == 1: return

        # ToDo: Treat skipped states

        self.frames = self.frames[::rate]
        self.observations = self.observations[::rate]
        self.actions = self.actions[::rate]
        self.rewards = self.rewards[::rate]

    def export_to_video(self, label, filename, fps=30):
        if not self.frames:
            raise ValueError("The list of RGB frames is empty.")

        full_filename = str(self.vis_clips_dir / f'{label}_{filename}.mp4')
        # Get the dimensions of the first frame
        height, width, _ = self.frames[0].shape
        
        # Create a VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can also use 'XVID' or 'MJPG'
        video_writer = cv2.VideoWriter(full_filename, fourcc, fps, (width, height))

        # Write each RGB frame to the video
        for rgb_frame in self.frames:
            # Convert RGB to BGR (OpenCV uses BGR format)
            bgr_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
            video_writer.write(bgr_frame)

        # Release the VideoWriter
        video_writer.release()
        video_writer = None
        print(f"\tVideo saved for trajectory {label}_{filename}")

    def create_dataframe(self, episode, label, dataset = None):
        if dataset is None:
            self.dataset = self.observations
        else:
            self.dataset = dataset
        
        flattened_dataset = [arr.flatten() for arr in self.dataset]

        df = pd.DataFrame(flattened_dataset)
        df["Episode"]= episode
        df['Step'] = range(len(flattened_dataset))
        df["y"]= label

        # print(df)
        return df
        
    # WIP
    def state_projection(self, method = "UMAP"):
        if method == "t-SNE":
            tsne = TSNE(n_components=2, random_state=42)
            X_tsne = tsne.fit_transform(X)

            X_reduced_2 = X_tsne
        elif method == "UMAP":
            # Preprocess again
            pipe = make_pipeline(SimpleImputer(strategy="mean"), QuantileTransformer())
            X = pipe.fit_transform(X.copy())

            # Fit UMAP to processed data
            manifold = umap.UMAP().fit(X, y)
            X_reduced_2 = manifold.transform(X)

            # Plot the results
            plt.scatter(X_reduced_2[:, 0], X_reduced_2[:, 1], c=y, s=0.5)
        elif method=='PCA':
            raise NotImplementedError
        else:
            raise NotImplementedError
        pass
        
        return X_reduced_2
    