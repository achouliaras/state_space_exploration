import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import numpy as np
import random
import minigrid
import lib.human_player as human_player
from hydra import initialize, compose
from omegaconf import OmegaConf
import time

CONFIG = {
    'domain': 'MiniGrid'
}

# Convert the dictionary into an OmegaConf object
cfg = OmegaConf.create(CONFIG)

# Autoencoder Model
class MinigridAutoencoder(nn.Module):
    def __init__(self, obs_shape, latent_dim):
        super().__init__()
        w, h, c = obs_shape
        self.encoder = nn.Sequential(
            nn.Conv2d(c, 16, kernel_size=2, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=2, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=2, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear((h - 2) * (w - 2) * 64, latent_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, (h - 2) * (w - 2) * 64),
            nn.ReLU(),
            nn.Unflatten(1, (64, w - 2, h - 2)),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=1, padding=0, output_padding=0),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=1, padding=0, output_padding=0),
            nn.ReLU(),
            nn.ConvTranspose2d(16, c, kernel_size=2, stride=1, padding=0),
            nn.Sigmoid()  # Output normalized to [0, 1]
        )

    def forward(self, s_t):
        x = s_t.permute(0, 3, 1, 2)
        z = self.encoder(x)
        s_t1 = self.decoder(z)
        s_t1 = s_t1.permute(0,2,3,1)
        return s_t1 

# Environment and Training
env = gym.make("MiniGrid-BlockedUnlockPickup-v0",render_mode='rgb_array')  # Example environment
env = minigrid.wrappers.FullyObsWrapper(env)
env = minigrid.wrappers.ImgObsWrapper(env)

obs_shape = env.observation_space.shape # (7, 7, 3)
print(obs_shape)
latent_dim = 128
model = MinigridAutoencoder(obs_shape, latent_dim)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

replay_buffer = deque(maxlen=20000)
env.observation_space.shape

# Collect data with random actions
for episode in range(50):  # Adjust number of episodes as needed
    obs, _ = env.reset()
    for _ in range(400):  # Adjust steps per episode
        action = env.action_space.sample()
        next_obs, _, terminate, truncate, _ = env.step(action)
        if terminate or truncate:
            break
        
        replay_buffer.append((obs, action, next_obs))
        obs = next_obs

env.close()

z_train=[]
train_labels=[]

batch_size = 1024
# Training Loop
for epoch in range(100):  # Adjust number of epochs
    batch = random.sample(replay_buffer, batch_size)
    s_t, a_t, s_t1 = zip(*batch)
    
    s_t = torch.tensor(np.stack(s_t), dtype=torch.float32) / 255.0
    s_t1 = torch.tensor(np.stack(s_t1), dtype=torch.float32) / 255.0

    prediction_s_t1, embedding = model(s_t)
    z_train.append(embedding.detach().cpu())
    for i in range(batch_size): train_labels.append('Train')
    loss = loss_fn(prediction_s_t1, s_t)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {100 * loss.item():.6f} %")
print(f"Epoch {epoch}, Loss: {loss.item()}")

print('START PLAYING')
input('')

eval_env = gym.make("MiniGrid-BlockedUnlockPickup-v0", render_mode='human')  # Example environment
eval_env = minigrid.wrappers.FullyObsWrapper(eval_env)
eval_env = minigrid.wrappers.ImgObsWrapper(eval_env)
agent = human_player.Agent(name=f'Human', action_space=eval_env.action_space, cfg=cfg)

z_human = []
test_labels = []

# Collect data with random actions
for episode in range(2):  # Adjust number of episodes as needed
    obs, _ = eval_env.reset()
    for _ in range(800):  # Adjust steps per episode
        action = agent.get_action()
        # action = eval_env.action_space.sample()
        next_obs, _, terminate, truncate, _ = eval_env.step(action)
        frame = eval_env.render()
        time.sleep(0.1)
        if terminate or truncate:
            break
                
        s_t = torch.tensor(np.stack(obs), dtype=torch.float32).unsqueeze(0) / 255.0
        s_t1 = torch.tensor(np.stack(next_obs), dtype=torch.float32) / 255.0
        
        with torch.no_grad():
            reconstruciton, embedding = model(s_t)
            loss = loss_fn(reconstruciton, s_t)
        
        z_human.append(embedding.detach().cpu())
        test_labels.append('Test')
        print(f'Loss after action {action}: {100 * loss.detach().cpu().numpy():.6f} %')

        obs = next_obs

embeddings = z_train + z_human
embeddings = torch.cat(embeddings, dim=0).numpy()
print(embeddings.shape)
labels = train_labels + test_labels
print(len(labels))

from sklearn.manifold import TSNE
from sklearn.preprocessing import QuantileTransformer, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

label_encoder = LabelEncoder()
# Convert string labels to integers
y = label_encoder.fit_transform(labels)
label_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
print("Label mapping:", label_mapping)

# Preprocess again
pipe = make_pipeline(SimpleImputer(strategy="mean"), QuantileTransformer())
embeddings = pipe.fit_transform(embeddings.copy())

# Perform t-SNE on the embeddings
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
embeddings_2d = tsne.fit_transform(embeddings)
    
import matplotlib.pyplot as plt

# Scatter plot
plt.figure(figsize=(10, 8))
scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=y, cmap='viridis', s=5)
plt.colorbar(scatter, label="Labels")  # Add a color bar if labels are available
plt.title("t-SNE Visualization of Latent Space")
plt.savefig()
plt.show()