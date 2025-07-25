import gymnasium as gym
from PIL import Image
import os
from extra_minigrid_envs import register_all

register_all()

# Map keys to MiniGrid actions
key_to_action = {
    'a': 0,     # Turn left
    'd': 1,    # Turn right
    'w': 2,       # Move forward
    ' ': 3,        # Pickup
    's': 4,        # Drop
    'e': 5,        # Toggle
    'q': 6,        # Done
}

# Create environment
env = gym.make("MiniGrid-UnlockPickup-v0", render_mode="rgb_array")
obs, info = env.reset()

done = False
step = 0
while not done:
    # Render current frame and save to file
    img_array = env.render()
    img = Image.fromarray(img_array)
    img.save(f"frame.png")
    
    # Get user input
    key = input("Enter action: ").lower()
    if key not in key_to_action:
        print("Invalid key. Try again.")
        continue

    action = key_to_action[key]
    obs, reward, terminated, truncated, info = env.step(action)
    # Show agent status
    print(f"Step {step} | Reward: {reward}")
    step += 1

    if terminated or truncated:
        print("Episode finished. Resetting environment.")
        obs, info = env.reset()
        step = 0

env.close()
