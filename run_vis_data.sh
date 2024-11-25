# seed=$RANDOM
seed=1
domain=MiniGrid # highway-env # ALE
env=BlockedUnlockPickup-v0 # highway-v0 # Breakout-v5

export PYTORCH_ENABLE_MPS_FALLBACK=1

# TO RUN LAST AFTER ALL PRETRAIN ALGOS AND HUMAN DEMOS
# Generate dataset from trajectories, apply Dimensionality Reduction and export plots
python themis_visualize_pretrain.py domain=$domain env=$env render_mode=rgb_array \
       device=cpu seed=$seed sample_size=30 sampling_rate=2 fps=30 \
       debug=True export_clips=False
