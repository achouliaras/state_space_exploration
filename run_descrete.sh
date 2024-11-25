# seed=$RANDOM
seed=1
test_name=PEBBLE
domain=MiniGrid # highway-env # ALE
env=BlockedUnlockPickup-v0 # highway-v0 # Breakout-v5
num_seed_steps=1001
num_unsup_steps=2000
num_train_steps=3000
episodes_2_generate=16 
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Get the number of available CPU cores
num_cores=$(sysctl -n hw.physicalcpu)
num_cores=4

# Calculate episodes per process to generate
episodes_per_core=$(($episodes_2_generate / $num_cores))

# Pretraining script
python themis_pretrain.py device=cpu \
       domain=$domain env=$env render_mode=rgb_array max_episode_steps=51 seed=$seed \
       num_seed_steps=$num_seed_steps num_unsup_steps=$num_unsup_steps num_train_steps=$num_train_steps \
       replay_buffer_capacity=5000 debug=True test=PEBBLE 

# Deploy copies of pretrained agent to environment to generate trajectories using using GNU parallel
parallel -j "$num_cores" \
       "python themis_generate_trajectories.py device=cpu \
              domain=$domain env=$env render_mode=rgb_array \
              seed=$seed episodes_to_gen=$episodes_per_core \
              num_seed_steps=$num_seed_steps num_unsup_steps=$num_unsup_steps num_train_steps=$num_train_steps \
              debug=True test=$test_name cpu_id={#} 2>&1 | tee gen_traj_output_{#}.log >/dev/null" ::: $(seq "$num_cores") &

trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
wait

# TO RUN LAST AFTER ALL PRETRAIN ALGOS AND HUMAN DEMOS
# Generate dataset from trajectories, apply Dimensionality Reduction and export plots
python themis_visualize_pretrain.py domain=$domain env=$env render_mode=rgb_array \
       device=cpu seed=$seed sample_size=30 sampling_rate=2 fps=30 \
       debug=True export_clips=False
