# seed=$RANDOM
seed=1
test_name=PEBBLE
domain=MiniGrid # highway-env # ALE
env=BlockedUnlockPickup-v0 # highway-v0 # Breakout-v5
num_seed_steps=2001
num_unsup_steps=3000
num_train_steps=40000
rb_pretrain_capacity=$num_seed_steps+$num_unsup_steps
rb_train_capacity=40000
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
       replay_buffer_capacity=$rb_pretrain_capacity debug=True test=PEBBLE 

# Deploy copies of pretrained agent to environment to generate trajectories using using GNU parallel
parallel -j "$num_cores" \
       "python themis_generate_trajectories.py device=cpu \
              domain=$domain env=$env render_mode=rgb_array \
              seed=$seed episodes_to_gen=$episodes_per_core \
              num_seed_steps=$num_seed_steps num_unsup_steps=$num_unsup_steps num_train_steps=$num_train_steps \
              debug=True test=$test_name cpu_id={#} 2>&1 | tee gen_traj_output_{#}.log >/dev/null" ::: $(seq "$num_cores") &

trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
wait

# Training script
python themis_train.py device=cpu \
       domain=$domain env=$env render_mode=rgb_array max_episode_steps=51 seed=$seed \
       num_seed_steps=$num_seed_steps num_unsup_steps=$num_unsup_steps num_train_steps=$num_train_steps \
       replay_buffer_capacity=$rb_train_capacity debug=True test=PEBBLE
