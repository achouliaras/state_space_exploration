# seed=$RANDOM
seed=1
sample=1
export PYTORCH_ENABLE_MPS_FALLBACK=1

# python themis_pretrain.py device=cpu \
#        domain=ALE env=Pong-v5 render_mode=rgb_array seed=$seed \
#        num_seed_steps=1000 num_unsup_steps=200 num_train_steps=1000 \
#        replay_buffer_capacity=1000 debug=True test=PEBBLE


# Get the number of available CPU cores
num_cores=$(sysctl -n hw.physicalcpu)

num_cores=1

# Calculate episodes per process to generate
episodes_per_core=$((16 / num_cores))

# # Run the command in parallel using GNU parallel
# parallel -j "$num_cores" \
#        python themis_generate_trajectories.py device=cpu \
#               domain=ALE env=Pong-v5 render_mode=rgb_array \
#               seed=$seed episodes_to_gen=$episodes_per_core \
#               num_seed_steps=1000 num_unsup_steps=200 num_train_steps=1000 \
#               debug=True test=PEBBLE cpu_id={#} ::: $(seq "$num_cores")

python themis_visualize_pretrain.py device=cpu \
       domain=ALE env=Pong-v5 render_mode=rgb_array seed=$seed \
       sample_size=10 sampling_rate=1 fps=30 debug=True
