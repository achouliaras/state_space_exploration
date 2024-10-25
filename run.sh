seed=$RANDOM
sample=1
export PYTORCH_ENABLE_MPS_FALLBACK=1

python themis_pretrain.py device=cpu \
       domain=ALE env=Breakout-v5 render_mode=rgb_array seed=$seed \
       num_seed_steps=2000 num_unsup_steps=10000 num_train_steps=1000000 \
       replay_buffer_capacity=1000000 debug=True experiment=PEBBLE
