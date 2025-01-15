seed=$RANDOM
sample=1
export PYTORCH_ENABLE_MPS_FALLBACK=1

python -m learning_on_policy.pretraining device=cpu \
       domain=MuJoCo env=InvertedPendulum-v5 render_mode=rgb_array seed=$seed \
       num_seed_steps=5e3 num_unsup_steps=100 num_train_steps=1e6 \
       replay_buffer_capacity=1000000 debug=True test=PEBBLE

python -m learning_on_policy.pretraining device=cpu \
       domain=Control env=Pendulum-v1 render_mode=rgb_array seed=$seed \
       num_seed_steps=5e3 num_unsup_steps=100 num_train_steps=1e6 \
       replay_buffer_capacity=1000000 debug=True test=PEBBLE



