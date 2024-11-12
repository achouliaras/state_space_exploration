# seed=$RANDOM
seed=1
num_seed_steps=1000
num_unsup_steps=200
num_train_steps=1000
episodes_2_generate=5 

# Generate trajectories with human player
python themis_generate_human_demos.py device=cpu domain=ALE env=Breakout-v5 render_mode=human seed=$seed \
              episodes_to_gen=$episodes_2_generate overwrite_trajectories=False \
              num_seed_steps=$num_seed_steps num_unsup_steps=$num_unsup_steps num_train_steps=$num_train_steps \
              debug=True algorithm.name=HUMAN frameskip=1 frame_stack=1 test=Demos cpu_id=1