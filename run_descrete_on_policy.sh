# seed=$RANDOM
# seed=1

test_name=Pret_Offline_NO_Freeze #Vanilla #Pret_Offline_Freeze #NoMemory
domain=MiniGrid # highway-env # ALE
env=Empty-8x8-v0 #Empty-5x5-v0 #BlockedUnlockPickup-v0 # highway-v0 # Breakout-v5
architecture=CNN-LSTM
offline_num_seed_steps=5e4
offline_epochs=100
import_model=True
import_protocol=OFFLINE

num_train_steps=100100
episodes_2_generate=16 
export PYTORCH_ENABLE_MPS_FALLBACK=1

device=cuda
# Get the number of available CPU cores
# num_cores=$(sysctl -n hw.physicalcpu)
num_cores=4

# Calculate episodes per process to generate
episodes_per_core=$(($episodes_2_generate / $num_cores))

# 1 2 3 4 5 6 7 8
for seed in 9 10; do
       # # Offline Training script
       # python -m learning_offline.pretraining device=$device \
       #        domain=$domain env=$env render_mode=rgb_array max_episode_steps=100 seed=$seed \
       #        architecture=$architecture offline_epochs=$offline_epochs \
       #        num_seed_steps=$offline_num_seed_steps debug=True test=$import_protocol

       # # Pretraining script
       # python -m learning_on_policy.pretraining device=$device \
       #        domain=$domain env=$env render_mode=rgb_array max_episode_steps=51 seed=$seed \
       #        num_seed_steps=$num_seed_steps num_unsup_steps=$num_unsup_steps num_train_steps=$num_train_steps \
       #        replay_buffer_capacity=$rb_pretrain_capacity debug=True test=PEBBLE 

       # # Deploy copies of pretrained agent to environment to generate trajectories using using GNU parallel
       # parallel -j "$num_cores" \
       #        "python -m learning_utils.generate_trajectories device=$device \
       #               domain=$domain env=$env render_mode=rgb_array \
       #               seed=$seed episodes_to_gen=$episodes_per_core \
       #               num_seed_steps=$num_seed_steps num_unsup_steps=$num_unsup_steps num_train_steps=$num_train_steps \
       #               debug=True test=$test_name cpu_id={#} 2>&1 | tee gen_traj_output_{#}.log >/dev/null" ::: $(seq "$num_cores") &

       # trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
       # wait

       # Training script
       python -m learning_on_policy.training device=$device \
              domain=$domain env=$env render_mode=rgb_array max_episode_steps=100 seed=$seed architecture=$architecture\
              offline_epochs=$offline_epochs import_model=$import_model import_protocol=$import_protocol\
              num_train_steps=$num_train_steps debug=True test=$test_name
done