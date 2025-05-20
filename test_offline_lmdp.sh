domain=MiniGrid
env=DoorKey-5x5-v0 #Empty-8x8-v0 #UnlockPickup-v0 # highway-v0 # Breakout-v5
max_episode_steps=100
architecture=CNN-LSTM
sequence_length=4
device=cuda

# Offline Pre-Training
offline_num_seed_steps=50e4
offline_epochs=100
off_export_protocol=OFFLINE

# Online Pre-Training
num_seed_steps=0
num_unsup_steps=20e4
pre_offline_epochs=10
buffer=256
topK=5
pre_import_model=False
pre_import_protocol=NORMAL
pre_export_protocol=ONLINE
pre_freeze_protocol=NO # NO, CNN-PART, CNN, ALL

# Training
num_train_steps=200100
import_model=True
import_protocol=OFFLINE # NORMAL, OFFLINE, ONLINE
freeze_protocol=NO # NO, CNN-PART, CNN, ALL

test_name=LMDP_Offline #Vanilla, LMDP_Offline, AE_Offline, AEGIS

# 1 2 3 4 5 6 7 8 9 10
for seed in 1 2 3 4 5 6 7 8 9 10; do
       # Offline Training script
       python -m learning_offline.pretraining device=$device \
              domain=$domain env=$env render_mode=rgb_array max_episode_steps=100 seed=$seed architecture=$architecture \
              offline_epochs=$offline_epochs export_protocol=$off_export_protocol \
              num_seed_steps=$offline_num_seed_steps debug=True test=$test_name

       # # Pretraining script
       # python -m learning_on_policy.pretraining device=$device \
       #        domain=$domain env=$env render_mode=rgb_array max_episode_steps=$max_episode_steps seed=$seed architecture=$architecture \
       #        agent.discrete_action.sequence_length=$sequence_length \
       #        offline_epochs=$pre_offline_epochs import_model=$import_model freeze_protocol=$pre_freeze_protocol\
       #        import_protocol=$pre_import_protocol export_protocol=$pre_export_protocol novelty_buffer_capacity=$buffer topK=$topK\
       #        num_seed_steps=$num_seed_steps num_unsup_steps=$num_unsup_steps debug=True test=$test_name

       # # Deploy copies of pretrained agent to environment to generate trajectories using using GNU parallel
       # parallel -j "$num_cores" \
       #        "python -m learning_utils.generate_trajectories device=$device \
       #               domain=$domain env=$env render_mode=rgb_array \
       #               seed=$seed episodes_to_gen=$episodes_per_core \
       #               num_seed_steps=$num_seed_steps num_unsup_steps=$num_unsup_steps num_train_steps=$num_train_steps \
       #               debug=True test=$test_name cpu_id={#} 2>&1 | tee gen_traj_output_{#}.log >/dev/null" ::: $(seq "$num_cores") &

       # trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
       # wait

       # Reward Model Training from trajectories

       # Training script
       python -m learning_on_policy.training device=$device \
              domain=$domain env=$env render_mode=rgb_array max_episode_steps=$max_episode_steps seed=$seed architecture=$architecture \
              agent.discrete_action.sequence_length=$sequence_length\
              offline_epochs=$offline_epochs import_model=$import_model import_protocol=$import_protocol freeze_protocol=$freeze_protocol\
              num_seed_steps=$num_seed_steps num_unsup_steps=$num_unsup_steps num_train_steps=$num_train_steps debug=True test=$test_name
done