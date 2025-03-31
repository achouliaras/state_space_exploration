test_name=LMDP_Offline_5E6 #Vanilla #Pret_Offline_Freeze #NoMemory
domain=MiniGrid # highway-env # ALE
env=Empty-8x8-v0 #Empty-8x8-v0 #UnlockPickup-v0 # highway-v0 # Breakout-v5
architecture=CNN-LSTM
offline_num_seed_steps=5e6
offline_epochs=100
import_model=True

off_export_protocol=OFFLINE
pre_import_protocol=NORMAL
pre_export_protocol=ONLINE
import_protocol=OFFLINE # NORMAL, OFFLINE, ONLINE

pre_freeze_protocol=NO
freeze_protocol=NO # NO, CNN-PART, CNN, ALL

num_seed_steps=0
num_unsup_steps=1e5
num_train_steps=100100
episodes_2_generate=16 
export PYTORCH_ENABLE_MPS_FALLBACK=1

device=cuda
# Get the number of available CPU cores
# num_cores=$(sysctl -n hw.physicalcpu)
num_cores=4

# Calculate episodes per process to generate
episodes_per_core=$(($episodes_2_generate / $num_cores))

# 1 2 3 4 5 6 7 8 9 10
for seed in 1 2 3 4 5 6 7 8 9 10; do
       # Offline Training script
       python -m learning_offline.pretraining device=$device \
              domain=$domain env=$env render_mode=rgb_array max_episode_steps=100 seed=$seed architecture=$architecture \
              offline_epochs=$offline_epochs export_protocol=$off_export_protocol \
              num_seed_steps=$offline_num_seed_steps debug=True test=$test_name

       # # Pretraining script
       # python -m learning_on_policy.pretraining device=$device \
       #        domain=$domain env=$env render_mode=rgb_array max_episode_steps=100 seed=$seed architecture=$architecture \
       #        offline_epochs=$offline_epochs import_model=$import_model freeze_protocol=$pre_freeze_protocol\
       #        import_protocol=$pre_import_protocol export_protocol=$pre_export_protocol \
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

       # # Training script
       # python -m learning_on_policy.training device=$device \
       #        domain=$domain env=$env render_mode=rgb_array max_episode_steps=100 seed=$seed architecture=$architecture \
       #        offline_epochs=$offline_epochs import_model=$import_model import_protocol=$import_protocol freeze_protocol=$freeze_protocol\
       #        num_seed_steps=$num_seed_steps num_unsup_steps=$num_unsup_steps num_train_steps=$num_train_steps debug=True test=$test_name
done