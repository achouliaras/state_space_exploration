domain=MiniGrid
env2=Unlock-v0
env3=UnlockPickup-v0
env4=BlockedUnlockPickup-v0
max_episode_steps=200
architecture=CNN
sequence_length=5
device=cuda

# Offline Pre-Training
offline_num_seed_steps=5e5
offline_epochs=20
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

# First Training
# num_train_steps=1000000
import_model=True
import_protocol=NORMAL # NORMAL, OFFLINE, ONLINE, OFFLINE-CURRICULUM
export_protocol=NORMAL
freeze_protocol=NO # NO, CNN-PART, CNN, ALL

experiment_type=Single_Task
test_name=Vanilla #Vanilla, LMDP_Offline, AE_Offline, AEGIS

num_train_steps=2000000
for seed in 1 2 3 4 5 6 7 8 9 10; do
       echo "Training for seed $seed in env $env2"
       # Training script
       python -m learning_on_policy.training device=$device \
              domain=$domain env=$env2 render_mode=rgb_array max_episode_steps=$max_episode_steps seed=$seed architecture=$architecture \
              agent.discrete_action.sequence_length=$sequence_length offline_epochs=$offline_epochs freeze_protocol=$freeze_protocol\
              import_model=$import_model import_protocol=$import_protocol export_protocol=$export_protocol experiment_type=$experiment_type\
              num_seed_steps=$num_seed_steps num_unsup_steps=$num_unsup_steps num_train_steps=$num_train_steps debug=True test=$test_name  
done

num_train_steps=3000000
for seed in 1 2 3 4 5 6 7 8 9 10; do
       echo "Training for seed $seed in env $env3"
       # Training script
       python -m learning_on_policy.training device=$device \
              domain=$domain env=$env3 render_mode=rgb_array max_episode_steps=$max_episode_steps seed=$seed architecture=$architecture \
              agent.discrete_action.sequence_length=$sequence_length offline_epochs=$offline_epochs freeze_protocol=$freeze_protocol\
              import_model=$import_model import_protocol=$import_protocol export_protocol=$export_protocol experiment_type=$experiment_type\
              num_seed_steps=$num_seed_steps num_unsup_steps=$num_unsup_steps num_train_steps=$num_train_steps debug=True test=$test_name  
done

num_train_steps=4000000
for seed in 1 2 3 4 5 6 7 8 9 10; do
       echo "Training for seed $seed in env $env4"
       # Training script
       python -m learning_on_policy.training device=$device \
              domain=$domain env=$env4 render_mode=rgb_array max_episode_steps=$max_episode_steps seed=$seed architecture=$architecture \
              agent.discrete_action.sequence_length=$sequence_length offline_epochs=$offline_epochs freeze_protocol=$freeze_protocol\
              import_model=$import_model import_protocol=$import_protocol export_protocol=$export_protocol experiment_type=$experiment_type\
              num_seed_steps=$num_seed_steps num_unsup_steps=$num_unsup_steps num_train_steps=$num_train_steps debug=True test=$test_name  
done