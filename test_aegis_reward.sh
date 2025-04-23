# Test of Aegis performance for various novelty buffer sizes
domain=MiniGrid
env=DistShift2-v0 #Empty-8x8-v0 #UnlockPickup-v0 # highway-v0 # Breakout-v5
max_episode_steps=100
architecture=CNN-LSTM
sequence_length=4
device=cuda

# Offline Pre-Training
offline_num_seed_steps=25e4
offline_epochs=10
off_export_protocol=OFFLINE

# Online Pre-Training
num_seed_steps=0
num_unsup_steps=15e4
pre_offline_epochs=10
topK=5
pre_import_model=False
pre_import_protocol=NORMAL
pre_export_protocol=ONLINE
pre_freeze_protocol=NO # NO, CNN-PART, CNN, ALL

# Training
num_train_steps=150100
import_model=True
import_protocol=ONLINE # NORMAL, OFFLINE, ONLINE
freeze_protocol=NO # NO, CNN-PART, CNN, ALL

# 64 128 256 512 1024
for buffer in 256; do
       test_name="AEGIS_NovelD"
       # test_name="AEGIS_Buffer_{}"
       # buffer={}

       # 1 2 3 4 5 6 7 8 9 10
       for seed in 1 2 3 4 5 6 7 8 9 10; do
              # # Offline Training script
              # python -m learning_offline.pretraining device=$device \
              #        domain=$domain env=$env render_mode=rgb_array max_episode_steps=$max_episode_steps seed=$seed architecture=$architecture \
              #        agent.discrete_action.sequence_length=$sequence_length \
              #        offline_epochs=$offline_epochs export_protocol=$off_export_protocol \
              #        num_seed_steps=$offline_num_seed_steps debug=True test=$test_name

              # Pretraining script
              python -m learning_on_policy.pretraining device=$device \
                     domain=$domain env=$env render_mode=rgb_array max_episode_steps=$max_episode_steps seed=$seed architecture=$architecture \
                     agent.discrete_action.sequence_length=$sequence_length \
                     offline_epochs=$pre_offline_epochs import_model=$pre_import_model freeze_protocol=$pre_freeze_protocol\
                     import_protocol=$pre_import_protocol export_protocol=$pre_export_protocol novelty_buffer_capacity=$buffer topK=$topK\
                     num_seed_steps=$num_seed_steps num_unsup_steps=$num_unsup_steps debug=True test=$test_name
              
              # Training script
              python -m learning_on_policy.training device=$device \
                     domain=$domain env=$env render_mode=rgb_array max_episode_steps=$max_episode_steps seed=$seed architecture=$architecture \
                     agent.discrete_action.sequence_length=$sequence_length \
                     offline_epochs=$offline_epochs import_model=$import_model import_protocol=$import_protocol freeze_protocol=$freeze_protocol\
                     num_seed_steps=$num_seed_steps num_unsup_steps=$num_unsup_steps num_train_steps=$num_train_steps debug=True test=$test_name
       done
done