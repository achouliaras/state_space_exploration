import torch
from agent.common.replay_buffers import NovelExperienceMemory
    
class IntrinsicRewardModel:
    def __init__(self, LMDP_model=None, capacity=64, batch_size = 64, local_reward_coef=0.5, global_reward_coef=0.5):  
        self.LMDP_model = LMDP_model
        encoder = LMDP_model.create_network()
        encoder.load_state_dict(self.LMDP_model.state_dict())
        self.experience_memory = NovelExperienceMemory(capacity=capacity, encoder=encoder)
        self.batch_size = batch_size
        self.l_coef = local_reward_coef
        self.g_coef = global_reward_coef

    def update_LMDP_params(self, new_LMDP_params):
        self.LMDP_model.load_state_dict(new_LMDP_params)
        self.experience_memory.update_embeddings(self.LMDP_model.encoder.state_dict())

    def calculate_intrinsic_reward(self, obs, actions, memories, dones, next_obs, next_done, next_memory, logger, step):
        obs = torch.DoubleTensor(obs).to(self.device)
        actions = torch.DoubleTensor(actions).to(self.device)
        mask = torch.DoubleTensor([1-dones]).to(self.device)
        if self.has_memory: 
            memories = torch.DoubleTensor(memories).to(self.device)
        
        next_obs = torch.DoubleTensor(next_obs).to(self.device).unsqueeze(0)      
        next_obs = torch.cat([obs[1:], next_obs])
        next_mask = torch.DoubleTensor([1-next_done]).to(self.device).unsqueeze(0)
        next_mask = torch.cat([mask[1:], next_mask])
        if self.has_memory:
            next_memory = torch.DoubleTensor(next_memory).to(self.device).unsqueeze(0)
            next_memory = torch.cat([memories[1:], next_memory])
        
        local_reward = self.calculate_local_reward(obs, actions, mask, memories, next_obs, next_mask, next_memory)
        global_reward = self.calculate_global_reward(next_obs)

        rewards = self.l_coef*local_reward + self.g_coef*global_reward

        logger.log('train/intrinsic_reward', rewards.mean(), step)
        logger.log('train/local_reward', local_reward.mean(), step)
        logger.log('train/global_reward', global_reward.mean(), step)
        
        return torch.DoubleTensor(rewards).to(self.device)

    # Maximized by finding key states in the state space that are far away
    # Returns 0 if the state isn't at least 20% dissimilar to it's kNN
    def calculate_global_reward(self, obs):
        novelty_score = self.experience_memory.try_add(obs)
        return novelty_score

    # Maximized by finding misrepresented states by the latent MDP. 
    # Reward drops when the latent MDP model learns the neighborhood. 
    def calculate_local_reward(self, obs, actions, mask, memories, next_obs, next_mask, next_memory):
        print(obs.shape)
        print(next_obs.shape)
        
        with torch.no_grad():
            if self.has_memory:
                # prediction_obs, _, memory  = self.autoencoder(obs_tensor, memory_tensor * mask_tensor)
                pred_action_logprobs, z_hat_t1, _, _, z_t, z_t1  = self.LMDP_model(obs, actions, next_obs, 
                                                                            memories * mask, next_memory * next_mask)
            else:
                # prediction_obs, _, memory,  = self.autoencoder(obs_tensor)
                pred_action_logprobs, z_hat_t1, _, _, z_t, z_t1  = self.LMDP_model(obs, actions, next_obs)

            action_loss = self.cross_entropy_loss(pred_action_logprobs, actions)
            
            z_a = z_t[:self.batch_size//2]
            z_b = z_t[self.batch_size//2:]
            
            contrastive_loss = self.hinge_loss(z_a, z_b) # not any effect, at least no harm
            transition_loss = self.mse_loss(z_hat_t1, z_t1) # Decent alone, not good with action loss
            # transition_loss = self.wasserstein_loss(z_hat_t1, z_t1) # Not good enough
            # locality_loss = self.l1_loss(z_t, z_t1) # Big performance drop
            # locality_loss = self.wasserstein_loss(z_t, z_t1) # Performance drop

            # Bisimilarity loss
            loss = 1.0 * action_loss
            loss = loss.detach().cpu().numpy()[0] # Find states the LMDP model can't understand
            abs_loss = torch.abs(loss)
            
        return abs_loss / (self.eps + abs_loss)

    