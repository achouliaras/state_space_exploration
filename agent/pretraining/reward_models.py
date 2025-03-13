import numpy as np
import torch
from agent.common.replay_buffers import NovelExperienceMemory
    
class IntrinsicRewardModel:
    def __init__(self, model=None, capacity=64, batch_size = 64, action_type ='Discrete', 
                 local_reward_coef=0.5, global_reward_coef=0.5, has_memory=False, k = 5, device= 'cpu'):  
        self.model = model
        encoder = model.create_network().double().to(device)
        encoder.load_state_dict(self.model.encoder.state_dict())
        self.experience_memory = NovelExperienceMemory(capacity=capacity, k=k, encoder=encoder)
        self.eps = 1 #self.experience_memory.eps
        self.batch_size = batch_size
        self.action_type = action_type
        self.l_coef = local_reward_coef
        self.g_coef = global_reward_coef
        self.has_memory = has_memory
        self.device = device
        # self.loss_mean = 0
        # self.loss_std = 1

    def update_model_params(self, new_model_params):
        self.model.load_state_dict(new_model_params)
        self.experience_memory.update_embeddings(self.model.encoder.state_dict())

    def calculate_intrinsic_reward(self, obs, actions, memories, dones, next_obs, next_done, next_memory, logger, step, add=True):
        obs = torch.DoubleTensor(obs).to(self.device).squeeze(1)
        if self.action_type == 'Continuous':
            actions = torch.DoubleTensor(actions).to(self.device).squeeze(1)
        elif self.action_type == 'Discrete':
            actions = torch.LongTensor(actions).to(self.device).squeeze(1).squeeze(1)
        mask = torch.DoubleTensor(np.array([1-dones])).to(self.device).squeeze(0)
        if self.has_memory: 
            memories = torch.DoubleTensor(memories).to(self.device).squeeze(1)        
        next_obs = torch.DoubleTensor(next_obs).to(self.device).unsqueeze(0)  
        next_obs = torch.cat([obs[1:], next_obs])     
        next_mask = torch.DoubleTensor(np.array([1-next_done])).to(self.device).unsqueeze(0)
        next_mask = torch.cat([mask[1:], next_mask])
        if self.has_memory:
            next_memory = torch.DoubleTensor(next_memory).to(self.device).unsqueeze(0)
            next_memory = torch.cat([memories[1:], next_memory])
        
        local_reward = self.calculate_local_reward(obs, actions, mask, memories, next_obs, next_mask, next_memory)
        global_reward = self.calculate_global_reward(next_obs, next_memory*next_mask, add)
        
        # print(local_reward, global_reward)
        
        rewards = self.l_coef*local_reward + self.g_coef*global_reward
        
        logger.log('train/intrinsic_reward', rewards.mean(), step)
        logger.log('train/local_reward', local_reward.mean(), step)
        logger.log('train/global_reward', global_reward.mean(), step)
        return rewards.reshape(-1,1)

    def r_hat(self, obs_tensor, action, memory_tensor, mask_tensor, next_obs, next_done, next_memory, logger, step, add=False):
        obs = obs_tensor.squeeze(1)
        if self.action_type == 'Continuous':
            action = torch.DoubleTensor((action)).to(self.device)
        elif self.action_type == 'Discrete':
            action = torch.LongTensor([action]).to(self.device)
        
        mask = mask_tensor
        if self.has_memory: 
            memory = memory_tensor.squeeze(1)    
        next_obs = torch.DoubleTensor(next_obs).to(self.device).unsqueeze(0)      
        next_mask = torch.DoubleTensor(np.array([1-next_done])).to(self.device).unsqueeze(0)
        if self.has_memory:
            next_memory = torch.DoubleTensor(next_memory).to(self.device).unsqueeze(0)
            next_memory = torch.cat([memory[1:], next_memory])

        local_reward = self.calculate_local_reward(obs, action, mask, memory, next_obs, next_mask, next_memory)
        global_reward = self.calculate_global_reward(next_obs, next_memory*next_mask, add)
        
        # print(local_reward, global_reward)

        reward = self.l_coef*local_reward + self.g_coef*global_reward

        # print(reward, local_reward, global_reward)
        # logger.log('train/intrinsic_reward', reward[0], step)
        # logger.log('train/local_reward', local_reward[0], step)
        # logger.log('train/global_reward',  global_reward[0], step)

        return reward[0]

    # Maximized by finding key states in the state space that are far away
    # Returns 0 if the state isn't at least 20% dissimilar to it's kNN
    def calculate_global_reward(self, obs, memory, add=False):
        if add:
            novelty_score = self.experience_memory.try_add(obs, memory)
        else:
            novelty_score, _, _ = self.experience_memory.estimate_novelty_score(obs, memory)
        novelty_score = torch.abs(novelty_score.detach().cpu()).numpy()
        novelty_score / (self.eps + novelty_score)
        return novelty_score

    # Maximized by finding misrepresented states by the latent MDP. 
    # Reward drops when the latent MDP model learns the neighborhood. 
    def calculate_local_reward(self, obs, actions, mask, memories, next_obs, next_mask, next_memory):        
        with torch.no_grad():
            if self.has_memory:
                # prediction_obs, _, memory  = self.autoencoder(obs_tensor, memory_tensor * mask_tensor)
                pred_action_logprobs, z_hat_t1, _, _, z_t, z_t1  = self.model(obs, actions, next_obs, 
                                                                            memories * mask, next_memory * next_mask)
            else:
                # prediction_obs, _, memory,  = self.autoencoder(obs_tensor)
                pred_action_logprobs, z_hat_t1, _, _, z_t, z_t1  = self.model(obs, actions, next_obs)

            action_loss = self.model.eval_cross_entropy(pred_action_logprobs, actions)
            
            z_a = z_t[:self.batch_size//2]
            z_b = z_t[self.batch_size//2:]
            
            contrastive_loss = self.model.hinge_loss(z_a, z_b) # not any effect, at least no harm
            transition_loss = self.model.mse_loss(z_hat_t1, z_t1) # Decent alone, not good with action loss
            # transition_loss = self.model.wasserstein_loss(z_hat_t1, z_t1) # Not good enough
            # locality_loss = self.model.l1_loss(z_t, z_t1) # Big performance drop
            # locality_loss = self.model.wasserstein_loss(z_t, z_t1) # Performance drop

            # Bisimilarity loss
            loss = 1.0 * action_loss
            
            if loss.shape[0] > 1:
                # self.loss_mean = loss.mean()
                # self.loss_std = loss.std()
                loss += 0.1 * contrastive_loss
        
        # loss = (loss-self.loss_mean) / self.loss_std
        loss = loss.detach().cpu() # Find states the LMDP model can't understand
        abs_loss = torch.abs(loss).numpy()
        return abs_loss / (self.eps + abs_loss)

    