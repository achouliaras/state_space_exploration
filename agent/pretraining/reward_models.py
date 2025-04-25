import numpy as np
import torch
from agent.common.replay_buffers import NovelExperienceMemory
    
class IntrinsicRewardModel:
    def __init__(self, model=None, capacity=64, batch_size = 64, action_type ='Discrete', 
                 local_reward_coef=0.5, global_reward_coef=0.5, has_memory=False, k = 5, device= 'cpu'):
        self.agent = model
        self.model = self.agent.latentMDP
        encoder = self.model.encoder
        # encoder = model.create_network().double().to(device)
        # encoder.load_state_dict(self.model.encoder.state_dict())
        self.experience_memory = NovelExperienceMemory(capacity=capacity, k=k, encoder=encoder)
        self.eps = 1
        self.batch_size = batch_size
        self.action_type = action_type
        self.l_coef = local_reward_coef
        self.g_coef = global_reward_coef
        self.has_memory = has_memory
        self.device = device
        self.loss_mean = 0
        self.loss_std = 1

    def update_model_params(self, new_model_params):
        # self.model.load_state_dict(new_model_params)
        self.experience_memory.update_embeddings(self.model.encoder.state_dict())

    def calculate_intrinsic_reward(self, trajectory, next, logger, step, add=True, diff=False):
        obs, actions, logprobs, values, rewards, dones, memories = trajectory
        next_obs, next_done, next_memory = next

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
        
        local_reward = self.calculate_local_reward(trajectory, next, diff=diff)
        global_reward, num_novel = self.calculate_global_reward(obs, memories*mask,next_obs, next_memory*next_mask, add)
        
        # print(local_reward, global_reward)
        
        rewards = self.l_coef*local_reward + self.g_coef*global_reward
        
        logger.log('train/intrinsic_reward', rewards.mean(), step)
        logger.log('train/local_reward', local_reward.mean(), step)
        logger.log('train/global_reward', global_reward.mean(), step)
        logger.log('train/XPMem_usage', self.experience_memory.usage*100, step)
        logger.log('train/novel_states', num_novel, step)
        
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

        with torch.no_grad():
            if self.has_memory:
                pred_action_logprobs, z_hat_t1, _, _, z_t, z_t1  = self.model(obs, action, next_obs, 
                                                                            memory * mask, next_memory * next_mask)
            else:
                pred_action_logprobs, z_hat_t1, _, _, z_t, z_t1  = self.model(obs, action, next_obs)

            action_loss = self.model.eval_cross_entropy(pred_action_logprobs, action)

            # Transition loss
            loss = 1.0 * action_loss
        loss = loss.detach().cpu() # Find states the LMDP model can't understand
        abs_loss = torch.abs(loss).numpy()
        local_reward = abs_loss / (self.eps + abs_loss)
        global_reward, _ = self.calculate_global_reward(obs, memory*mask,next_obs, next_memory*next_mask, add)
        reward = self.l_coef*local_reward + self.g_coef*global_reward
        return reward[0]

    # Maximized by finding key states in the state space that are far away
    # Returns 0 if the state isn't dissimilar to it's kNN
    def calculate_global_reward(self, obs, memory, next_obs, next_memory, add=False):
        if add:
            if self.batch_size <= self.experience_memory.capacity:
                obs_novelty_score, num_novel = self.experience_memory.try_add(obs, memory)
            else:
                num_novel=0
                obs_novelty_score=torch.empty(self.batch_size)
                for i in range(0, self.batch_size, self.experience_memory.capacity):
                    obs_batch = obs[i:i+self.experience_memory.capacity]
                    memory_batch = memory[i:i+self.experience_memory.capacity]
                    batch_novelty_score, batch_num_novel = self.experience_memory.try_add(obs_batch, memory_batch)
                    obs_novelty_score[i:i+self.experience_memory.capacity]=batch_novelty_score
                    num_novel+=batch_num_novel
        else:
            obs_novelty_score, _, _ = self.experience_memory.estimate_novelty_score(obs, memory)
            num_novel = 0
        next_obs_novelty, _, _ = self.experience_memory.estimate_novelty_score(next_obs, next_memory)
        novelty_score = torch.clamp(next_obs_novelty - 0.5 * obs_novelty_score, min=0).detach().cpu().numpy()
        return novelty_score, num_novel

    # Maximized by finding misrepresented states by the latent MDP. 
    # Reward drops when the latent MDP model learns the neighborhood. 
    def calculate_local_reward(self, trajectory, next, diff=False):  
        # TODO: Add a way to calculate the loss from the encoder
        encoder_loss = self.agent.get_lmdp_loss(trajectory, next, diff=diff)
        if diff:    
            batch_loss, prev_batch_loss = encoder_loss
            novelty_diff = batch_loss - 0.5 * prev_batch_loss
            novelty_diff = novelty_diff / (1 + novelty_diff)
            local_reward = torch.clamp(novelty_diff, min=0).numpy()
        else:
            batch_loss = encoder_loss
            local_reward = batch_loss / (self.eps + batch_loss)
        return local_reward

    