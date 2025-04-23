import numpy as np
import torch
from gymnasium.spaces import utils as gym_utils
import torchrl
from torchrl.data import LazyTensorStorage
from torchrl.data.replay_buffers.samplers import RandomSampler

class NovelExperienceMemory(torchrl.data.ReplayBuffer):
    def __init__(self, capacity=1024, k=5, threshold=0.01, device="cuda", encoder=None):
        super().__init__(
            storage=LazyTensorStorage(capacity, device=device),
            sampler=RandomSampler(),
            batch_size=32
        )
        self.capacity = capacity
        self.k = k
        self.eps = 1e-4
        self.threshold = threshold
        self.device = device
        self.encoder = encoder
        
        # Keep track of states for fast similarity checking
        self.observations = torch.zeros((capacity, *self.encoder.obs_shape), dtype=torch.double, device=device)
        self.memories = torch.zeros((capacity, self.encoder.memory_size), dtype=torch.double, device=device)
        self.state_embeddings = torch.zeros((capacity, self.encoder.embedding_size), dtype=torch.double ,device=device)
        self.novelty_scores = torch.zeros(capacity, device=device)
        self.ptr = 0
        self.full =False
    
    @property
    def usage(self):
        return self.ptr / self.capacity

    def update_embeddings(self, new_encoder_params):
        """Refresh all embeddings using the new encoder"""
        self.encoder.load_state_dict(new_encoder_params)

        if self.ptr == 0:
            return
            
        # Get valid observations and Memories
        valid_obs = self.observations[:self.ptr] if not self.full else self.observations
        valid_memories = self.memories[:self.ptr] if not self.full else self.memories
        
        # Re-encode all stored observations
        new_embeddings, new_memories= self._encode_states(valid_obs, valid_memories)
        
        # Update embeddings in place
        if not self.full:
            self.state_embeddings[:self.ptr] = new_embeddings
            self.memories[:self.ptr] = new_memories
        else:
            self.state_embeddings = new_embeddings
            self.memories = new_memories
        
        # Recalculate novelty scores
        self._update_novelty_scores()

    def estimate_novelty_score(self, obs, memory):
        batch_size = obs.shape[0]

        obs = obs.detach().to(self.device)
        memory = memory.detach().to(self.device)

        # Calculate embeddings
        state_embeddings, _ = self._encode_states(obs, memory)

        # Calculate novelty score for each state
        if len(self) > 0:
            valid_embeddings = self.state_embeddings[:self.ptr] if not self.full else self.state_embeddings
            similarities = self._knn_similarity(state_embeddings, valid_embeddings)
        else:
            similarities = torch.zeros(batch_size, device=self.device)

        # Select novel states. 1 for novel states 0 for not novel enough
        novel_mask = similarities < self.threshold
        return (1-similarities)*novel_mask, similarities, novel_mask

    def try_add(self, obs, memory):
        """Add state batch to buffer if novel, with eviction policy"""
        reward, similarities, novel_mask = self.estimate_novelty_score(obs, memory)
        
        # Calculate embeddings
        state_embeddings, _ = self._encode_states(obs, memory)

        novel_obs = obs[novel_mask]
        novel_mem = memory[novel_mask]
        novel_embeddings = state_embeddings[novel_mask]
        num_novel = novel_mask.sum().item()
        
        # print("Added: ",num_novel)
        # Add novel states to buffer
        if num_novel > 0:
            remaining_capacity = self.capacity - self.ptr
            if  num_novel <= remaining_capacity:
                # Add to end of buffer
                self.observations[self.ptr:(self.ptr+num_novel)] = novel_obs
                self.memories[self.ptr:(self.ptr+num_novel)] = novel_mem
                self.state_embeddings[self.ptr:(self.ptr+num_novel)] = novel_embeddings
                self.ptr += num_novel
            elif num_novel > remaining_capacity:
                # Fill remaining space and evict old entries
                self.observations[self.ptr:] = novel_obs[:remaining_capacity]
                self.memories[self.ptr:] = novel_mem[:remaining_capacity]
                self.state_embeddings[self.ptr:] = novel_embeddings[:remaining_capacity]
                
                # Evict least novel entries for remaining
                _, evict_indices = torch.topk(self.novelty_scores, k=num_novel - remaining_capacity, largest=False)
                
                # Replace evicted entries
                self.observations[evict_indices] = novel_obs[remaining_capacity:]
                self.memories[evict_indices] = novel_mem[remaining_capacity:]
                self.state_embeddings[evict_indices] = novel_embeddings[remaining_capacity:]
                self.ptr += remaining_capacity
                self.full = True
            
            # Update novelty scores
            self._update_novelty_scores()

        return reward, num_novel

    def _encode_states(self, obs, memory):
        """Encode observations using current encoder"""
        with torch.no_grad():
            z_t, memory = self.encoder(obs, memory)
            return z_t, memory
        
    def _knn_similarity(self, query_embeddings, target_embeddings):
        """Compute average similarity to k-nearest neighbors"""
        # Calculate pairwise distances
        dists = torch.cdist(query_embeddings, target_embeddings, p=2)
        
        # Find k-nearest neighbors for each query
        knn_dists, _ = torch.topk(dists, k=self.k, dim=1, largest=False, sorted=True)

        # Convert distances to similarities (you can use other similarity measures)
        similarities = self.eps / (self.eps + knn_dists.mean(dim=1))

        # # Normalize embeddings to unit length
        # query_embeddings = query_embeddings / query_embeddings.norm(dim=1, keepdim=True)
        # target_embeddings = target_embeddings / target_embeddings.norm(dim=1, keepdim=True)
        
        # # Compute cosine similarity (dot product of normalized vectors)
        # cosine_similarities = torch.mm(query_embeddings, target_embeddings.t())
        
        # # Find k-nearest neighbors for each query
        # knn_sims, _ = torch.topk(cosine_similarities, k=self.k, dim=1, largest=True, sorted=True)
        
        # # Convert distances to similarities (you can use other similarity measures)
        # similarities = knn_sims.mean(dim=1)
        return similarities    

    def _update_novelty_scores(self):
        """Update novelty scores for all states in buffer"""
        if len(self) < 2:
            return
            
        # Compute pairwise similarities
        valid_embeddings = self.state_embeddings[:self.ptr] if not self.full else self.state_embeddings

        dists = torch.cdist(valid_embeddings, valid_embeddings, p=2)
        eye = torch.eye(valid_embeddings.shape[0], device=self.device).bool()
        dists[eye] = float('inf')  # Ignore self-similarity
        # Get k-nearest similarities for each state
        knn_dists, _ = torch.topk(dists, k=self.k, dim=1, largest=False)
        mean_kdist = knn_dists.mean(dim=1)
        self.novelty_scores[:len(valid_embeddings)] = mean_kdist / (self.eps + mean_kdist)

        # valid_embeddings = valid_embeddings / valid_embeddings.norm(dim=1, keepdim=True)
        # cosine_similarities = torch.mm(valid_embeddings, valid_embeddings.t())
        # eye = torch.eye(valid_embeddings.shape[0], device=self.device).bool()
        # cosine_similarities[eye] = 0  # Ignore self-similarity        
        # knn_sims, _ = torch.topk(cosine_similarities, k=self.k, dim=1, largest=True)
        # mean_sims = knn_sims.mean(dim=1)
        # self.novelty_scores[:len(valid_embeddings)] = mean_sims

    def __len__(self):
        return self.ptr if not self.full else self.capacity

class TrajectoryBuffer(object):
    """Buffer to store environment transitions."""
    def __init__(self, obs_shape, action_shape, capacity, has_memory, memory_size=128, window=1):
        self.capacity = capacity
        self.has_memory = has_memory
        self.memory_size = memory_size

        self.obses = np.empty((capacity, 1, *obs_shape), dtype=np.float64)
        self.actions = np.empty((capacity, 1, *action_shape), dtype=np.float64)
        self.log_probs = np.empty((capacity, 1), dtype=np.float64)
        self.rewards = np.empty((capacity, 1), dtype=np.float64)
        self.dones = np.empty((capacity, 1), dtype=np.float64)
        self.values = np.empty((capacity, 1), dtype=np.float64)
        self.memories = None
        if self.has_memory:
            self.memories = np.empty((capacity, 1, *memory_size), dtype=np.float64)

        self.window = window

        self.idx = 0
        self.last_save = 0
        self.full = False

    def __len__(self):
        return self.capacity if self.full else self.idx

    def add(self, trajectory, next_state):
        obs, action, logprob, value, reward, next_obs, done, memory = trajectory
        next_obs, next_done, next_memory = next_state

        self.next_obs = next_obs
        self.next_done = next_done
        self.next_memory = next_memory

        np.copyto(self.obses[self.idx], obs)
        np.copyto(self.actions[self.idx], action)
        np.copyto(self.log_probs[self.idx], logprob)
        np.copyto(self.values[self.idx], value)
        np.copyto(self.rewards[self.idx], reward)
        np.copyto(self.dones[self.idx], done)
        if self.has_memory:
            np.copyto(self.memories[self.idx], memory)

        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0
    
    def add_batch(self, trajectory, next_state):
        obs, action, logprob, value, reward, done, memory = trajectory
        next_obs, next_done, next_memory = next_state

        self.next_obs = next_obs
        self.next_done = next_done
        self.next_memory = next_memory
        
        next_index = self.idx + self.window
        if next_index >= self.capacity:
            self.full = True
            maximum_index = self.capacity - self.idx
            np.copyto(self.obses[self.idx:self.capacity], obs[:maximum_index])
            np.copyto(self.actions[self.idx:self.capacity], action[:maximum_index])
            np.copyto(self.log_probs[self.idx:self.capacity], logprob[:maximum_index])
            np.copyto(self.values[self.idx:self.capacity], value[:maximum_index])
            np.copyto(self.rewards[self.idx:self.capacity], reward[:maximum_index])
            np.copyto(self.dones[self.idx:self.capacity], done[:maximum_index])
            if self.has_memory:
                np.copyto(self.memories[self.idx:self.capacity], memory[:maximum_index])

            remain = self.window - (maximum_index)
            if remain > 0:
                np.copyto(self.obses[0:remain], obs[maximum_index:])
                np.copyto(self.actions[0:remain], action[maximum_index:])
                np.copyto(self.log_probs[0:remain], logprob[maximum_index:])
                np.copyto(self.rewards[0:remain], reward[maximum_index:])
                np.copyto(self.dones[0:remain], done[maximum_index:])
                np.copyto(self.values[0:remain], value[maximum_index:])
                if self.has_memory:
                    np.copyto(self.memories[0:remain], memory[maximum_index:])
            self.idx = remain
        else:
            np.copyto(self.obses[self.idx:next_index], obs)
            np.copyto(self.actions[self.idx:next_index], action)
            np.copyto(self.log_probs[self.idx:next_index], logprob)
            np.copyto(self.values[self.idx:next_index], value)
            np.copyto(self.rewards[self.idx:next_index], reward)
            np.copyto(self.dones[self.idx:next_index], done)
            if self.has_memory:
                np.copyto(self.memories[self.idx:next_index], memory)
            self.idx = next_index
    
    def get_data(self):
        cap = self.capacity if self.full else self.idx
        obses = np.concatenate((self.obses[self.idx:cap], self.obses[:self.idx])) 
        actions = np.concatenate((self.actions[self.idx:cap], self.actions[:self.idx]))
        log_probs = np.concatenate((self.log_probs[self.idx:cap],self.log_probs[:self.idx]))
        values = np.concatenate((self.values[self.idx:cap],self.values[:self.idx]))
        rewards = np.concatenate((self.rewards[self.idx:cap],self.rewards[:self.idx]))
        dones = np.concatenate((self.dones[self.idx:cap],self.dones[:self.idx]))
        if self.has_memory:
            memories = np.concatenate((self.memories[self.idx:cap],self.memories[:self.idx]))
        next_obs = self.next_obs
        next_done = self.next_done
        next_memory = self.next_memory
        return [obses, actions, log_probs, values, rewards, dones, memories],[next_obs,next_done,next_memory]

class ReplayBuffer(object):
    """Buffer to store environment transitions."""
    def __init__(self, obs_space, obs_shape, action_shape, action_type, capacity, device, window=1):
        self.capacity = capacity
        self.device = device
        self.obs_space=obs_space

        # the proprioceptive obs is stored as float32, pixels obs as uint8
        #obs_dtype = np.float32 if len(obs_shape) == 1 else np.uint8
        obs_dtype = np.float32
        act_dtype = np.float32 if action_type == 'Continuous' else np.uint8

        self.obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        self.next_obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        self.actions = np.empty((capacity, *action_shape), dtype=act_dtype)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones_no_max = np.empty((capacity, 1), dtype=np.float32)
        self.window = window

        self.idx = 0
        self.last_save = 0
        self.full = False

    def __len__(self):
        return self.capacity if self.full else self.idx

    def add(self, obs, action, reward, next_obs, done, done_no_max):
        np.copyto(self.obses[self.idx], obs)
        np.copyto(self.actions[self.idx], action)
        np.copyto(self.rewards[self.idx], reward)
        np.copyto(self.next_obses[self.idx], next_obs)
        np.copyto(self.not_dones[self.idx], not done)
        np.copyto(self.not_dones_no_max[self.idx], not done_no_max)

        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0
    
    def add_batch(self, obs, action, reward, next_obs, done, done_no_max):
        
        next_index = self.idx + self.window
        if next_index >= self.capacity:
            self.full = True
            maximum_index = self.capacity - self.idx
            np.copyto(self.obses[self.idx:self.capacity], obs[:maximum_index])
            np.copyto(self.actions[self.idx:self.capacity], action[:maximum_index])
            np.copyto(self.rewards[self.idx:self.capacity], reward[:maximum_index])
            np.copyto(self.next_obses[self.idx:self.capacity], next_obs[:maximum_index])
            np.copyto(self.not_dones[self.idx:self.capacity], done[:maximum_index] <= 0)
            np.copyto(self.not_dones_no_max[self.idx:self.capacity], done_no_max[:maximum_index] <= 0)
            remain = self.window - (maximum_index)
            if remain > 0:
                np.copyto(self.obses[0:remain], obs[maximum_index:])
                np.copyto(self.actions[0:remain], action[maximum_index:])
                np.copyto(self.rewards[0:remain], reward[maximum_index:])
                np.copyto(self.next_obses[0:remain], next_obs[maximum_index:])
                np.copyto(self.not_dones[0:remain], done[maximum_index:] <= 0)
                np.copyto(self.not_dones_no_max[0:remain], done_no_max[maximum_index:] <= 0)
            self.idx = remain
        else:
            np.copyto(self.obses[self.idx:next_index], obs)
            np.copyto(self.actions[self.idx:next_index], action)
            np.copyto(self.rewards[self.idx:next_index], reward)
            np.copyto(self.next_obses[self.idx:next_index], next_obs)
            np.copyto(self.not_dones[self.idx:next_index], done <= 0)
            np.copyto(self.not_dones_no_max[self.idx:next_index], done_no_max <= 0)
            self.idx = next_index
        
    def relabel_with_predictor(self, predictor):
        batch_size = 200
        total_iter = int(self.idx/batch_size)
        
        if self.idx > batch_size*total_iter:
            total_iter += 1
            
        for index in range(total_iter):
            last_index = (index+1)*batch_size
            if (index+1)*batch_size > self.idx:
                last_index = self.idx
                
            obses = self.obses[index*batch_size:last_index]
            actions = self.actions[index*batch_size:last_index]

            obs_flat = np.array([gym_utils.flatten(self.obs_space, i) for i in obses])
            
            inputs = np.concatenate([obs_flat, actions], axis=-1)
            
            pred_reward = predictor.r_hat_batch(inputs)
            self.rewards[index*batch_size:last_index] = pred_reward
            
    def sample_tensors(self, batch_size):
        idxs = np.random.randint(0,self.capacity if self.full else self.idx, size=batch_size)
        obses = torch.as_tensor(self.obses[idxs], dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        next_obses = torch.as_tensor(self.next_obses[idxs], dtype=torch.float32,
                                     device=self.device)
        not_dones = torch.as_tensor(self.not_dones[idxs], device=self.device)
        not_dones_no_max = torch.as_tensor(self.not_dones_no_max[idxs],
                                           device=self.device)

        return obses, actions, rewards, next_obses, not_dones, not_dones_no_max
    
    def sample(self, batch_size):
        idxs = np.random.randint(0,self.capacity if self.full else self.idx, size=batch_size)
        obses = self.obses[idxs]
        actions = self.actions[idxs]
        rewards = self.rewards[idxs]
        next_obses = self.next_obses[idxs]
        not_dones = self.not_dones[idxs]
        not_dones_no_max = self.not_dones_no_max[idxs]

        return obses, actions, rewards, next_obses, not_dones, not_dones_no_max
    
    def sample_state_ent(self, batch_size):
        obses, actions, rewards, next_obses, not_dones, not_dones_no_max = self.sample_tensors(batch_size)
                
        if self.full:
            full_obs = self.obses
        else:
            full_obs = self.obses[: self.idx]
        full_obs = torch.as_tensor(full_obs, dtype=torch.float32, device=self.device)
        
        return obses, full_obs, actions, rewards, next_obses, not_dones, not_dones_no_max
    
    def save(self, model_dir, step):
        keys_to_save = ['obses', 'next_obses', 'actions', 'rewards', 'not_dones', 'not_dones_no_max', 'idx']
        payload = {k: self.__dict__[k] for k in keys_to_save}
        torch.save(payload, '%s/replay_buffer_%s.pt' % (model_dir, step), pickle_protocol=4)
        
    def load(self, model_dir, step):
        keys_to_load = ['obses', 'next_obses', 'actions', 'rewards', 'not_dones', 'not_dones_no_max', 'idx']
        payload = torch.load('%s/replay_buffer_%s.pt' % (model_dir, step))
        self.obses, self.next_obses, self.actions, self.rewards, self.not_dones, self.not_dones_no_max, self.idx = [payload[k] for k in keys_to_load]

        self.idx = (self.idx + 1) % self.capacity
        self.full = False or self.idx == 0