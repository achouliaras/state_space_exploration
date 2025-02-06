import lib.env_setup as env_setup
import lib.agent_setup as agent_setup
import torch
import numpy as np
from tqdm import tqdm

def evaluate_agent(agent, cfg, logger):
    cfg.save_video=True
    env, cfg, obs_space = env_setup.make_env(cfg, cfg.render_mode)
    
    average_episode_reward = 0
    average_true_episode_reward = 0
    success_rate = 0
    step = 0
    print('EVALUATION STARTS')
    for episode in tqdm(range(cfg.num_eval_episodes)):
        obs, _ = env.reset()
        # obs, _, _, _, _ = env.step(1) # FIRE action for breakout
        terminated = False
        truncated = False
        done = 0 

        episode_reward = 0
        true_episode_reward = 0
        if agent.has_memory:
            memory = np.zeros(agent.memory_size)

        if cfg.log_success:
            episode_success = 0

        while not (terminated or truncated):
            
            action = env.action_space.sample()
            with agent_setup.eval_mode(agent):
                if agent.has_memory:
                    obs_tensor = torch.DoubleTensor(obs).to(cfg.device).unsqueeze(0)
                    memory_tensor = torch.DoubleTensor(memory).to(cfg.device).unsqueeze(0)
                    mask_tensor = torch.DoubleTensor(1-done).to(cfg.device).unsqueeze(0)
                    # print(memory_tensor.shape)
                    # print(mask_tensor.shape)
                    action, logprob, _, value, memory = agent.get_action(obs=obs_tensor,
                                                                        action=None,
                                                                        memory=memory_tensor * mask_tensor)
                    memory = memory.detach().cpu().numpy()[0]
                else:
                    action, logprob, _, value, _ = agent.get_action(torch.DoubleTensor(obs).to(cfg.device).unsqueeze(0))
            action = action.detach().cpu().numpy()[0]

            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            true_episode_reward += reward
            if cfg.log_success:
                episode_success = max(episode_success, terminated)
            
            obs = next_obs
            step += 1
            
        average_episode_reward += episode_reward
        average_true_episode_reward += true_episode_reward
        if cfg.log_success:
            success_rate += episode_success
    
    env.close()
    average_episode_reward /= cfg.num_eval_episodes
    average_true_episode_reward /= cfg.num_eval_episodes
    if cfg.log_success:
        success_rate /= cfg.num_eval_episodes
        success_rate *= 100.0
    
    logger.log('eval/episode', episode+1, step)
    logger.log('eval/avg_episode_reward', average_episode_reward, step)
    logger.log('eval/avg_true_episode_reward', average_true_episode_reward, step)
    if cfg.log_success:
        logger.log('eval/success_rate', success_rate, step)
    logger.dump(step, ty='eval')
    print('EVALUATION FINISHED')
    return logger