import lib.env_setup as env_setup
import lib.agent_setup as agent_setup

def evaluate_agent(agent, cfg, logger):
    cfg.save_video=True
    env, cfg, obs_space = env_setup.make_env(cfg, cfg.render_mode)
    
    average_episode_reward = 0
    average_true_episode_reward = 0
    success_rate = 0
    step = 0
    print('EVALUATION STARTS')
    for episode in range(cfg.num_eval_episodes):
        obs, info = env.reset(seed = cfg.seed)
        if cfg.action_type == 'Discrete' and cfg.state_type == 'grid':
            obs = obs['image']
        # agent.reset()
        terminated = False
        truncated = False
        episode_reward = 0
        true_episode_reward = 0
        if cfg.log_success:
            episode_success = 0

        while not (terminated or truncated):
            # with agent_setup.eval_mode(agent):
            #     action = agent.act(obs, sample=False, determ=False) # set determ=True in experiments
            #     #print(action)
            action = env.action_space.sample()
            next_obs, reward, terminated, truncated, info = env.step(action)

            episode_reward += reward
            true_episode_reward += reward
            if cfg.log_success:
                episode_success = max(episode_success, terminated)
            if cfg.action_type == 'Discrete' and cfg.state_type == 'grid':
                next_obs = next_obs['image']
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
        # logger.log('train/true_episode_success', success_rate, step)
    logger.dump(step)
    print('EVALUAITON FINISHED')
    return logger