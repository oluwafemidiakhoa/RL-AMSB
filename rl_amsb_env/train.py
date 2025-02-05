import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy

def train_rl_model_selection(env, total_timesteps=10000):
    """
    Train a DQN agent on the custom environment.
    """
    model = DQN("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=total_timesteps)
    return model

def evaluate_rl_agent(model, env, n_eval_episodes=10):
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=n_eval_episodes, deterministic=True)
    print(f"Mean Reward: {mean_reward:.3f} ± {std_reward:.3f}")
    return mean_reward, std_reward
