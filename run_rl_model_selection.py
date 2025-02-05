import pandas as pd
import torch
from rl_amsb_env.environment import ModelSelectionEnv
from rl_amsb_env.train import train_rl_model_selection, evaluate_rl_agent

if __name__ == "__main__":
    # Example: Load a local CSV (e.g., heart.csv)
    data = pd.read_csv("data/heart.csv")
    
    # Basic assumption: there's a "target" column.
    target_column = "target"
    
    # Potential models
    model_candidates = ["XGBoost", "LightGBM", "DNN"]
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    env = ModelSelectionEnv(data, target_column, model_candidates, device=device)
    
    # Train the agent
    rl_model = train_rl_model_selection(env, total_timesteps=10000)

    # Evaluate the agent
    evaluate_rl_agent(rl_model, env, n_eval_episodes=10)
