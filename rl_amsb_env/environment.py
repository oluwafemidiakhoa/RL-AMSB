import gymnasium as gym
import numpy as np
import pandas as pd
import torch
from gymnasium import spaces
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from .models import DNNModel
from .utils import calc_auc_score, calc_f1_score

class ModelSelectionEnv(gym.Env):
    """
    Custom Gym environment that encapsulates model selection logic.
    """
    def __init__(self, data, target_column, model_candidates, vectorizer=None, device='cpu'):
        super().__init__()
        self.data = data
        self.target_column = target_column
        self.model_candidates = model_candidates
        self.device = device
        self.vectorizer = vectorizer or TfidfVectorizer(max_features=100)
        if "text" in data.columns:
            self.vectorizer.fit(data["text"])

        # Action: which model to pick
        self.action_space = spaces.Discrete(len(model_candidates))
        
        # Observation: summary stats
        obs_size = self._get_state().shape[0]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32)

        self.current_step = 0
        self.done = False

    def _get_state(self):
        """
        Extract summary stats from the data: numeric means, std, skew, kurtosis, text embeddings, etc.
        """
        X = self.data.drop(self.target_column, axis=1)
        text_features = np.zeros((len(X), 0))
        if "text" in X.columns:
            text_features = self.vectorizer.transform(X["text"]).toarray()

        X_numeric = X.select_dtypes(include=np.number)
        if len(X_numeric.columns) > 0:
            state = np.concatenate([
                X_numeric.mean().values,
                X_numeric.std().values,
                X_numeric.skew().values,
                X_numeric.kurtosis().values,
                text_features.mean(axis=0),
            ])
        else:
            state = text_features.mean(axis=0)

        return state

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.done = False
        return self._get_state().astype(np.float32), {}

    def step(self, action):
        model_name = self.model_candidates[action]
        reward, auc, f1 = self._evaluate_model(model_name)
        
        self.current_step += 1
        if self.current_step >= 1:
            self.done = True

        return self._get_state().astype(np.float32), reward, self.done, False, {
            "AUC": auc, "F1": f1, "Model": model_name
        }

    def _evaluate_model(self, model_name):
        """
        Train/evaluate the model or approach specified by action, 
        compute performance metrics for the reward.
        """
        df = self.data
        X = df.drop(self.target_column, axis=1)
        y = df[self.target_column]
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        # Handle text
        X_train_text, X_val_text = np.zeros((len(X_train),0)), np.zeros((len(X_val),0))
        if "text" in X_train.columns:
            X_train_text = self.vectorizer.transform(X_train["text"]).toarray()
            X_val_text   = self.vectorizer.transform(X_val["text"]).toarray()

        X_train_num = X_train.select_dtypes(include=np.number)
        X_val_num   = X_val.select_dtypes(include=np.number)

        if len(X_train_num.columns) > 0:
            X_train_combined = np.concatenate([X_train_num, X_train_text], axis=1)
            X_val_combined   = np.concatenate([X_val_num, X_val_text], axis=1)
        else:
            X_train_combined, X_val_combined = X_train_text, X_val_text

        if len(np.unique(y_val)) != 2:
            # Fallback if not binary
            auc, f1_score_ = 0.5, 0.5
        else:
            # Choose model
            if model_name == "XGBoost":
                model = XGBClassifier()
                model.fit(X_train_combined, y_train)
                y_prob = model.predict_proba(X_val_combined)[:,1]
            elif model_name == "LightGBM":
                model = LGBMClassifier()
                model.fit(X_train_combined, y_train)
                y_prob = model.predict_proba(X_val_combined)[:,1]
            elif model_name == "DNN":
                # Torch-based training
                import torch
                import torch.optim as optim
                import torch.nn as nn

                X_train_t = torch.tensor(X_train_combined, dtype=torch.float32).to(self.device)
                X_val_t   = torch.tensor(X_val_combined, dtype=torch.float32).to(self.device)
                y_train_t = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1).to(self.device)

                model = DNNModel(X_train_t.shape[1]).to(self.device)
                optimizer = optim.Adam(model.parameters(), lr=1e-3)
                loss_fn = nn.BCELoss()

                for epoch in range(10):
                    model.train()
                    optimizer.zero_grad()
                    pred = model(X_train_t)
                    loss = loss_fn(pred, y_train_t)
                    loss.backward()
                    optimizer.step()

                with torch.no_grad():
                    y_prob_t = model(X_val_t)
                y_prob = y_prob_t.cpu().numpy().flatten()
            else:
                y_prob = np.zeros(len(X_val))  # default

            auc  = calc_auc_score(y_val, y_prob)
            f1_score_ = calc_f1_score(y_val, y_prob)

        # Add complexity penalty for DNN
        complexity_penalty = 0.05 if model_name == "DNN" else 0
        reward = auc + f1_score_ - complexity_penalty
        return reward, auc, f1_score_
