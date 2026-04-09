"""
策略网络 (PolicyNetwork) 和价值网络 (ValueNetwork)

阶段二: 策略梯度学习 (Actor-Critic)

策略网络 (Actor):
  - 输入: state (95维) + 合法动作集合 (变长, 每个 24维)
  - 对每个合法动作计算 score(s, a), 然后 masked softmax 得到概率分布
  - 输出: 每个合法动作的选择概率

价值网络 (Critic / Baseline):
  - 输入: state (95维)
  - 输出: V(s) 局面价值标量
  - 作用: 降低策略梯度方差
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple

from q_net.q_network import STATE_DIM, ACTION_DIM


# ────────────────────────────────────────────
# 策略网络 (Actor)
# ────────────────────────────────────────────
class PolicyNetwork(nn.Module):
    """
    策略网络: 对每个候选 (state, action) 对计算一个 logit,
    然后在合法动作集上做 softmax 得到概率分布。

    结构:
      state  → 状态嵌入(128)  ─┐
      action → 动作嵌入(64)   ─┤→ 拼接 → FC256 → FC128 → FC1 (logit)
    """

    def __init__(self, state_dim: int = STATE_DIM, action_dim: int = ACTION_DIM):
        super().__init__()

        self.state_embed = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
        )

        self.action_embed = nn.Sequential(
            nn.Linear(action_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
        )

        self.shared = nn.Sequential(
            nn.Linear(128 + 64, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
        )

        # 输出一个 logit 标量 (非概率, softmax 在外部做)
        self.logit_head = nn.Linear(128, 1)

    def forward_logits(
        self,
        state: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """
        计算每个 (state, action) 对的 logit。

        Args:
            state   : (N, STATE_DIM)   N 个状态-动作对的状态
            actions : (N, ACTION_DIM)  N 个动作编码
        Returns:
            logits  : (N,)  每个对的 logit
        """
        s_emb = self.state_embed(state)
        a_emb = self.action_embed(actions)
        x = torch.cat([s_emb, a_emb], dim=-1)
        h = self.shared(x)
        return self.logit_head(h).squeeze(-1)

    def get_action_probs(
        self,
        state_vec: np.ndarray,
        action_encs: List[np.ndarray],
    ) -> Tuple[np.ndarray, torch.Tensor]:
        """
        给定一个状态和合法动作列表, 返回 softmax 概率分布。

        Args:
            state_vec   : (STATE_DIM,) numpy
            action_encs : List[np.ndarray], 每个 (ACTION_DIM,)
        Returns:
            probs  : (N,) numpy  概率分布
            logits : (N,) tensor logits (用于训练时计算 log_prob)
        """
        n = len(action_encs)
        s = torch.FloatTensor(state_vec).unsqueeze(0).expand(n, -1)
        a = torch.FloatTensor(np.stack(action_encs))
        logits = self.forward_logits(s, a)
        probs  = F.softmax(logits, dim=0)
        return probs.detach().cpu().numpy(), logits

    @torch.no_grad()
    def select_action(
        self,
        state_vec: np.ndarray,
        action_encs: List[np.ndarray],
        deterministic: bool = False,
    ) -> Tuple[int, float]:
        """
        根据概率分布选择动作。

        Args:
            state_vec    : 状态向量
            action_encs  : 合法动作编码列表
            deterministic: True=选最大概率, False=按概率采样
        Returns:
            (action_idx, log_prob)
        """
        if not action_encs:
            return 0, 0.0

        self.eval()
        probs_np, logits = self.get_action_probs(state_vec, action_encs)

        if deterministic:
            idx = int(np.argmax(probs_np))
        else:
            idx = int(np.random.choice(len(probs_np), p=probs_np))

        log_prob = float(torch.log(F.softmax(logits, dim=0)[idx] + 1e-8))
        return idx, log_prob

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        torch.save({'policy_net': self.state_dict()}, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        self.load_state_dict(ckpt['policy_net'])


# ────────────────────────────────────────────
# 价值网络 (Critic / Baseline)
# ────────────────────────────────────────────
class ValueNetwork(nn.Module):
    """
    价值网络: V(s) — 评估局面价值, 作为策略梯度的 baseline。

    结构:
      state → FC256 → FC256 → FC128 → FC1 → Tanh
    """

    def __init__(self, state_dim: int = STATE_DIM):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Tanh(),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state : (batch, STATE_DIM) 或 (STATE_DIM,)
        Returns:
            value : (batch,) 或 scalar
        """
        return self.network(state).squeeze(-1)

    @torch.no_grad()
    def predict(self, state_vec: np.ndarray) -> float:
        """单次推理, 返回 V(s) float"""
        self.eval()
        s = torch.FloatTensor(state_vec).unsqueeze(0)
        return self.forward(s).item()

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        torch.save({'value_net': self.state_dict()}, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        self.load_state_dict(ckpt['value_net'])
