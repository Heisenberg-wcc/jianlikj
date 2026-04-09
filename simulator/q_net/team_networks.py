"""
Team Networks: 按队伍拆分的网络架构

核心设计:
  - Team 0 (玩家0, 2) 和 Team 1 (玩家1, 3) 各自拥有独立的网络
  - 解决对抗游戏中目标函数冲突的问题
  - 每个网络只从己方队伍的经验中学习

支持的架构:
  - TeamQNetwork: 用于TD-DQN训练
  - TeamPolicyNetwork: 用于策略梯度训练
  - TeamValueNetwork: 用于A-C训练的Critic
"""

import os
import numpy as np
import torch
import torch.nn as nn
from typing import List, Optional, Tuple, Dict

from .q_network import QNetwork, encode_state, encode_action, STATE_DIM, ACTION_DIM
from .policy_network import PolicyNetwork, ValueNetwork


def get_player_team(player_idx: int) -> int:
    """获取玩家所属队伍: 0 或 1"""
    return 0 if player_idx in [0, 2] else 1


# ────────────────────────────────────────────
# Team Q-Network (用于 DQN 训练)
# ────────────────────────────────────────────
class TeamQNetwork(nn.Module):
    """
    按队伍拆分的Q网络。

    Team 0 (玩家0, 2) 和 Team 1 (玩家1, 3) 各自拥有独立的Q网络，
    避免对抗训练中目标函数冲突的问题。
    """

    def __init__(self, state_dim: int = STATE_DIM, action_dim: int = ACTION_DIM):
        super().__init__()
        self.team0_net = QNetwork(state_dim, action_dim)
        self.team1_net = QNetwork(state_dim, action_dim)

    def get_net(self, player_idx: int) -> QNetwork:
        """根据玩家索引获取对应的网络"""
        return self.team0_net if get_player_team(player_idx) == 0 else self.team1_net

    def forward(self, state: torch.Tensor, action: torch.Tensor, player_idx: int) -> torch.Tensor:
        """前向传播，自动选择对应队伍的网络"""
        net = self.get_net(player_idx)
        return net(state, action)

    @torch.no_grad()
    def predict(self, state_vec: np.ndarray, action_vec: np.ndarray, player_idx: int) -> float:
        """单次推理"""
        net = self.get_net(player_idx)
        return net.predict(state_vec, action_vec)

    @torch.no_grad()
    def select_best(
        self,
        state_vec: np.ndarray,
        actions: list,
        action_encs: List[np.ndarray],
        player_idx: int,
    ) -> Tuple[Optional[list], float, int]:
        """选择最佳动作"""
        net = self.get_net(player_idx)
        return net.select_best(state_vec, actions, action_encs)

    @torch.no_grad()
    def compute_max_q(
        self,
        state_vec: np.ndarray,
        action_encs: List[np.ndarray],
        player_idx: int,
    ) -> float:
        """计算最大Q值"""
        net = self.get_net(player_idx)
        return net.compute_max_q(state_vec, action_encs)

    @torch.no_grad()
    def compute_q_for_action(
        self,
        state_vec: np.ndarray,
        action_enc: np.ndarray,
        player_idx: int,
    ) -> float:
        """计算特定动作的Q值"""
        net = self.get_net(player_idx)
        return net.compute_q_for_action(state_vec, action_enc)

    @torch.no_grad()
    def select_best_action_idx(
        self,
        state_vec: np.ndarray,
        action_encs: List[np.ndarray],
        player_idx: int,
    ) -> int:
        """选择最佳动作索引（用于Double DQN）"""
        net = self.get_net(player_idx)
        return net.select_best_action_idx(state_vec, action_encs)

    def get_all_parameters(self) -> List[torch.nn.Parameter]:
        """获取所有参数（用于优化器）"""
        return list(self.team0_net.parameters()) + list(self.team1_net.parameters())

    def save(self, path: str):
        """保存两个网络"""
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        torch.save({
            'team0': self.team0_net.state_dict(),
            'team1': self.team1_net.state_dict(),
        }, path)

    def load(self, path: str):
        """加载两个网络（自动适配旧版本网络尺寸）"""
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        for key, net in [('team0', self.team0_net), ('team1', self.team1_net)]:
            sd = ckpt[key]
            sh = sd['state_embed.0.weight'].shape[0]
            ah = sd['action_embed.0.weight'].shape[0]
            if sh != net.state_embed[0].out_features or ah != net.action_embed[0].out_features:
                net.__init__(state_hidden=sh, action_hidden=ah)
            net.load_state_dict(sd)


# ────────────────────────────────────────────
# Team Policy Network (用于策略梯度训练)
# ────────────────────────────────────────────
class TeamPolicyNetwork(nn.Module):
    """
    按队伍拆分的策略网络 (Actor)。
    """

    def __init__(self, state_dim: int = STATE_DIM, action_dim: int = ACTION_DIM):
        super().__init__()
        self.team0_net = PolicyNetwork(state_dim, action_dim)
        self.team1_net = PolicyNetwork(state_dim, action_dim)

    def get_net(self, player_idx: int) -> PolicyNetwork:
        """根据玩家索引获取对应的网络"""
        return self.team0_net if get_player_team(player_idx) == 0 else self.team1_net

    def forward_logits(
        self,
        state: torch.Tensor,
        actions: torch.Tensor,
        player_idx: int,
    ) -> torch.Tensor:
        """计算logits"""
        net = self.get_net(player_idx)
        return net.forward_logits(state, actions)

    def get_action_probs(
        self,
        state_vec: np.ndarray,
        action_encs: List[np.ndarray],
        player_idx: int,
    ) -> Tuple[np.ndarray, torch.Tensor]:
        """获取动作概率分布"""
        net = self.get_net(player_idx)
        return net.get_action_probs(state_vec, action_encs)

    @torch.no_grad()
    def select_action(
        self,
        state_vec: np.ndarray,
        action_encs: List[np.ndarray],
        player_idx: int,
        deterministic: bool = False,
    ) -> Tuple[int, float]:
        """选择动作"""
        net = self.get_net(player_idx)
        return net.select_action(state_vec, action_encs, deterministic)

    def get_all_parameters(self) -> List[torch.nn.Parameter]:
        """获取所有参数"""
        return list(self.team0_net.parameters()) + list(self.team1_net.parameters())

    def save(self, path: str):
        """保存两个网络"""
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        torch.save({
            'team0': self.team0_net.state_dict(),
            'team1': self.team1_net.state_dict(),
        }, path)

    def load(self, path: str):
        """加载两个网络"""
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        self.team0_net.load_state_dict(ckpt['team0'])
        self.team1_net.load_state_dict(ckpt['team1'])


# ────────────────────────────────────────────
# Team Value Network (用于A-C训练的Critic)
# ────────────────────────────────────────────
class TeamValueNetwork(nn.Module):
    """
    按队伍拆分的价值网络 (Critic)。
    """

    def __init__(self, state_dim: int = STATE_DIM):
        super().__init__()
        self.team0_net = ValueNetwork(state_dim)
        self.team1_net = ValueNetwork(state_dim)

    def get_net(self, player_idx: int) -> ValueNetwork:
        """根据玩家索引获取对应的网络"""
        return self.team0_net if get_player_team(player_idx) == 0 else self.team1_net

    def forward(self, state: torch.Tensor, player_idx: int) -> torch.Tensor:
        """前向传播"""
        net = self.get_net(player_idx)
        return net(state)

    @torch.no_grad()
    def predict(self, state_vec: np.ndarray, player_idx: int) -> float:
        """单次推理"""
        net = self.get_net(player_idx)
        return net.predict(state_vec)

    def get_all_parameters(self) -> List[torch.nn.Parameter]:
        """获取所有参数"""
        return list(self.team0_net.parameters()) + list(self.team1_net.parameters())

    def save(self, path: str):
        """保存两个网络"""
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        torch.save({
            'team0': self.team0_net.state_dict(),
            'team1': self.team1_net.state_dict(),
        }, path)

    def load(self, path: str):
        """加载两个网络"""
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        self.team0_net.load_state_dict(ckpt['team0'])
        self.team1_net.load_state_dict(ckpt['team1'])
