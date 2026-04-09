"""
Q-Network: 基于状态-动作联合表示的 Q 值回归网络

设计思路:
  - 输入: state_vec (42维) + action_vec (16维) = 58维
  - 输出: 单标量 Q 值 ∈ [-1, 1]，表示该动作在当前局面下的胜负期望
  - 无 action mask，由规则引擎提前过滤合法动作集合

卡牌编码 (16维, 基于rank计数, 花色无关):
  - ranks 3-15 (13种) → 每种的计数/最大可能数, 归一化到[0,1]
  - 小王 (rank=16) → index 13
  - 大王 (rank=17) → index 14
  - 纯色510K标记 → index 15 (0或1)

状态向量 (42维):
  [ 我的手牌计数(16) | 上家出的牌计数(16) | 我的位置one-hot(4) | 各玩家剩余牌数比(4) | 是否为轮首(1) | 我的队伍(1) ]

动作向量 (16维):
  出牌动作: 各rank计数归一化 + 纯色510K标记
  不要(pass): 全零向量
"""

import os
import numpy as np
import torch
import torch.nn as nn
from typing import List, Optional, Tuple

# ────────────────────────────────────────────
# 维度常量
# ────────────────────────────────────────────
CARD_DIM     = 16       # 牌编码维度: 13 rank计数 + 2 王牌 + 1 纯色510K标记
STATE_DIM    = 42       # 状态向量维度
ACTION_DIM   = CARD_DIM # 动作向量维度
SUIT_ORDER   = ['spade', 'heart', 'club', 'diamond']


# ────────────────────────────────────────────
# 编码工具函数
# ────────────────────────────────────────────
def rank_to_idx(rank: int) -> int:
    """将 rank 映射到 0-14 的索引 (花色无关)"""
    if rank == 16:
        return 13   # 小王
    if rank == 17:
        return 14   # 大王
    # ranks 3-15 → 0-12
    return rank - 3


def _check_pure_510k(cards) -> bool:
    """检查牌列表中是否包含纯色510K"""
    ranks = [c.rank for c in cards]
    if 5 not in ranks or 10 not in ranks or 13 not in ranks:
        return False
    for suit in SUIT_ORDER:
        has_5 = any(c.rank == 5 and c.suit == suit for c in cards)
        has_10 = any(c.rank == 10 and c.suit == suit for c in cards)
        has_k = any(c.rank == 13 and c.suit == suit for c in cards)
        if has_5 and has_10 and has_k:
            return True
    return False


def encode_cards(cards) -> np.ndarray:
    """
    将牌列表编码为 16 维向量 (基于rank计数, 花色无关)

    结构: [rank3计数, rank4计数, ..., rank15计数, 小王计数, 大王计数, 纯色510K标记]
    归一化: ranks 3-15 除以8 (2副牌×4花色), 王牌除以2
    cards: List[Card]
    """
    vec = np.zeros(CARD_DIM, dtype=np.float32)
    for c in cards:
        idx = rank_to_idx(c.rank)
        vec[idx] += 1.0

    # 归一化: ranks 3-15 最多8张, 王牌最多2张
    for i in range(13):   # ranks 3-15
        vec[i] /= 8.0
    for i in range(13, 15):  # 小王, 大王
        vec[i] /= 2.0

    # 纯色510K标记
    vec[15] = 1.0 if _check_pure_510k(cards) else 0.0

    return vec


def encode_state(game, player_idx: int, last_played_cards: list) -> np.ndarray:
    """
    编码当前状态 (42维), 从 player_idx 的视角

    Args:
        game           : GameEngine 实例
        player_idx     : 当前玩家索引 (0-3)
        last_played_cards: 上家最新出的牌 (由 Trainer 追踪, 非 game.state.current_round_cards)
    """
    player   = game.get_player(player_idx)
    gs       = game.get_game_state()

    # 1. 我的手牌 (16维)
    hand_vec = encode_cards(player.hand)

    # 2. 上家出的牌 (16维) — 需要压过的目标
    last_vec = encode_cards(last_played_cards)

    # 3. 我的位置 one-hot (4维)
    pos_vec  = np.zeros(4, dtype=np.float32)
    pos_vec[player_idx] = 1.0

    # 4. 各玩家剩余牌数 (4维, 归一化)
    remain_vec = np.array(
        [len(game.get_player(i).hand) / 27.0 for i in range(4)],
        dtype=np.float32
    )

    # 5. 是否为轮首(没有上家出牌) (1维)
    is_round_start = np.array([1.0 if not last_played_cards else 0.0], dtype=np.float32)

    # 6. 我的队伍 (1维: 0=Team1[0,2], 1=Team2[1,3])
    my_team = np.array([0.0 if player_idx in [0, 2] else 1.0], dtype=np.float32)

    return np.concatenate([hand_vec, last_vec, pos_vec, remain_vec, is_round_start, my_team])


def encode_action(action: Optional[list]) -> np.ndarray:
    """
    将出牌动作编码为 16 维向量
    action=None 表示 pass, 返回全零向量
    """
    if action is None:
        return np.zeros(CARD_DIM, dtype=np.float32)
    return encode_cards(action)


# ────────────────────────────────────────────
# Q 网络
# ────────────────────────────────────────────
class QNetwork(nn.Module):
    """
    深度 Q 网络: (state, action) → Q value

    网络结构:
      [state(42) | action(16)] → FC128 → ReLU → FC128 → ReLU → FC64 → ReLU → FC1 → Tanh
    """

    def __init__(self, state_dim: int = STATE_DIM, action_dim: int = ACTION_DIM):
        super().__init__()
        in_dim = state_dim + action_dim

        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh(),      # 输出 [-1, 1], +1=必赢, -1=必输
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state  : (batch, STATE_DIM) 或 (STATE_DIM,)
            action : (batch, ACTION_DIM) 或 (ACTION_DIM,)
        Returns:
            q      : (batch,) 或 scalar
        """
        x = torch.cat([state, action], dim=-1)
        return self.net(x).squeeze(-1)

    @torch.no_grad()
    def predict(self, state_vec: np.ndarray, action_vec: np.ndarray) -> float:
        """单次推理，返回 Q 值 float"""
        self.eval()
        s = torch.FloatTensor(state_vec).unsqueeze(0)
        a = torch.FloatTensor(action_vec).unsqueeze(0)
        return self.forward(s, a).item()

    @torch.no_grad()
    def select_best(
        self,
        state_vec: np.ndarray,
        actions: list,
        action_encs: List[np.ndarray],
    ) -> Tuple[Optional[list], float, int]:
        """
        在合法动作列表中选 Q 值最大的动作

        Returns:
            best_action, best_q, best_idx
        """
        if not actions:
            return None, -1.0, -1

        self.eval()
        s  = torch.FloatTensor(state_vec).unsqueeze(0).expand(len(actions), -1)
        a  = torch.FloatTensor(np.stack(action_encs))
        qs = self.forward(s, a).cpu().numpy()

        best_idx = int(np.argmax(qs))
        return actions[best_idx], float(qs[best_idx]), best_idx

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        torch.save({'q_net': self.state_dict()}, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        self.load_state_dict(ckpt['q_net'])
