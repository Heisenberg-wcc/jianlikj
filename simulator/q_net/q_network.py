"""
Q-Network: 增强版 Dueling DQN

设计升级:
  - 状态向量: 42维 → 95维 (新增记牌器/得分/控牌者/出局等信息)
  - 动作向量: 16维 → 24维 (新增牌型oneshot/张数/分值)
  - 网络结构: Dueling DQN, V(s) + A(s,a)
  - 容量扩大: 256-256-128-64
  - 新增Dropout防止过拟合

状态向量 (95维):
  [ 我的手牌(16) | 上家出的牌(16) | 已出牌记录(16) | 本轮场上牌(16) |
    位置(4) | 剩余牌数(4) | 队伍得分(2) | 本轮分值(1) |
    控牌者(4) | 轮首标记(1) | 队伍(1) | 已出局(4) |
    出局顺序(4) | 队伍排名优势(1) | PASS计数(4) | 游戏进程(1) ]

动作向量 (24维):
  [ 牌编码(16) | 牌型oneshot(6) | 出牌张数(1) | 分值牌总分(1) ]
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
STATE_DIM    = 95       # 状态向量维度 (增强版)
ACTION_DIM   = 24       # 动作向量维度 (增强版)
SUIT_ORDER   = ['spade', 'heart', 'club', 'diamond']

# 牌型索引映射 (one-hot 6维)
_HAND_TYPE_IDX = {
    'single': 0, 'pair': 1, 'consecutive_pairs': 2,
    'consecutive_threes': 3, 'bomb': 4, '510K': 5,
}


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
    """
    vec = np.zeros(CARD_DIM, dtype=np.float32)
    for c in cards:
        idx = rank_to_idx(c.rank)
        vec[idx] += 1.0
    for i in range(13):
        vec[i] /= 8.0
    for i in range(13, 15):
        vec[i] /= 2.0
    vec[15] = 1.0 if _check_pure_510k(cards) else 0.0
    return vec


def encode_state(game, player_idx: int, last_played_cards: list,
                 played_cards_history: list = None,
                 current_round_cards: list = None,
                 last_play_player: int = -1,
                 pass_counts: list = None,
                 step: int = 0) -> np.ndarray:
    """
    编码当前状态 (95维), 从 player_idx 的视角

    新增参数均有默认值, 兼容旧代码调用。
    """
    player = game.get_player(player_idx)
    gs     = game.get_game_state()

    # 1. 我的手牌 (16维)
    hand_vec = encode_cards(player.hand)

    # 2. 上家出的牌 (16维)
    last_vec = encode_cards(last_played_cards)

    # 3. 已出牌记录 / 记牌器 (16维)
    if played_cards_history:
        played_vec = encode_cards(played_cards_history)
    else:
        played_vec = np.zeros(CARD_DIM, dtype=np.float32)

    # 4. 本轮场上牌 (16维)
    if current_round_cards:
        round_cards_vec = encode_cards(current_round_cards)
    else:
        round_cards_vec = encode_cards(gs.current_round_cards)

    # 5. 我的位置 one-hot (4维)
    pos_vec = np.zeros(4, dtype=np.float32)
    pos_vec[player_idx] = 1.0

    # 6. 各玩家剩余牌数 (4维, 归一化)
    remain_vec = np.array(
        [len(game.get_player(i).hand) / 27.0 for i in range(4)],
        dtype=np.float32
    )

    # 7. 双方队伍得分 (2维, 除以200归一化)
    scores_vec = np.array(
        [gs.team_scores[0] / 200.0, gs.team_scores[1] / 200.0],
        dtype=np.float32
    )

    # 8. 本轮场上分值 (1维)
    round_score = sum(c.get_score_value() for c in gs.current_round_cards) / 50.0
    round_score_vec = np.array([round_score], dtype=np.float32)

    # 9. 控牌者 one-hot (4维)
    ctrl_vec = np.zeros(4, dtype=np.float32)
    if last_play_player >= 0:
        ctrl_vec[last_play_player] = 1.0

    # 10. 是否为轮首 (1维)
    is_round_start = np.array([1.0 if not last_played_cards else 0.0], dtype=np.float32)

    # 11. 我的队伍 (1维)
    my_team = np.array([0.0 if player_idx in [0, 2] else 1.0], dtype=np.float32)

    # 12. 已出局玩家 (4维)
    out_vec = np.array(
        [1.0 if game.get_player(i).is_out() else 0.0 for i in range(4)],
        dtype=np.float32
    )

    # 13. 出局顺序 (4维)
    order_vec = np.zeros(4, dtype=np.float32)
    for i in range(4):
        p = game.get_player(i)
        if p.position is not None:
            order_vec[i] = p.position / 4.0

    # 14. 队伍排名优势 (1维)
    my_team_idx = 0 if player_idx in [0, 2] else 1
    my_team_players = [0, 2] if my_team_idx == 0 else [1, 3]
    opp_team_players = [1, 3] if my_team_idx == 0 else [0, 2]
    my_out = sum(1 for p in my_team_players if game.get_player(p).is_out())
    opp_out = sum(1 for p in opp_team_players if game.get_player(p).is_out())
    rank_adv = np.array([(my_out - opp_out) / 2.0], dtype=np.float32)

    # 15. 各玩家连续PASS次数 (4维)
    if pass_counts:
        pass_vec = np.array([min(p, 5) / 5.0 for p in pass_counts], dtype=np.float32)
    else:
        pass_vec = np.zeros(4, dtype=np.float32)

    # 16. 游戏进程 (1维)
    progress = np.array([min(step, 300) / 300.0], dtype=np.float32)

    return np.concatenate([
        hand_vec,        # 16
        last_vec,        # 16
        played_vec,      # 16
        round_cards_vec, # 16
        pos_vec,         # 4
        remain_vec,      # 4
        scores_vec,      # 2
        round_score_vec, # 1
        ctrl_vec,        # 4
        is_round_start,  # 1
        my_team,         # 1
        out_vec,         # 4
        order_vec,       # 4
        rank_adv,        # 1
        pass_vec,        # 4
        progress,        # 1
    ])  # total = 95


def encode_action(action: Optional[list]) -> np.ndarray:
    """
    将出牌动作编码为 24 维向量
    action=None 表示 pass, 返回全零向量
    结构: [rank编码(16) | 牌型oneshot(6) | 出牌张数(1) | 分值牌总分(1)]
    """
    if action is None:
        return np.zeros(ACTION_DIM, dtype=np.float32)

    # 1. 牌编码 (16维)
    card_vec = encode_cards(action)

    # 2. 牌型 one-hot (6维)
    from rules import RulesEngine
    _rules = RulesEngine()
    ht = _rules.detect_hand_type(action)
    type_vec = np.zeros(6, dtype=np.float32)
    idx = _HAND_TYPE_IDX.get(ht.value, -1)
    if idx >= 0:
        type_vec[idx] = 1.0

    # 3. 出牌张数 (1维, /8归一化)
    count_vec = np.array([len(action) / 8.0], dtype=np.float32)

    # 4. 分值牌总分 (1维, /50归一化)
    score_vec = np.array(
        [sum(c.get_score_value() for c in action) / 50.0],
        dtype=np.float32
    )

    return np.concatenate([card_vec, type_vec, count_vec, score_vec])


# ────────────────────────────────────────────
# Dueling Q 网络
# ────────────────────────────────────────────
class QNetwork(nn.Module):
    """
    Dueling DQN: (state, action) → Q value

    网络结构 (修复版 - 扩大容量，移除值域限制):
      state → 状态嵌入(256)   → 拼接 → FC512 → FC512 → FC256
      action → 动作嵌入(128)  ↑
                                      ├→ V(s) 分支 → FC128 → 1 (无Tanh)
                                      └→ A(s,a) 分支 → FC128 → 1 (无Tanh)
      Q(s,a) = V(s) + A(s,a) (无clamp)
    """

    def __init__(self, state_dim: int = STATE_DIM, action_dim: int = ACTION_DIM,
                 state_hidden: int = 256, action_hidden: int = 128):
        super().__init__()
        sh = state_hidden
        ah = action_hidden
        shared_in  = sh + ah
        shared_mid = sh * 2        # 256 for sh=128, 512 for sh=256
        shared_out = sh             # 128 for sh=128, 256 for sh=256
        head_in    = shared_out
        head_hid   = sh // 2       # 64  for sh=128, 128 for sh=256

        self.state_embed = nn.Sequential(
            nn.Linear(state_dim, sh),
            nn.LayerNorm(sh),
            nn.ReLU(),
        )
        self.action_embed = nn.Sequential(
            nn.Linear(action_dim, ah),
            nn.LayerNorm(ah),
            nn.ReLU(),
        )
        self.shared = nn.Sequential(
            nn.Linear(shared_in, shared_mid),
            nn.LayerNorm(shared_mid),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(shared_mid, shared_mid),
            nn.LayerNorm(shared_mid),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(shared_mid, shared_out),
            nn.ReLU(),
        )
        self.value_head = nn.Sequential(
            nn.Linear(head_in, head_hid),
            nn.ReLU(),
            nn.Linear(head_hid, 1),
        )
        self.advantage_head = nn.Sequential(
            nn.Linear(head_in, head_hid),
            nn.ReLU(),
            nn.Linear(head_hid, 1),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state  : (batch, STATE_DIM) 或 (STATE_DIM,)
            action : (batch, ACTION_DIM) 或 (ACTION_DIM,)
        Returns:
            q      : (batch,) 或 scalar
        """
        s_emb = self.state_embed(state)
        a_emb = self.action_embed(action)
        x = torch.cat([s_emb, a_emb], dim=-1)
        shared = self.shared(x)

        v = self.value_head(shared).squeeze(-1)      # V(s)
        a = self.advantage_head(shared).squeeze(-1)   # A(s,a)

        # Q(s,a) = V(s) + A(s,a), 单个动作时不需要减去 mean(A)
        q = v + a
        return q  # 移除clamp，让网络自由学习Q值范围

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

    @torch.no_grad()
    def compute_max_q(
        self,
        state_vec: np.ndarray,
        action_encs: List[np.ndarray],
    ) -> float:
        """
        计算给定状态下所有合法动作的最大 Q 值 (用于 TD 目标计算)。

        Args:
            state_vec  : 状态向量 (STATE_DIM,)
            action_encs: 合法动作编码列表
        Returns:
            最大 Q 值 (float)。若无合法动作返回 0.0。
        """
        if not action_encs:
            return 0.0
        self.eval()
        n = len(action_encs)
        s = torch.FloatTensor(state_vec).unsqueeze(0).expand(n, -1)
        a = torch.FloatTensor(np.stack(action_encs))
        qs = self.forward(s, a).cpu().numpy()
        return float(np.max(qs))

    @torch.no_grad()
    def compute_q_for_action(
        self,
        state_vec: np.ndarray,
        action_enc: np.ndarray,
    ) -> float:
        """
        计算给定 (state, action) 对的 Q 值。
        """
        self.eval()
        s = torch.FloatTensor(state_vec).unsqueeze(0)
        a = torch.FloatTensor(action_enc).unsqueeze(0)
        return self.forward(s, a).item()

    @torch.no_grad()
    def select_best_action_idx(
        self,
        state_vec: np.ndarray,
        action_encs: List[np.ndarray],
    ) -> int:
        """
        返回最优动作的索引 (用于 Double DQN 中 online 网络选动作)。
        若无动作返回 -1。
        """
        if not action_encs:
            return -1
        self.eval()
        n = len(action_encs)
        s = torch.FloatTensor(state_vec).unsqueeze(0).expand(n, -1)
        a = torch.FloatTensor(np.stack(action_encs))
        qs = self.forward(s, a).cpu().numpy()
        return int(np.argmax(qs))

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        torch.save({'q_net': self.state_dict()}, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        sd = ckpt['q_net']
        # 自动检测网络尺寸，如果与当前不符则重建对应大小
        sh = sd['state_embed.0.weight'].shape[0]   # 128 or 256
        ah = sd['action_embed.0.weight'].shape[0]  # 64  or 128
        if sh != self.state_embed[0].out_features or ah != self.action_embed[0].out_features:
            self.__init__(state_hidden=sh, action_hidden=ah)
        self.load_state_dict(sd)
