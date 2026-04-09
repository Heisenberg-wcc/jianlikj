"""
Monte Carlo Q-Learning 自博弈训练器

核心思路:
  1. 生成合法动作集合 (规则引擎)
  2. 对每个合法动作计算 Q(state, action)
  3. epsilon-greedy 选择动作执行
  4. 完整对局结束后, 用最终胜负 (+1/-1) 监督更新网络
  5. 无价值函数 bootstrap, 纯 Monte Carlo 回报

训练循环: 生成合法动作 → 逐个算 Q → 选最优 → 完整对局 → 监督更新
"""

import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, namedtuple
from typing import List, Optional, Dict, Tuple

# ── 路径设置 ──
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from game_engine import GameEngine
from action_space import ActionSpace
from rules import RulesEngine, Hand, HandType
from card import Card
from q_net.q_network import (
    QNetwork, encode_state, encode_action,
    STATE_DIM, ACTION_DIM
)

# ────────────────────────────────────────────
# 训练样本 & 回放缓冲区
# ────────────────────────────────────────────
Sample = namedtuple('Sample', ['state', 'action', 'result'])


class ReplayBuffer:
    """经验回放缓冲区"""

    def __init__(self, max_size: int = 200_000):
        self.buffer = deque(maxlen=max_size)

    def push(self, state, action_enc, result):
        self.buffer.append(Sample(
            state=state.astype(np.float32),
            action=action_enc.astype(np.float32),
            result=np.float32(result)
        ))

    def sample(self, batch_size: int):
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))

    def __len__(self):
        return len(self.buffer)


# ────────────────────────────────────────────
# 训练器
# ────────────────────────────────────────────
class MCQTrainer:
    """Monte Carlo Q-Learning 自博弈训练器"""

    # ── 超参数 ──
    LR             = 3e-4
    BATCH_SIZE     = 512
    BUFFER_MAX     = 200_000
    EPSILON_START  = 1.0
    EPSILON_MIN    = 0.05
    EPSILON_DECAY  = 0.997       # 每局衰减
    PASS_EPSILON   = 0.15        # 轮首不能 pass; 有牌时 pass 概率上限
    SAVE_DIR       = "q_models"
    SAVE_INTERVAL  = 500         # 每N局保存一次

    def __init__(self):
        self.q_net      = QNetwork()
        self.optimizer  = optim.Adam(self.q_net.parameters(), lr=self.LR)
        self.loss_fn    = nn.MSELoss()
        self.action_sp  = ActionSpace()
        self.rules      = RulesEngine()
        self.replay     = ReplayBuffer(self.BUFFER_MAX)
        self.epsilon    = self.EPSILON_START
        self.total_steps = 0
        os.makedirs(self.SAVE_DIR, exist_ok=True)

    # ──────────────────────────────────────────
    # 动作选择
    # ──────────────────────────────────────────
    def _get_last_hand(self, last_played_cards: list) -> Optional[Hand]:
        """将 last_played_cards 转为 Hand 对象"""
        if not last_played_cards:
            return None
        ht = self.rules.detect_hand_type(last_played_cards)
        return Hand(last_played_cards, ht) if ht != HandType.INVALID else None

    def select_action(
        self,
        game: GameEngine,
        player_idx: int,
        last_played_cards: list,
        explore: bool = True,
    ) -> Tuple[Optional[list], np.ndarray, float]:
        """
        选择动作

        Returns:
            action        : 出牌列表 or None(pass)
            action_enc    : 54维编码
            q_value       : 预测 Q 值 (仅供日志)
        """
        player       = game.get_player(player_idx)
        last_hand    = self._get_last_hand(last_played_cards)
        legal_actions= self.action_sp.get_all_actions(player.hand, last_hand)

        # 是否可以 pass (轮首不能 pass)
        can_pass     = len(last_played_cards) > 0

        # ── 构建候选集 (含 pass) ──
        candidates   = []          # (action, action_enc)
        if can_pass:
            candidates.append((None, encode_action(None)))    # pass
        for a in legal_actions:
            candidates.append((a, encode_action(a)))

        # 若无候选 (不该出现), 强制 pass
        if not candidates:
            return None, encode_action(None), 0.0

        # ── Epsilon-greedy 探索 ──
        if explore and random.random() < self.epsilon:
            chosen = random.choice(candidates)
            return chosen[0], chosen[1], 0.0

        # ── Exploit: 批量计算 Q, 选最大 ──
        state_vec = encode_state(game, player_idx, last_played_cards)
        actions   = [c[0] for c in candidates]
        encs      = [c[1] for c in candidates]
        best_a, best_q, _ = self.q_net.select_best(state_vec, actions, encs)
        best_enc = encode_action(best_a)
        return best_a, best_enc, best_q

    # ──────────────────────────────────────────
    # 对局执行
    # ──────────────────────────────────────────
    def _update_round_state(
        self,
        game: GameEngine,
        last_played_cards: list,
        last_play_player: int,
        passed_players: set,
    ) -> Tuple[list, int, set]:
        """
        检测其他活跃玩家是否全部 pass，若是则重置轮次控制权。

        规则：出牌者是「当前轮控制者」，其余所有还没出完牌的人
        都 pass 了，则出牌权重置 (任意人可自由出牌)。
        """
        if last_play_player < 0:
            # 还没有人出过牌, 无需检测
            return last_played_cards, last_play_player, passed_players

        # 除控制者之外还在游戏中的玩家
        active_others = {
            i for i in range(4)
            if i != last_play_player and not game.get_player(i).is_out()
        }

        if active_others and active_others.issubset(passed_players):
            # 所有其他活跃玩家都已 pass → 新一轮, 出牌权自由
            return [], -1, set()

        return last_played_cards, last_play_player, passed_players

    def play_episode(self, render: bool = False) -> Tuple[List[Sample], dict]:
        """
        运行完整一局并收集 (state, action_enc, player_idx) 三元组

        Returns:
            samples : 回填结果后的 Sample 列表
            info    : 统计信息 dict
        """
        game = GameEngine()
        game.initialize()

        episode_data: List[Tuple[np.ndarray, np.ndarray, int]] = []

        # 独立追踪轮次状态
        last_played_cards: list = []   # 当前轮控制者出的牌
        last_play_player:  int  = -1   # 当前轮控制者 idx (-1=无)
        passed_players:    set  = set() # 本轮已 pass 的玩家集合

        max_steps = 800
        step      = 0

        while not game.get_game_state().game_over and step < max_steps:
            step += 1
            current_idx = game.get_game_state().current_player_idx
            player      = game.get_player(current_idx)

            # 已出完牌的玩家自动 pass（必须调用 pass_round 以推进轮次计数）
            if player.is_out():
                game.pass_round(current_idx)
                if last_play_player >= 0:
                    passed_players.add(current_idx)
                    last_played_cards, last_play_player, passed_players = \
                        self._update_round_state(game, last_played_cards,
                                                 last_play_player, passed_players)
                continue

            # 编码当前状态
            state_vec = encode_state(game, current_idx, last_played_cards)

            # 选择动作
            action, action_enc, q_val = self.select_action(
                game, current_idx, last_played_cards, explore=True
            )

            # 执行动作
            if action is not None:
                success = game.play_card(current_idx, action)
                if not success:
                    # 非法出牌 → 强制 pass
                    game.pass_round(current_idx)
                    action     = None
                    action_enc = encode_action(None)
                    if last_play_player >= 0:
                        passed_players.add(current_idx)
                else:
                    # 出牌成功 → 成为本轮控制者
                    last_played_cards = list(action)
                    last_play_player  = current_idx
                    passed_players    = set()

                    if render:
                        try:
                            cards_str = ', '.join(str(c) for c in action)
                            remaining = len(player.hand)
                            print(f"  Step {step:3d} | P{current_idx+1} 出牌: [{cards_str}]"
                                  f"  Q={q_val:.3f}  剩余:{remaining}张")
                        except UnicodeEncodeError:
                            print(f"  Step {step:3d} | P{current_idx+1} 出{len(action)}张牌"
                                  f"  Q={q_val:.3f}  剩余:{len(player.hand)}张")
            else:
                # Pass
                game.pass_round(current_idx)
                if last_play_player >= 0:
                    passed_players.add(current_idx)

                if render:
                    try:
                        print(f"  Step {step:3d} | P{current_idx+1} PASS"
                              f"  Q={q_val:.3f}  剩余:{len(player.hand)}张")
                    except UnicodeEncodeError:
                        print(f"  Step {step:3d} | P{current_idx+1} PASS  Q={q_val:.3f}")

            # 轮次重置检测
            last_played_cards, last_play_player, passed_players = \
                self._update_round_state(game, last_played_cards,
                                         last_play_player, passed_players)

            # 记录样本
            episode_data.append((state_vec, action_enc, current_idx))

        # 计算最终胜负结果
        results = self._get_results(game)

        # 用胜负结果标注所有样本
        samples = [
            Sample(sv, ae, results[pidx])
            for sv, ae, pidx in episode_data
        ]

        info = {
            'steps'       : step,
            'team1_win'   : results[0] > 0,
            'finished'    : game.finished_order,
            'team_scores' : game.get_game_state().team_scores,
        }
        return samples, info

    def _get_results(self, game: GameEngine) -> Dict[int, float]:
        """
        根据最终排名计算每个玩家的胜负奖励

        胜利: +1.0, 失败: -1.0
        头游/末游差距: 用排名加权
        """
        order  = game.finished_order
        scores = game.get_game_state().team_scores

        # 判断胜利队伍
        if scores[0] > scores[1]:
            winning_team = [0, 2]
        elif scores[1] > scores[0]:
            winning_team = [1, 3]
        else:
            # 平局根据排名
            team0_rank = sum(order.index(p) for p in [0, 2] if p in order)
            team1_rank = sum(order.index(p) for p in [1, 3] if p in order)
            winning_team = [0, 2] if team0_rank <= team1_rank else [1, 3]

        results = {}
        for i in range(4):
            results[i] = 1.0 if i in winning_team else -1.0

        # 加权: 排名越高奖励越大 (鼓励尽快出完)
        rank_bonus = {0: 0.3, 1: 0.1, 2: -0.1, 3: -0.3}
        for rank, pid in enumerate(order):
            base = results[pid]
            results[pid] = float(np.clip(base + rank_bonus.get(rank, 0.0), -1.0, 1.0))

        return results

    # ──────────────────────────────────────────
    # 训练步
    # ──────────────────────────────────────────
    def train_step(self) -> Optional[float]:
        """从回放缓冲区采样一个 batch 进行监督更新"""
        if len(self.replay) < self.BATCH_SIZE:
            return None

        batch   = self.replay.sample(self.BATCH_SIZE)
        states  = torch.FloatTensor(np.array([s.state  for s in batch]))
        actions = torch.FloatTensor(np.array([s.action for s in batch]))
        targets = torch.FloatTensor(np.array([s.result for s in batch]))

        self.q_net.train()
        preds = self.q_net(states, actions)
        loss  = self.loss_fn(preds, targets)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.0)
        self.optimizer.step()

        return loss.item()

    # ──────────────────────────────────────────
    # 主训练循环
    # ──────────────────────────────────────────
    def train(self, num_episodes: int = 10_000):
        """
        主训练入口

        每局结束后:
          1. 将所有 (s, a, result) 存入回放缓冲区
          2. 执行多次梯度更新
          3. 衰减 epsilon
        """
        win_hist  = deque(maxlen=200)
        loss_hist = deque(maxlen=200)

        print("=" * 60)
        print("Monte Carlo Q-Learning 自博弈训练")
        print(f"  网络维度: STATE={STATE_DIM}, ACTION={ACTION_DIM}")
        print(f"  总局数: {num_episodes}, Batch={self.BATCH_SIZE}")
        print("=" * 60)

        for ep in range(1, num_episodes + 1):
            # ── 对局 ──
            samples, info = self.play_episode(render=False)

            # ── 存入缓冲区 ──
            for s in samples:
                self.replay.push(s.state, s.action, s.result)

            # ── 梯度更新 (每局训练 4 次) ──
            ep_losses = []
            for _ in range(4):
                loss = self.train_step()
                if loss is not None:
                    ep_losses.append(loss)

            # ── 统计 ──
            win_hist.append(1 if info['team1_win'] else 0)
            if ep_losses:
                loss_hist.append(np.mean(ep_losses))

            # ── Epsilon 衰减 ──
            self.epsilon = max(self.EPSILON_MIN, self.epsilon * self.EPSILON_DECAY)

            # ── 打印进度 ──
            if ep % 100 == 0:
                win_rate  = np.mean(win_hist)
                avg_loss  = np.mean(loss_hist) if loss_hist else float('nan')
                buf_size  = len(self.replay)
                print(
                    f"[Ep {ep:>6}/{num_episodes}] "
                    f"Win={win_rate:.1%}  Loss={avg_loss:.4f}  "
                    f"ε={self.epsilon:.3f}  Buffer={buf_size:>6}"
                )

            # ── 保存检查点 ──
            if ep % self.SAVE_INTERVAL == 0:
                path = os.path.join(self.SAVE_DIR, f"q_net_ep{ep}.pth")
                self.q_net.save(path)
                print(f"  [Save] {path}")

        # ── 最终保存 ──
        final_path = os.path.join(self.SAVE_DIR, "q_net_final.pth")
        self.q_net.save(final_path)
        print(f"\n[Done] 训练完成, 模型保存至: {final_path}")

    def load(self, path: str):
        self.q_net.load(path)
        print(f"[Load] {path}")


# ────────────────────────────────────────────
# 入口
# ────────────────────────────────────────────
if __name__ == "__main__":
    trainer = MCQTrainer()
    trainer.train(num_episodes=10_000)
