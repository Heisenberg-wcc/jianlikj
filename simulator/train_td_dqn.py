"""
TD Q-Learning + Target Network + Double DQN 自博弈训练器 (队伍网络版)

核心升级:
  1. 按队伍拆分网络: Team 0 (玩家0,2) 和 Team 1 (玩家1,3) 各自拥有独立的Q网络
  2. 解决对抗游戏中目标函数冲突的问题
  3. 数据结构: (state, action, reward, next_state, next_legal_action_encs, done, player_idx)
  4. 训练目标: TD target = r + gamma * Q_target(s', a_best)  (Double DQN)
  5. 目标网络: 每 TARGET_SYNC 步硬同步一次, 提供稳定的目标值
  6. 即时奖励: 每一步动作执行后立即计算, 不再等整局结束回填

训练循环: 执行动作 → 计算即时奖励 → 编码 next_state → 存入缓冲区 → 按队伍采样训练
"""

import os
import sys
import copy
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, namedtuple
from typing import List, Optional, Dict, Tuple

# ── 进度条工具 ──
def print_progress_bar(episode, total, win_rate, loss, epsilon, steps, buffer_size, start_time):
    """打印实时进度条"""
    percent = episode / total
    bar_len = 30
    filled = int(bar_len * percent)
    bar = '█' * filled + '░' * (bar_len - filled)
    elapsed = time.time() - start_time
    eps_per_sec = episode / elapsed if elapsed > 0 else 0
    eta = (total - episode) / eps_per_sec if eps_per_sec > 0 else 0
    
    print(f'\r[{bar}] {episode}/{total} ({percent:.1%}) | '
          f'Win:{win_rate:.1%} Loss:{loss:.4f} ε:{epsilon:.3f} | '
          f'Steps:{steps:.0f} Buf:{buffer_size} | '
          f'Speed:{eps_per_sec:.1f}eps/s ETA:{eta/60:.1f}min', end='', flush=True)

# ── 路径设置 ──
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from game_engine import GameEngine
from action_space import ActionSpace
from rules import RulesEngine, Hand, HandType
from card import Card
from config import Config
from q_net.q_network import encode_state, encode_action, STATE_DIM, ACTION_DIM
from q_net.team_networks import TeamQNetwork, get_player_team

# ────────────────────────────────────────────
# TD 经验样本 & 回放缓冲区
# ────────────────────────────────────────────
TDSample = namedtuple('TDSample', [
    'state',                # np.ndarray (STATE_DIM,)
    'action',               # np.ndarray (ACTION_DIM,)
    'reward',               # float  即时奖励
    'next_state',           # np.ndarray (STATE_DIM,)
    'next_action_encs',     # List[np.ndarray]  下一状态合法动作编码
    'done',                 # bool  是否结束
    'player_idx',           # int  执行动作的玩家 (用于选择对应队伍的网络)
])


class TDReplayBuffer:
    """TD 经验回放缓冲区 — 存储完整的 (s, a, r, s', legal_a', done) 转移"""

    def __init__(self, max_size: int = 200_000):
        self.buffer = deque(maxlen=max_size)

    def push(self, state, action_enc, reward, next_state, next_action_encs, done, player_idx):
        self.buffer.append(TDSample(
            state=state.astype(np.float32),
            action=action_enc.astype(np.float32),
            reward=np.float32(reward),
            next_state=next_state.astype(np.float32),
            next_action_encs=[ae.astype(np.float32) for ae in next_action_encs],
            done=bool(done),
            player_idx=player_idx,
        ))

    def sample(self, batch_size: int):
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))

    def __len__(self):
        return len(self.buffer)


# ────────────────────────────────────────────
# TD DQN 训练器
# ────────────────────────────────────────────
class TDDQNTrainer:
    """TD Q-Learning + Target Network + Double DQN 自博弈训练器"""

    def __init__(self):
        # ── 超参数 (从 Config 读取) ──
        self.gamma          = Config.TD_GAMMA
        self.lr             = Config.TD_LR
        self.batch_size     = Config.TD_BATCH_SIZE
        self.buffer_max     = Config.TD_BUFFER_MAX
        self.epsilon        = Config.TD_EPSILON_START
        self.epsilon_min    = Config.TD_EPSILON_MIN
        self.epsilon_decay  = Config.TD_EPSILON_DECAY
        self.target_sync    = Config.TD_TARGET_SYNC
        self.grad_clip      = Config.TD_GRAD_CLIP
        self.train_per_ep   = Config.TD_TRAIN_PER_EP
        self.save_interval  = Config.TD_SAVE_INTERVAL
        self.save_dir       = Config.TD_SAVE_DIR

        # ── 网络 (按队伍拆分) ──
        self.q_net      = TeamQNetwork()
        self.target_net = TeamQNetwork()
        self._sync_target()                        # 初始化时同步

        self.optimizer  = optim.Adam(self.q_net.get_all_parameters(), lr=self.lr)
        self.loss_fn    = nn.MSELoss()
        self.action_sp  = ActionSpace()
        self.rules      = RulesEngine()
        self.replay     = TDReplayBuffer(self.buffer_max)

        self.total_train_steps = 0                 # 累计梯度更新步数
        os.makedirs(self.save_dir, exist_ok=True)

    # ──────────────────────────────────────────
    # 目标网络同步
    # ──────────────────────────────────────────
    def _sync_target(self):
        """硬同步: 将 Q 网络参数完整复制到目标网络"""
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

    # ──────────────────────────────────────────
    # 动作选择
    # ──────────────────────────────────────────
    def _get_last_hand(self, last_played_cards: list) -> Optional[Hand]:
        if not last_played_cards:
            return None
        ht = self.rules.detect_hand_type(last_played_cards)
        return Hand(last_played_cards, ht) if ht != HandType.INVALID else None

    def _encode_state_full(self, game: GameEngine, player_idx: int, last_played_cards: list) -> np.ndarray:
        """编码完整状态，包含所有追踪信息"""
        gs = game.get_game_state()
        return encode_state(
            game, player_idx, last_played_cards,
            played_cards_history=game.get_played_cards_history(),
            current_round_cards=gs.current_round_cards,
            last_play_player=gs.last_played_player if gs.last_played_player is not None else -1,
            pass_counts=game.get_pass_counts(),
            step=game.get_total_step()
        )

    def select_action(
        self,
        game: GameEngine,
        player_idx: int,
        last_played_cards: list,
        explore: bool = True,
    ) -> Tuple[Optional[list], np.ndarray, float]:
        """
        选择动作 (epsilon-greedy)

        Returns:
            action, action_enc, q_value
        """
        player       = game.get_player(player_idx)
        last_hand    = self._get_last_hand(last_played_cards)
        legal_actions= self.action_sp.get_all_actions(player.hand, last_hand)

        can_pass     = len(last_played_cards) > 0

        candidates   = []
        if can_pass:
            candidates.append((None, encode_action(None)))
        for a in legal_actions:
            candidates.append((a, encode_action(a)))

        if not candidates:
            return None, encode_action(None), 0.0

        # Epsilon-greedy
        if explore and random.random() < self.epsilon:
            chosen = random.choice(candidates)
            return chosen[0], chosen[1], 0.0

        # Exploit
        state_vec = self._encode_state_full(game, player_idx, last_played_cards)
        actions   = [c[0] for c in candidates]
        encs      = [c[1] for c in candidates]
        best_a, best_q, _ = self.q_net.select_best(state_vec, actions, encs, player_idx)
        best_enc = encode_action(best_a)
        return best_a, best_enc, best_q

    def _get_legal_action_encs(
        self,
        game: GameEngine,
        player_idx: int,
        last_played_cards: list,
    ) -> List[np.ndarray]:
        """获取当前合法动作的编码列表 (含 pass)"""
        player    = game.get_player(player_idx)
        last_hand = self._get_last_hand(last_played_cards)
        legal     = self.action_sp.get_all_actions(player.hand, last_hand)
        can_pass  = len(last_played_cards) > 0

        encs = []
        if can_pass:
            encs.append(encode_action(None))
        for a in legal:
            encs.append(encode_action(a))
        if not encs:
            encs.append(encode_action(None))
        return encs

    # ──────────────────────────────────────────
    # 即时奖励计算
    # ──────────────────────────────────────────
    def _compute_step_reward(
        self,
        game: GameEngine,
        player_idx: int,
        action: Optional[list],
        action_success: bool,
        round_score_before: list,
        game_over_before: bool,
    ) -> float:
        """
        计算单步即时奖励

        奖励设计:
          - 赢得轮次分值: +分值/200
          - 丢失轮次分值: -分值/200
          - 出完牌奖励:   +0.3 ~ +0.5 (按名次递减)
          - 对局最终胜负: +1.0 / -1.0
          - 非法出牌惩罚: -0.05
        """
        reward = 0.0
        gs = game.get_game_state()

        # 非法出牌惩罚
        if action is not None and not action_success:
            reward -= 0.05
            return reward

        # 轮次分值变化
        new_scores = gs.team_scores
        my_team = 0 if player_idx in [0, 2] else 1
        opp_team = 1 - my_team

        my_delta  = new_scores[my_team]  - round_score_before[my_team]
        opp_delta = new_scores[opp_team] - round_score_before[opp_team]

        if my_delta > 0:
            reward += my_delta / 200.0
        if opp_delta > 0:
            reward -= opp_delta / 200.0

        # 出完牌奖励
        player = game.get_player(player_idx)
        if player.is_out() and player.position is not None:
            finish_bonus = {1: 0.5, 2: 0.3, 3: 0.1, 4: 0.0}
            reward += finish_bonus.get(player.position, 0.0)

        # 对局最终胜负
        if gs.game_over and not game_over_before:
            scores = gs.team_scores
            if scores[my_team] > scores[opp_team]:
                reward += 1.0
            elif scores[my_team] < scores[opp_team]:
                reward -= 1.0

        return reward

    # ──────────────────────────────────────────
    # 轮次状态管理
    # ──────────────────────────────────────────
    def _update_round_state(
        self,
        game: GameEngine,
        last_played_cards: list,
        last_play_player: int,
        passed_players: set,
    ) -> Tuple[list, int, set]:
        if last_play_player < 0:
            return last_played_cards, last_play_player, passed_players

        active_others = {
            i for i in range(4)
            if i != last_play_player and not game.get_player(i).is_out()
        }

        if active_others and active_others.issubset(passed_players):
            return [], -1, set()

        return last_played_cards, last_play_player, passed_players

    # ──────────────────────────────────────────
    # 对局执行 (逐步收集 TD 样本)
    # ──────────────────────────────────────────
    def play_episode(self, render: bool = False) -> dict:
        """
        运行完整一局, 逐步收集 TD 样本并立即存入缓冲区。

        Returns:
            info: 统计信息 dict
        """
        game = GameEngine()
        game.initialize()

        last_played_cards: list = []
        last_play_player:  int  = -1
        passed_players:    set  = set()

        max_steps = 800
        step      = 0
        sample_count = 0

        while not game.get_game_state().game_over and step < max_steps:
            step += 1
            current_idx = game.get_game_state().current_player_idx
            player      = game.get_player(current_idx)

            # 已出完牌的玩家自动 pass
            if player.is_out():
                game.pass_round(current_idx)
                if last_play_player >= 0:
                    passed_players.add(current_idx)
                    last_played_cards, last_play_player, passed_players = \
                        self._update_round_state(game, last_played_cards,
                                                 last_play_player, passed_players)
                continue

            # ── 编码当前状态 ──
            state_vec = self._encode_state_full(game, current_idx, last_played_cards)

            # ── 记录分数快照 (用于奖励计算) ──
            score_before   = list(game.get_game_state().team_scores)
            game_over_before = game.get_game_state().game_over

            # ── 选择动作 ──
            action, action_enc, q_val = self.select_action(
                game, current_idx, last_played_cards, explore=True
            )

            # ── 执行动作 ──
            action_success = True
            if action is not None:
                success = game.play_card(current_idx, action)
                if not success:
                    action_success = False
                    game.pass_round(current_idx)
                    action     = None
                    action_enc = encode_action(None)
                    if last_play_player >= 0:
                        passed_players.add(current_idx)
                else:
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
                game.pass_round(current_idx)
                if last_play_player >= 0:
                    passed_players.add(current_idx)

                if render:
                    try:
                        print(f"  Step {step:3d} | P{current_idx+1} PASS"
                              f"  Q={q_val:.3f}  剩余:{len(player.hand)}张")
                    except UnicodeEncodeError:
                        print(f"  Step {step:3d} | P{current_idx+1} PASS  Q={q_val:.3f}")

            # ── 轮次重置检测 ──
            last_played_cards, last_play_player, passed_players = \
                self._update_round_state(game, last_played_cards,
                                         last_play_player, passed_players)

            # ── 计算即时奖励 ──
            reward = self._compute_step_reward(
                game, current_idx, action, action_success,
                score_before, game_over_before,
            )

            # ── 编码 next_state (从当前玩家视角) ──
            done = game.get_game_state().game_over
            next_state_vec = self._encode_state_full(game, current_idx, last_played_cards)

            # ── 获取 next_state 的合法动作编码 ──
            if done or player.is_out():
                next_action_encs = [encode_action(None)]
            else:
                next_action_encs = self._get_legal_action_encs(
                    game, current_idx, last_played_cards
                )

            # ── 存入缓冲区 ──
            self.replay.push(
                state_vec, action_enc, reward,
                next_state_vec, next_action_encs, done,
                current_idx,  # 记录玩家索引，用于选择对应队伍的网络
            )
            sample_count += 1

        # ── 统计 ──
        gs = game.get_game_state()
        scores = gs.team_scores
        info = {
            'steps'       : step,
            'samples'     : sample_count,
            'team1_win'   : scores[0] > scores[1],
            'finished'    : game.finished_order,
            'team_scores' : scores,
        }
        return info

    # ──────────────────────────────────────────
    # 训练步 (Double DQN)
    # ──────────────────────────────────────────
    def train_step(self) -> Optional[float]:
        """
        从回放缓冲区采样一个 batch, 用 Double DQN TD 目标更新网络。

        Double DQN:
          a_best = argmax_a' Q_online(s', a')
          target = r + gamma * Q_target(s', a_best)   (若 done 则 target = r)
        """
        if len(self.replay) < self.batch_size:
            return None

        batch = self.replay.sample(self.batch_size)

        states  = torch.FloatTensor(np.array([s.state  for s in batch]))
        actions = torch.FloatTensor(np.array([s.action for s in batch]))
        rewards = torch.FloatTensor(np.array([s.reward for s in batch]))
        dones   = torch.FloatTensor(np.array([1.0 if s.done else 0.0 for s in batch]))

        # ── 计算 Double DQN 目标 ──
        next_max_qs = np.zeros(len(batch), dtype=np.float32)
        for i, s in enumerate(batch):
            if s.done or len(s.next_action_encs) == 0:
                next_max_qs[i] = 0.0
            else:
                # Online 网络选动作 (使用对应队伍的网络)
                best_idx = self.q_net.select_best_action_idx(
                    s.next_state, s.next_action_encs, s.player_idx
                )
                if best_idx < 0:
                    next_max_qs[i] = 0.0
                else:
                    # Target 网络评估价值 (使用对应队伍的网络)
                    next_max_qs[i] = self.target_net.compute_q_for_action(
                        s.next_state, s.next_action_encs[best_idx], s.player_idx
                    )

        next_max_qs_t = torch.FloatTensor(next_max_qs)
        targets = rewards + self.gamma * next_max_qs_t * (1.0 - dones)

        # ── 前向 + 反向 ──
        self.q_net.train()
        # 按玩家索引分别计算Q值 (每个样本使用对应队伍的网络)
        preds_list = []
        for i, s in enumerate(batch):
            pred = self.q_net(states[i:i+1], actions[i:i+1], s.player_idx)
            preds_list.append(pred)
        preds = torch.cat(preds_list)
        loss  = self.loss_fn(preds, targets)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.get_all_parameters(), self.grad_clip)
        self.optimizer.step()

        self.total_train_steps += 1

        # ── 目标网络同步 ──
        if self.total_train_steps % self.target_sync == 0:
            self._sync_target()

        return loss.item()

    # ──────────────────────────────────────────
    # 主训练循环
    # ──────────────────────────────────────────
    def train(self, num_episodes: int = 10_000):
        win_hist     = deque(maxlen=200)
        loss_hist    = deque(maxlen=200)
        steps_hist   = deque(maxlen=200)
        samples_hist = deque(maxlen=200)

        print("=" * 70)
        print("TD Q-Learning + Target Network + Double DQN 自博弈训练")
        print("  [队伍网络版] Team 0 (玩家0,2) 和 Team 1 (玩家1,3) 各自独立网络")
        print(f"  网络维度: STATE={STATE_DIM}, ACTION={ACTION_DIM}")
        print(f"  总局数: {num_episodes}, Batch={self.batch_size}, 每局训练{self.train_per_ep}次")
        print(f"  Gamma={self.gamma}, TargetSync={self.target_sync}, GradClip={self.grad_clip}")
        print("=" * 70)
        start_time = time.time()
        last_print_time = start_time

        for ep in range(1, num_episodes + 1):
            # ── 对局 (逐步收集 TD 样本) ──
            info = self.play_episode(render=False)

            # ── 梯度更新 ──
            ep_losses = []
            for _ in range(self.train_per_ep):
                loss = self.train_step()
                if loss is not None:
                    ep_losses.append(loss)

            # ── 统计 ──
            win_hist.append(1 if info['team1_win'] else 0)
            steps_hist.append(info['steps'])
            samples_hist.append(info['samples'])
            if ep_losses:
                loss_hist.append(np.mean(ep_losses))

            # ── Epsilon 衰减 ──
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

            # ── 实时进度条 ──
            win_rate = np.mean(win_hist) if win_hist else 0.0
            avg_steps = np.mean(steps_hist) if steps_hist else 0.0
            avg_loss = np.mean(loss_hist) if loss_hist else 0.0
            buf_size = len(self.replay)
            
            # 每1局都打印简要信息
            team_scores = info['team_scores']
            elapsed = time.time() - start_time
            eps_per_sec = ep / elapsed if elapsed > 0 else 0
            print(f"[Ep {ep:>6}/{num_episodes}] Win={win_rate:.1%} Loss={avg_loss:.4f} ε={self.epsilon:.3f} | "
                  f"Steps={info['steps']:.0f} Score=[{team_scores[0]:>3}:{team_scores[1]:>3}] | "
                  f"Buffer={buf_size:>6} | {eps_per_sec:.1f}eps/s")

            # ── 保存检查点 ──
            if ep % self.save_interval == 0:
                path = os.path.join(self.save_dir, f"q_net_ep{ep}.pth")
                self.q_net.save(path)
                print(f"\n  [Save] 检查点已保存: {path}")

        # ── 最终统计 ──
        print("\n" + "=" * 70)
        final_win_rate = np.mean(win_hist) if win_hist else 0.0
        final_loss = np.mean(loss_hist) if loss_hist else 0.0
        total_time = time.time() - start_time
        print(f"[Done] 训练完成!")
        print(f"       总局数: {num_episodes} | 总时间: {total_time/60:.1f}分钟 | 速度: {num_episodes/total_time:.1f}局/秒")
        print(f"       最终胜率: {final_win_rate:.1%} | 最终Loss: {final_loss:.4f}")
        print(f"       总训练步数: {self.total_train_steps} | 经验缓冲区: {len(self.replay)}")

        # ── 最终保存 ──
        final_path = os.path.join(self.save_dir, "q_net_final.pth")
        self.q_net.save(final_path)
        print(f"[Save] 模型保存至: {final_path}")
        print("=" * 70)

    def load(self, path: str):
        self.q_net.load(path)
        self._sync_target()
        print(f"[Load] {path} (已同步目标网络)")


# ────────────────────────────────────────────
# 入口
# ────────────────────────────────────────────
if __name__ == "__main__":
    trainer = TDDQNTrainer()
    trainer.train(num_episodes=10_000)
