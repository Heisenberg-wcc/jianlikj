"""
策略梯度 (Actor-Critic) 自博弈训练器

核心设计 (对标 DouZero 类策略学习):
  1. 策略网络 (Actor): 输出每个合法动作的选择概率
  2. 价值网络 (Critic): 输出 V(s) 作为 baseline
  3. 决策方式: 按概率分布采样 (自然探索, 无需 epsilon)
  4. 损失函数:
       policy_loss  = -log(pi(a|s)) * advantage
       value_loss   = MSE(V(s), G_t)
       entropy_bonus= -sum(pi * log(pi))
       total_loss   = policy_loss + c1*value_loss - c2*entropy_bonus

训练循环: 生成完整对局轨迹 → 计算折扣回报 → 策略梯度更新
"""

import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
from typing import List, Optional, Dict, Tuple

# ── 路径设置 ──
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from game_engine import GameEngine
from action_space import ActionSpace
from rules import RulesEngine, Hand, HandType
from card import Card
from config import Config
from q_net.q_network import encode_state, encode_action, STATE_DIM, ACTION_DIM
from q_net.policy_network import PolicyNetwork, ValueNetwork


# ────────────────────────────────────────────
# 轨迹步
# ────────────────────────────────────────────
class TrajectoryStep:
    """单步轨迹数据"""
    __slots__ = ['state', 'action_encs', 'chosen_idx', 'log_prob',
                 'reward', 'player_idx']

    def __init__(self, state, action_encs, chosen_idx, log_prob,
                 reward, player_idx):
        self.state       = state         # np.ndarray (STATE_DIM,)
        self.action_encs = action_encs   # List[np.ndarray]
        self.chosen_idx  = chosen_idx    # int
        self.log_prob    = log_prob      # float
        self.reward      = reward        # float (即时奖励, 后续会被替换为折扣回报)
        self.player_idx  = player_idx    # int


# ────────────────────────────────────────────
# 策略梯度训练器
# ────────────────────────────────────────────
class PolicyGradientTrainer:
    """Actor-Critic 策略梯度自博弈训练器"""

    def __init__(self):
        # ── 超参数 ──
        self.gamma        = Config.PG_GAMMA
        self.entropy_coef = Config.PG_ENTROPY_COEF
        self.value_coef   = Config.PG_VALUE_COEF
        self.grad_clip    = Config.PG_GRAD_CLIP
        self.save_dir     = Config.PG_SAVE_DIR
        self.save_interval= Config.PG_SAVE_INTERVAL

        # ── 网络 ──
        self.policy_net = PolicyNetwork()
        self.value_net  = ValueNetwork()

        self.policy_optimizer = optim.Adam(
            self.policy_net.parameters(), lr=Config.PG_LR_ACTOR
        )
        self.value_optimizer = optim.Adam(
            self.value_net.parameters(), lr=Config.PG_LR_CRITIC
        )

        self.action_sp = ActionSpace()
        self.rules     = RulesEngine()

        os.makedirs(self.save_dir, exist_ok=True)

    # ──────────────────────────────────────────
    # 辅助
    # ──────────────────────────────────────────
    def _get_last_hand(self, last_played_cards: list) -> Optional[Hand]:
        if not last_played_cards:
            return None
        ht = self.rules.detect_hand_type(last_played_cards)
        return Hand(last_played_cards, ht) if ht != HandType.INVALID else None

    def _get_legal_encs(
        self, game: GameEngine, player_idx: int, last_played_cards: list
    ) -> Tuple[List[Optional[list]], List[np.ndarray]]:
        """返回 (actions, action_encs), 含 pass"""
        player    = game.get_player(player_idx)
        last_hand = self._get_last_hand(last_played_cards)
        legal     = self.action_sp.get_all_actions(player.hand, last_hand)
        can_pass  = len(last_played_cards) > 0

        actions = []
        encs    = []
        if can_pass:
            actions.append(None)
            encs.append(encode_action(None))
        for a in legal:
            actions.append(a)
            encs.append(encode_action(a))
        if not actions:
            actions.append(None)
            encs.append(encode_action(None))
        return actions, encs

    # ──────────────────────────────────────────
    # 即时奖励 (复用阶段一设计)
    # ──────────────────────────────────────────
    def _compute_step_reward(
        self,
        game: GameEngine,
        player_idx: int,
        action: Optional[list],
        action_success: bool,
        score_before: list,
        game_over_before: bool,
    ) -> float:
        reward = 0.0
        gs = game.get_game_state()

        if action is not None and not action_success:
            reward -= 0.05
            return reward

        new_scores = gs.team_scores
        my_team  = 0 if player_idx in [0, 2] else 1
        opp_team = 1 - my_team

        my_delta  = new_scores[my_team]  - score_before[my_team]
        opp_delta = new_scores[opp_team] - score_before[opp_team]

        if my_delta > 0:
            reward += my_delta / 200.0
        if opp_delta > 0:
            reward -= opp_delta / 200.0

        player = game.get_player(player_idx)
        if player.is_out() and player.position is not None:
            finish_bonus = {1: 0.5, 2: 0.3, 3: 0.1, 4: 0.0}
            reward += finish_bonus.get(player.position, 0.0)

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
        self, game, last_played_cards, last_play_player, passed_players
    ):
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
    # 对局执行 (收集轨迹)
    # ──────────────────────────────────────────
    def play_episode(self, render: bool = False, deterministic: bool = False):
        """
        运行完整一局, 按策略网络概率采样选择动作。

        Returns:
            trajectories: Dict[int, List[TrajectoryStep]]  每个玩家的轨迹
            info: dict 统计信息
        """
        game = GameEngine()
        game.initialize()

        last_played_cards: list = []
        last_play_player:  int  = -1
        passed_players:    set  = set()

        # 每个玩家的轨迹
        trajectories: Dict[int, List[TrajectoryStep]] = {i: [] for i in range(4)}

        max_steps = 800
        step      = 0

        while not game.get_game_state().game_over and step < max_steps:
            step += 1
            current_idx = game.get_game_state().current_player_idx
            player      = game.get_player(current_idx)

            if player.is_out():
                game.pass_round(current_idx)
                if last_play_player >= 0:
                    passed_players.add(current_idx)
                    last_played_cards, last_play_player, passed_players = \
                        self._update_round_state(game, last_played_cards,
                                                 last_play_player, passed_players)
                continue

            # 编码状态
            state_vec = encode_state(game, current_idx, last_played_cards)

            # 获取合法动作
            actions, action_encs = self._get_legal_encs(
                game, current_idx, last_played_cards
            )

            # 策略网络选择动作
            self.policy_net.eval()
            chosen_idx, log_prob = self.policy_net.select_action(
                state_vec, action_encs, deterministic=deterministic
            )

            action = actions[chosen_idx]

            # 记录分数快照
            score_before     = list(game.get_game_state().team_scores)
            game_over_before = game.get_game_state().game_over

            # 执行动作
            action_success = True
            if action is not None:
                success = game.play_card(current_idx, action)
                if not success:
                    action_success = False
                    game.pass_round(current_idx)
                    action = None
                    if last_play_player >= 0:
                        passed_players.add(current_idx)
                else:
                    last_played_cards = list(action)
                    last_play_player  = current_idx
                    passed_players    = set()

                    if render:
                        try:
                            cards_str = ', '.join(str(c) for c in action)
                            print(f"  Step {step:3d} | P{current_idx+1} 出牌: [{cards_str}]"
                                  f"  剩余:{len(player.hand)}张")
                        except UnicodeEncodeError:
                            print(f"  Step {step:3d} | P{current_idx+1} 出牌"
                                  f"  剩余:{len(player.hand)}张")
            else:
                game.pass_round(current_idx)
                if last_play_player >= 0:
                    passed_players.add(current_idx)
                if render:
                    print(f"  Step {step:3d} | P{current_idx+1} PASS"
                          f"  剩余:{len(player.hand)}张")

            # 轮次重置
            last_played_cards, last_play_player, passed_players = \
                self._update_round_state(game, last_played_cards,
                                         last_play_player, passed_players)

            # 计算即时奖励
            reward = self._compute_step_reward(
                game, current_idx, action, action_success,
                score_before, game_over_before,
            )

            # 记录轨迹步
            trajectories[current_idx].append(TrajectoryStep(
                state=state_vec,
                action_encs=action_encs,
                chosen_idx=chosen_idx,
                log_prob=log_prob,
                reward=reward,
                player_idx=current_idx,
            ))

        gs = game.get_game_state()
        info = {
            'steps'       : step,
            'team1_win'   : gs.team_scores[0] > gs.team_scores[1],
            'finished'    : game.finished_order,
            'team_scores' : gs.team_scores,
        }
        return trajectories, info

    # ──────────────────────────────────────────
    # 计算折扣回报
    # ──────────────────────────────────────────
    def _compute_returns(self, trajectory: List[TrajectoryStep]) -> List[float]:
        """计算每步的折扣累计回报 G_t"""
        returns = []
        G = 0.0
        for step in reversed(trajectory):
            G = step.reward + self.gamma * G
            returns.insert(0, G)
        return returns

    # ──────────────────────────────────────────
    # 训练步
    # ──────────────────────────────────────────
    def train_on_episode(
        self, trajectories: Dict[int, List[TrajectoryStep]]
    ) -> Dict[str, float]:
        """
        用一局的轨迹数据训练 Actor 和 Critic。

        total_loss = policy_loss + c1*value_loss - c2*entropy_bonus
        """
        all_policy_losses = []
        all_value_losses  = []
        all_entropies     = []

        for pid in range(4):
            traj = trajectories[pid]
            if not traj:
                continue

            returns = self._compute_returns(traj)

            for t_step, G_t in zip(traj, returns):
                state_t = torch.FloatTensor(t_step.state)

                # ── Critic: V(s) ──
                self.value_net.train()
                v_pred = self.value_net(state_t.unsqueeze(0))
                value_target = torch.FloatTensor([G_t])
                value_loss = F.mse_loss(v_pred, value_target)
                all_value_losses.append(value_loss)

                # ── Advantage ──
                advantage = G_t - v_pred.item()

                # ── Actor: log_prob * advantage ──
                self.policy_net.train()
                n = len(t_step.action_encs)
                s = state_t.unsqueeze(0).expand(n, -1)
                a = torch.FloatTensor(np.stack(t_step.action_encs))
                logits = self.policy_net.forward_logits(s, a)
                probs  = F.softmax(logits, dim=0)
                log_probs = F.log_softmax(logits, dim=0)

                chosen_log_prob = log_probs[t_step.chosen_idx]
                policy_loss = -chosen_log_prob * advantage
                all_policy_losses.append(policy_loss)

                # ── Entropy bonus ──
                entropy = -(probs * log_probs).sum()
                all_entropies.append(entropy)

        if not all_policy_losses:
            return {'policy_loss': 0.0, 'value_loss': 0.0, 'entropy': 0.0}

        # ── 汇总损失 ──
        total_policy_loss = torch.stack(all_policy_losses).mean()
        total_value_loss  = torch.stack(all_value_losses).mean()
        total_entropy     = torch.stack(all_entropies).mean()

        total_loss = (
            total_policy_loss
            + self.value_coef * total_value_loss
            - self.entropy_coef * total_entropy
        )

        # ── 反向传播 (Actor + Critic) ──
        self.policy_optimizer.zero_grad()
        self.value_optimizer.zero_grad()
        total_loss.backward()

        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.grad_clip)
        torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), self.grad_clip)

        self.policy_optimizer.step()
        self.value_optimizer.step()

        return {
            'policy_loss': total_policy_loss.item(),
            'value_loss' : total_value_loss.item(),
            'entropy'    : total_entropy.item(),
        }

    # ──────────────────────────────────────────
    # 主训练循环
    # ──────────────────────────────────────────
    def train(self, num_episodes: int = 10_000):
        win_hist   = deque(maxlen=200)
        loss_hist  = deque(maxlen=200)

        print("=" * 60)
        print("策略梯度 (Actor-Critic) 自博弈训练")
        print(f"  网络维度: STATE={STATE_DIM}, ACTION={ACTION_DIM}")
        print(f"  总局数: {num_episodes}")
        print(f"  gamma={self.gamma}, entropy_coef={self.entropy_coef}")
        print("=" * 60)

        for ep in range(1, num_episodes + 1):
            # ── 对局 + 收集轨迹 ──
            trajectories, info = self.play_episode(render=False)

            # ── 训练 ──
            metrics = self.train_on_episode(trajectories)

            # ── 统计 ──
            win_hist.append(1 if info['team1_win'] else 0)
            loss_hist.append(metrics['policy_loss'])

            # ── 打印进度 ──
            if ep % 100 == 0:
                win_rate = np.mean(win_hist)
                avg_ploss = np.mean(loss_hist) if loss_hist else float('nan')
                print(
                    f"[Ep {ep:>6}/{num_episodes}] "
                    f"Win={win_rate:.1%}  "
                    f"PLoss={avg_ploss:.4f}  "
                    f"VLoss={metrics['value_loss']:.4f}  "
                    f"Entropy={metrics['entropy']:.4f}"
                )

            # ── 保存检查点 ──
            if ep % self.save_interval == 0:
                p_path = os.path.join(self.save_dir, f"policy_ep{ep}.pth")
                v_path = os.path.join(self.save_dir, f"value_ep{ep}.pth")
                self.policy_net.save(p_path)
                self.value_net.save(v_path)
                print(f"  [Save] {p_path}")

        # ── 最终保存 ──
        self.policy_net.save(os.path.join(self.save_dir, "policy_final.pth"))
        self.value_net.save(os.path.join(self.save_dir, "value_final.pth"))
        print(f"\n[Done] 训练完成, 模型保存至: {self.save_dir}/")

    def load(self, policy_path: str, value_path: str = None):
        self.policy_net.load(policy_path)
        print(f"[Load] policy: {policy_path}")
        if value_path and os.path.exists(value_path):
            self.value_net.load(value_path)
            print(f"[Load] value: {value_path}")

    # ──────────────────────────────────────────
    # 动作选择接口 (兼容 replay.py)
    # ──────────────────────────────────────────
    def select_action(
        self,
        game: GameEngine,
        player_idx: int,
        last_played_cards: list,
        explore: bool = True,
    ) -> Tuple[Optional[list], np.ndarray, float]:
        """兼容接口: 返回 (action, action_enc, q_value_placeholder)"""
        actions, action_encs = self._get_legal_encs(
            game, player_idx, last_played_cards
        )
        deterministic = not explore
        chosen_idx, log_prob = self.policy_net.select_action(
            encode_state(game, player_idx, last_played_cards),
            action_encs,
            deterministic=deterministic,
        )
        action = actions[chosen_idx]
        action_enc = action_encs[chosen_idx]
        return action, action_enc, log_prob


# ────────────────────────────────────────────
# 入口
# ────────────────────────────────────────────
if __name__ == "__main__":
    trainer = PolicyGradientTrainer()
    trainer.train(num_episodes=10_000)
