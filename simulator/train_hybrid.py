"""
混合决策训练器 (规则 + 策略 + 掩码 + 历史模型池)

终极形态:
  1. 三层决策架构: 规则层 → 合法动作掩码 → 策略网络
  2. 增强奖励体系: 团队奖励 / 控牌权奖励 / 节奏奖励 / 对手接近出完惩罚
  3. 探索机制: 纯概率采样 + 动态熵控制 (无 epsilon)
  4. 历史模型池: 当前模型 vs 历史版本, 防止策略退化

训练循环:
  对局(当前模型 vs 历史版本) → 收集轨迹 → 增强奖励 → 策略梯度更新
"""

import os
import sys
import glob
import copy
import random
import math
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
from ai_base import ThreeLayerHybridAI, EnhancedRewardCalculator


# ────────────────────────────────────────────
# 轨迹步
# ────────────────────────────────────────────
class HybridTrajectoryStep:
    """单步轨迹数据 (含增强奖励信息)"""
    __slots__ = ['state', 'action_encs', 'chosen_idx', 'log_prob',
                 'reward', 'player_idx', 'source']

    def __init__(self, state, action_encs, chosen_idx, log_prob,
                 reward, player_idx, source):
        self.state       = state
        self.action_encs = action_encs
        self.chosen_idx  = chosen_idx
        self.log_prob    = log_prob
        self.reward      = reward
        self.player_idx  = player_idx
        self.source      = source      # 'rule' or 'policy'


# ────────────────────────────────────────────
# 历史模型池
# ────────────────────────────────────────────
class ModelPool:
    """
    历史模型池: 维护最近 N 个版本的策略网络参数。
    对战时以一定概率选择历史版本作为对手, 防止策略退化。
    """

    def __init__(self, max_size: int = 20):
        self.max_size = max_size
        self.pool: List[dict] = []       # 存储 state_dict 快照

    def add(self, policy_net: PolicyNetwork):
        """将当前策略网络参数添加到池中"""
        snapshot = copy.deepcopy(policy_net.state_dict())
        self.pool.append(snapshot)
        if len(self.pool) > self.max_size:
            self.pool.pop(0)

    def sample_opponent(self, latest_ratio: float = 0.7) -> Optional[dict]:
        """
        采样一个对手的参数。
        latest_ratio 概率选最新, 其余随机选历史版本。
        若池为空返回 None (使用当前模型)。
        """
        if not self.pool:
            return None

        if random.random() < latest_ratio or len(self.pool) == 1:
            return self.pool[-1]
        else:
            return random.choice(self.pool[:-1])

    def __len__(self):
        return len(self.pool)


# ────────────────────────────────────────────
# 混合决策训练器
# ────────────────────────────────────────────
class HybridTrainer:
    """混合决策系统训练器"""

    def __init__(self):
        # ── 超参数 ──
        self.gamma          = Config.PG_GAMMA
        self.entropy_start  = Config.HY_ENTROPY_START
        self.entropy_end    = Config.HY_ENTROPY_END
        self.entropy_decay  = Config.HY_ENTROPY_DECAY
        self.value_coef     = Config.PG_VALUE_COEF
        self.grad_clip      = Config.PG_GRAD_CLIP
        self.save_dir       = Config.HY_SAVE_DIR
        self.save_interval  = Config.HY_SAVE_INTERVAL
        self.pool_save_freq = Config.HY_POOL_SAVE_FREQ
        self.latest_ratio   = Config.HY_LATEST_RATIO

        # ── 网络 ──
        self.policy_net = PolicyNetwork()
        self.value_net  = ValueNetwork()
        # 对手网络 (独立实例, 从模型池加载参数)
        self.opponent_net = PolicyNetwork()

        self.policy_optimizer = optim.Adam(
            self.policy_net.parameters(), lr=Config.PG_LR_ACTOR
        )
        self.value_optimizer = optim.Adam(
            self.value_net.parameters(), lr=Config.PG_LR_CRITIC
        )

        self.action_sp = ActionSpace()
        self.rules     = RulesEngine()

        # ── 模型池 ──
        self.model_pool = ModelPool(max_size=Config.HY_MODEL_POOL_SIZE)

        self.current_episode = 0

        os.makedirs(self.save_dir, exist_ok=True)

    # ──────────────────────────────────────────
    # 动态熵系数
    # ──────────────────────────────────────────
    def _get_entropy_coef(self) -> float:
        """余弦退火式熵系数衰减"""
        progress = min(self.current_episode / max(self.entropy_decay, 1), 1.0)
        coef = self.entropy_end + 0.5 * (self.entropy_start - self.entropy_end) * (
            1 + math.cos(math.pi * progress)
        )
        return coef

    # ──────────────────────────────────────────
    # 辅助
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

    def _get_legal_encs(self, game, player_idx, last_played_cards):
        player    = game.get_player(player_idx)
        last_hand = self._get_last_hand(last_played_cards)
        legal     = self.action_sp.get_all_actions(player.hand, last_hand)
        can_pass  = len(last_played_cards) > 0
        actions, encs = [], []
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
    # 选择动作 (使用三层架构)
    # ──────────────────────────────────────────
    def _select_action_for_player(
        self,
        game: GameEngine,
        player_idx: int,
        last_played_cards: list,
        last_play_player: int,
        net: PolicyNetwork,
        deterministic: bool = False,
    ) -> Tuple[Optional[list], np.ndarray, float, str]:
        """
        三层决策:
          1. 规则层检查
          2. 合法动作掩码
          3. 策略网络选择

        Returns:
            (action, action_enc, log_prob, source)
        """
        hybrid_ai = ThreeLayerHybridAI(player_idx, game)

        last_hand = self._get_last_hand(last_played_cards)

        def policy_select_fn(actions, action_encs):
            state_vec = self._encode_state_full(game, player_idx, last_played_cards)
            net.eval()
            chosen_idx, log_prob = net.select_action(
                state_vec, action_encs, deterministic=deterministic
            )
            action = actions[chosen_idx]
            action_enc = action_encs[chosen_idx]
            return action, action_enc, log_prob

        action, source = hybrid_ai.decide_action(
            last_hand, last_play_player, last_played_cards,
            policy_select_fn=policy_select_fn,
        )

        action_enc = encode_action(action)

        # 如果是规则决策, log_prob 设为 0 (不参与梯度)
        if source == 'rule':
            return action, action_enc, 0.0, source

        # 重新计算 log_prob (用于策略梯度)
        state_vec = self._encode_state_full(game, player_idx, last_played_cards)
        actions, action_encs = self._get_legal_encs(game, player_idx, last_played_cards)
        net.eval()
        chosen_idx, log_prob = net.select_action(
            state_vec, action_encs, deterministic=deterministic
        )
        # 找到实际选择的动作在列表中的位置
        actual_idx = None
        for i, a in enumerate(actions):
            if action is None and a is None:
                actual_idx = i
                break
            if action is not None and a is not None:
                if len(action) == len(a) and all(c1.rank == c2.rank for c1, c2 in zip(action, a)):
                    actual_idx = i
                    break
        if actual_idx is not None:
            probs_np, logits = net.get_action_probs(state_vec, action_encs)
            log_prob = float(torch.log(F.softmax(logits, dim=0)[actual_idx].detach() + 1e-8))

        return action, action_enc, log_prob, source

    # ──────────────────────────────────────────
    # 对局执行 (混合对战)
    # ──────────────────────────────────────────
    def play_episode(self, render: bool = False) -> Tuple[Dict[int, List[HybridTrajectoryStep]], dict]:
        """
        运行一局: team1(P0,P2) 用当前策略, team2(P1,P3) 用对手网络。
        """
        game = GameEngine()
        game.initialize()

        # 加载对手参数
        opp_params = self.model_pool.sample_opponent(self.latest_ratio)
        if opp_params is not None:
            self.opponent_net.load_state_dict(opp_params)
        else:
            self.opponent_net.load_state_dict(self.policy_net.state_dict())
        self.opponent_net.eval()

        last_played_cards: list = []
        last_play_player:  int  = -1
        passed_players:    set  = set()

        trajectories: Dict[int, List[HybridTrajectoryStep]] = {i: [] for i in range(4)}

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

            # 选择网络: team1 用当前策略, team2 用对手
            net = self.policy_net if current_idx in [0, 2] else self.opponent_net

            # 编码状态
            state_vec = self._encode_state_full(game, current_idx, last_played_cards)

            # 获取合法动作
            actions, action_encs = self._get_legal_encs(
                game, current_idx, last_played_cards
            )

            # 三层决策
            action, action_enc, log_prob, source = self._select_action_for_player(
                game, current_idx, last_played_cards, last_play_player,
                net, deterministic=False,
            )

            # 记录快照
            score_before     = list(game.get_game_state().team_scores)
            game_over_before = game.get_game_state().game_over
            lpp_before       = last_play_player

            # 执行动作
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
            else:
                game.pass_round(current_idx)
                if last_play_player >= 0:
                    passed_players.add(current_idx)

            # 轮次重置
            last_played_cards, last_play_player, passed_players = \
                self._update_round_state(game, last_played_cards,
                                         last_play_player, passed_players)

            # 增强奖励
            reward = EnhancedRewardCalculator.compute(
                game, current_idx, action, action_success,
                score_before, game_over_before,
                lpp_before, last_play_player,
            )

            # 记录轨迹 (仅 team1 用于训练)
            if current_idx in [0, 2]:
                # 找到 chosen_idx
                chosen_idx = 0
                for i, ae in enumerate(action_encs):
                    if np.array_equal(ae, action_enc):
                        chosen_idx = i
                        break

                trajectories[current_idx].append(HybridTrajectoryStep(
                    state=state_vec,
                    action_encs=action_encs,
                    chosen_idx=chosen_idx,
                    log_prob=log_prob,
                    reward=reward,
                    player_idx=current_idx,
                    source=source,
                ))

        gs = game.get_game_state()
        info = {
            'steps'       : step,
            'team1_win'   : gs.team_scores[0] > gs.team_scores[1],
            'finished'    : game.finished_order,
            'team_scores' : gs.team_scores,
        }
        return trajectories, info

    def _update_round_state(self, game, last_played_cards, last_play_player, passed_players):
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
    # 计算折扣回报
    # ──────────────────────────────────────────
    def _compute_returns(self, trajectory: List[HybridTrajectoryStep]) -> List[float]:
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
        self, trajectories: Dict[int, List[HybridTrajectoryStep]]
    ) -> Dict[str, float]:
        """训练 (仅用 team1 的轨迹)"""
        entropy_coef = self._get_entropy_coef()

        all_policy_losses = []
        all_value_losses  = []
        all_entropies     = []
        rule_count = 0
        policy_count = 0

        for pid in [0, 2]:  # 只训练 team1
            traj = trajectories[pid]
            if not traj:
                continue

            returns = self._compute_returns(traj)

            for t_step, G_t in zip(traj, returns):
                # 跳过规则决策的步骤 (不参与策略梯度)
                if t_step.source == 'rule':
                    rule_count += 1
                    # 但仍然训练 Critic
                    state_t = torch.FloatTensor(t_step.state)
                    self.value_net.train()
                    v_pred = self.value_net(state_t.unsqueeze(0))
                    value_target = torch.FloatTensor([G_t])
                    value_loss = F.mse_loss(v_pred, value_target)
                    all_value_losses.append(value_loss)
                    continue

                policy_count += 1
                state_t = torch.FloatTensor(t_step.state)

                # Critic
                self.value_net.train()
                v_pred = self.value_net(state_t.unsqueeze(0))
                value_target = torch.FloatTensor([G_t])
                value_loss = F.mse_loss(v_pred, value_target)
                all_value_losses.append(value_loss)

                # Advantage
                advantage = G_t - v_pred.item()

                # Actor
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

                entropy = -(probs * log_probs).sum()
                all_entropies.append(entropy)

        if not all_value_losses:
            return {'policy_loss': 0.0, 'value_loss': 0.0,
                    'entropy': 0.0, 'rule_pct': 0.0}

        total_value_loss = torch.stack(all_value_losses).mean()

        if all_policy_losses:
            total_policy_loss = torch.stack(all_policy_losses).mean()
            total_entropy     = torch.stack(all_entropies).mean()
            total_loss = (
                total_policy_loss
                + self.value_coef * total_value_loss
                - entropy_coef * total_entropy
            )
        else:
            total_policy_loss = torch.tensor(0.0)
            total_entropy     = torch.tensor(0.0)
            total_loss = self.value_coef * total_value_loss

        self.policy_optimizer.zero_grad()
        self.value_optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.grad_clip)
        torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), self.grad_clip)
        self.policy_optimizer.step()
        self.value_optimizer.step()

        total_steps = rule_count + policy_count
        rule_pct = rule_count / max(total_steps, 1) * 100

        return {
            'policy_loss': total_policy_loss.item(),
            'value_loss' : total_value_loss.item(),
            'entropy'    : total_entropy.item() if all_entropies else 0.0,
            'rule_pct'   : rule_pct,
        }

    # ──────────────────────────────────────────
    # 主训练循环
    # ──────────────────────────────────────────
    def train(self, num_episodes: int = 50_000):
        win_hist  = deque(maxlen=200)
        loss_hist = deque(maxlen=200)

        print("=" * 60)
        print("混合决策系统训练 (规则 + 策略 + 掩码 + 历史模型池)")
        print(f"  总局数: {num_episodes}")
        print(f"  模型池: max={Config.HY_MODEL_POOL_SIZE}, "
              f"入池频率={self.pool_save_freq}")
        print(f"  熵控制: {self.entropy_start} → {self.entropy_end}")
        print("=" * 60)

        for ep in range(1, num_episodes + 1):
            self.current_episode = ep

            # ── 对局 ──
            trajectories, info = self.play_episode()

            # ── 训练 ──
            metrics = self.train_on_episode(trajectories)

            # ── 统计 ──
            win_hist.append(1 if info['team1_win'] else 0)
            loss_hist.append(metrics['policy_loss'])

            # ── 模型池更新 ──
            if ep % self.pool_save_freq == 0:
                self.model_pool.add(self.policy_net)

            # ── 打印进度 ──
            if ep % 100 == 0:
                win_rate = np.mean(win_hist)
                avg_ploss = np.mean(loss_hist) if loss_hist else float('nan')
                ent_coef = self._get_entropy_coef()
                print(
                    f"[Ep {ep:>6}/{num_episodes}] "
                    f"Win={win_rate:.1%}  "
                    f"PLoss={avg_ploss:.4f}  "
                    f"VLoss={metrics['value_loss']:.4f}  "
                    f"Ent={metrics['entropy']:.3f}  "
                    f"EntCoef={ent_coef:.4f}  "
                    f"Rule={metrics['rule_pct']:.0f}%  "
                    f"Pool={len(self.model_pool)}"
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
    # 兼容接口
    # ──────────────────────────────────────────
    def select_action(
        self,
        game: GameEngine,
        player_idx: int,
        last_played_cards: list,
        explore: bool = True,
    ) -> Tuple[Optional[list], np.ndarray, float]:
        """兼容 replay.py 接口"""
        last_hand = self._get_last_hand(last_played_cards)
        # 更新game_state中的last_play_player以确保_encode_state_full正确工作
        gs = game.get_game_state()
        action, action_enc, log_prob, source = self._select_action_for_player(
            game, player_idx, last_played_cards, gs.last_played_player if gs.last_played_player is not None else -1,
            self.policy_net, deterministic=not explore,
        )
        return action, action_enc, log_prob


# ────────────────────────────────────────────
# 入口
# ────────────────────────────────────────────
if __name__ == "__main__":
    trainer = HybridTrainer()
    trainer.train(num_episodes=50_000)
