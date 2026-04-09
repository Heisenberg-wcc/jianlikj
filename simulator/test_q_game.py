"""
模型效果验证脚本

验证方式:
  1. 模型自博弈 (纯贪心, ε=0): 验证胜率是否均衡、行为是否合理
  2. 模型 vs 随机 (Team1=模型, Team2=随机): 验证是否强于随机策略
"""

import os
import sys
import random
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from game_engine import GameEngine
from action_space import ActionSpace
from rules import RulesEngine, Hand, HandType
from train_q_mc import MCQTrainer
from q_net.q_network import encode_state, encode_action


def _get_last_hand(rules, last_played_cards):
    if not last_played_cards:
        return None
    ht = rules.detect_hand_type(last_played_cards)
    from rules import Hand, HandType
    return Hand(last_played_cards, ht) if ht != HandType.INVALID else None


def play_eval_episode(trainer, model_teams, render=False):
    """
    运行一局评估对局
    model_teams: 使用模型的玩家集合, 例如 {0,2} 表示 Team1 用模型, Team2 随机
    """
    game = GameEngine()
    game.initialize()
    rules = RulesEngine()
    action_sp = ActionSpace()

    last_played_cards = []
    last_play_player  = -1
    passed_players    = set()
    max_steps = 800
    step = 0

    while not game.get_game_state().game_over and step < max_steps:
        step += 1
        current_idx = game.get_game_state().current_player_idx
        player = game.get_player(current_idx)

        if player.is_out():
            game.pass_round(current_idx)
            if game.get_game_state().round_just_reset:
                last_played_cards = []
                last_play_player  = -1
                passed_players    = set()
                game.get_game_state().round_just_reset = False
            elif last_play_player >= 0:
                passed_players.add(current_idx)
            continue

        can_pass = len(last_played_cards) > 0
        last_hand = _get_last_hand(rules, last_played_cards)
        legal_actions = action_sp.get_all_actions(player.hand, last_hand)

        # 构建候选集
        candidates = []
        if can_pass:
            candidates.append(None)
        candidates.extend(legal_actions)
        if not candidates:
            candidates = [None]

        # 选择动作
        if current_idx in model_teams:
            # 模型贪心选择 (ε=0)
            action, action_enc, q_val = trainer.select_action(
                game, current_idx, last_played_cards, explore=False
            )
        else:
            # 纯随机选择
            action = random.choice(candidates)
            action_enc = encode_action(action)
            q_val = 0.0

        # 执行动作
        if action is not None:
            success = game.play_card(current_idx, action)
            if not success:
                game.pass_round(current_idx)
                action = None
                if game.get_game_state().round_just_reset:
                    last_played_cards = []
                    last_play_player  = -1
                    passed_players    = set()
                    game.get_game_state().round_just_reset = False
                elif last_play_player >= 0:
                    passed_players.add(current_idx)
            else:
                last_played_cards = list(action)
                last_play_player  = current_idx
                passed_players    = set()
                if render:
                    try:
                        cards_str = ', '.join(str(c) for c in action)
                        tag = "[模型]" if current_idx in model_teams else "[随机]"
                        print(f"  Step {step:3d} | P{current_idx+1}{tag} 出牌: [{cards_str}]  Q={q_val:.3f}  剩余:{len(player.hand)}张")
                    except UnicodeEncodeError:
                        pass
        else:
            game.pass_round(current_idx)
            if game.get_game_state().round_just_reset:
                last_played_cards = []
                last_play_player  = -1
                passed_players    = set()
                game.get_game_state().round_just_reset = False
            elif last_play_player >= 0:
                passed_players.add(current_idx)
            if render:
                tag = "[模型]" if current_idx in model_teams else "[随机]"
                print(f"  Step {step:3d} | P{current_idx+1}{tag} PASS")

    scores = game.get_game_state().team_scores
    team1_win = scores[0] > scores[1]
    return {
        'steps'       : step,
        'team1_win'   : team1_win,
        'team_scores' : scores,
        'finished'    : game.finished_order,
    }


def evaluate(model_path, num_games=500):
    print("=" * 60)
    print(f"模型效果验证  模型: {model_path}")
    print(f"每项评估局数: {num_games} 局")
    print("=" * 60)

    trainer = MCQTrainer()
    if os.path.exists(model_path):
        trainer.load(model_path)
        print(f"[OK] 模型加载成功\n")
    else:
        print(f"[WARN] 未找到模型文件: {model_path}, 使用随机初始化网络\n")

    # ── 1. 模型自博弈 (ε=0) ──
    print("[测试1] 模型自博弈 (4人全用模型, 贪心)")
    all_players = {0, 1, 2, 3}
    wins, steps_list, score_diffs = 0, [], []
    for i in range(num_games):
        info = play_eval_episode(trainer, model_teams=all_players)
        wins += int(info['team1_win'])
        steps_list.append(info['steps'])
        score_diffs.append(info['team_scores'][0] - info['team_scores'][1])
        if (i + 1) % 100 == 0:
            print(f"  进度: {i+1}/{num_games}")

    print(f"  Team1 胜率: {wins/num_games:.1%}  (期望约50%, 自博弈对称)")
    print(f"  平均步数: {np.mean(steps_list):.1f}  标准差: {np.std(steps_list):.1f}")
    print(f"  平均分差: {np.mean(score_diffs):.1f}")
    print()

    # ── 2. 模型 vs 随机 ──
    print("[测试2] 模型(Team1: P1+P3) vs 随机(Team2: P2+P4)")
    model_teams = {0, 2}   # 0-indexed
    wins_vs_random, steps_list2 = 0, []
    for i in range(num_games):
        info = play_eval_episode(trainer, model_teams=model_teams)
        wins_vs_random += int(info['team1_win'])
        steps_list2.append(info['steps'])
        if (i + 1) % 100 == 0:
            print(f"  进度: {i+1}/{num_games}")

    wr = wins_vs_random / num_games
    print(f"  模型胜率: {wr:.1%}  (随机基线约50%, 越高越好)")
    print(f"  平均步数: {np.mean(steps_list2):.1f}")
    print()

    # ── 3. 随机 vs 随机 (基线对照) ──
    print("[测试3] 随机 vs 随机 (基线对照)")
    wins_rand, steps_list3 = 0, []
    for i in range(num_games):
        info = play_eval_episode(trainer, model_teams=set())
        wins_rand += int(info['team1_win'])
        steps_list3.append(info['steps'])
        if (i + 1) % 100 == 0:
            print(f"  进度: {i+1}/{num_games}")

    print(f"  Team1 胜率: {wins_rand/num_games:.1%}  (基线, 期望约50%)")
    print(f"  平均步数: {np.mean(steps_list3):.1f}")
    print()

    print("=" * 60)
    print("验证结论:")
    print(f"  模型 vs 随机胜率: {wr:.1%}")
    if wr >= 0.60:
        print("  [优秀] 模型显著强于随机策略")
    elif wr >= 0.55:
        print("  [良好] 模型略强于随机策略")
    elif wr >= 0.48:
        print("  [一般] 模型与随机策略接近, 可继续训练")
    else:
        print("  [待改进] 模型弱于随机, 建议检查训练流程")
    print("=" * 60)


def compare_models(path_a, path_b, num_games=300):
    """对比两个模型各自 vs 随机的胜率"""
    print("=" * 60)
    print(f"模型对比: {os.path.basename(path_a)}  vs  {os.path.basename(path_b)}")
    print(f"各自 vs 随机 {num_games} 局")
    print("=" * 60)

    results = {}
    for label, path in [(os.path.basename(path_a), path_a),
                        (os.path.basename(path_b), path_b)]:
        trainer = MCQTrainer()
        if os.path.exists(path):
            trainer.load(path)
        else:
            print(f"[WARN] 文件不存在: {path}")
            results[label] = None
            continue

        wins = 0
        print(f"\n测试: {label}")
        for i in range(num_games):
            info = play_eval_episode(trainer, model_teams={0, 2})
            wins += int(info['team1_win'])
            if (i + 1) % 100 == 0:
                print(f"  进度: {i+1}/{num_games}")

        wr = wins / num_games
        results[label] = wr
        print(f"  vs随机胜率: {wr:.1%}")

    print("\n" + "=" * 60)
    print("对比结果:")
    items = [(k, v) for k, v in results.items() if v is not None]
    if len(items) == 2:
        (n1, w1), (n2, w2) = items
        print(f"  {n1}: {w1:.1%}")
        print(f"  {n2}: {w2:.1%}")
        diff = w1 - w2
        if abs(diff) < 0.03:
            print(f"  结论: 两者接近 (diff={diff:+.1%}), 没有显著差异")
        elif diff > 0:
            print(f"  结论: {n1} 更强 (+{diff:.1%})")
        else:
            print(f"  结论: {n2} 更强 (+{-diff:.1%})")
    print("=" * 60)


if __name__ == "__main__":
    path_a = os.path.join("q_models", "model1.0.pth")
    path_b = os.path.join("q_models", "base.pth")
    compare_models(path_a, path_b, num_games=300)
