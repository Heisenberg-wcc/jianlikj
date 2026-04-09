"""
对局回放脚本 - Q-MC 模型
用法: python replay.py
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def run_replay():
    """Monte Carlo Q 网络对局回放"""
    from train_q_mc import MCQTrainer
    from game_engine import GameEngine
    from q_net.q_network import encode_state

    print("=" * 70)
    print("加载 Q-MC 模型...")
    print("=" * 70)

    trainer   = MCQTrainer()
    model_path = "q_models/q_net_final.pth"
    if os.path.exists(model_path):
        trainer.load(model_path)
    else:
        print(f"[WARN] 模型不存在: {model_path}, 使用随机权重运行")

    game = GameEngine()
    game.initialize()

    dealer_idx = next((i for i, p in enumerate(game.players) if p.is_dealer), 0)
    print(f"\n庄家: Player{dealer_idx + 1}")
    print("队伍: Team1 = [Player1, Player3], Team2 = [Player2, Player4]")
    print("\n--- 初始手牌 ---")
    for i, p in enumerate(game.players):
        hand_str = ', '.join(str(c) for c in p.hand)
        print(f"Player{i+1} ({'Team1' if i in [0,2] else 'Team2'}): {hand_str}")

    print("\n" + "=" * 70)
    print("游戏过程")
    print("=" * 70)

    step              = 0
    max_steps         = 800
    last_played_cards = []
    last_play_player  = -1
    passed_players    = set()

    while not game.get_game_state().game_over and step < max_steps:
        step += 1
        current_idx = game.get_game_state().current_player_idx
        player      = game.get_player(current_idx)
        team        = "Team1" if current_idx in [0, 2] else "Team2"

        # 已出完牌的玩家自动 pass
        if player.is_out():
            game.pass_round(current_idx)
            if last_play_player >= 0:
                passed_players.add(current_idx)
                last_played_cards, last_play_player, passed_players = \
                    trainer._update_round_state(game, last_played_cards,
                                                last_play_player, passed_players)
            continue

        # 选择动作 (不探索)
        action, action_enc, q_val = trainer.select_action(
            game, current_idx, last_played_cards, explore=False
        )

        if action is not None:
            cards_before = len(player.hand)
            success      = game.play_card(current_idx, action)
            if success:
                remaining = cards_before - len(action)
                try:
                    cards_str = ', '.join(str(c) for c in action)
                    print(f"Step {step:3d} | Player{current_idx+1} ({team}) "
                          f"| 出牌: [{cards_str}] "
                          f"| Q={q_val:+.3f} | 剩余: {remaining}张")
                except UnicodeEncodeError:
                    print(f"Step {step:3d} | Player{current_idx+1} ({team}) "
                          f"| 出{len(action)}张牌 "
                          f"| Q={q_val:+.3f} | 剩余: {remaining}张")
                last_played_cards = list(action)
                last_play_player  = current_idx
                passed_players    = set()
                if player.is_out():
                    pos = len(game.finished_order)
                    print(f"        >>> Player{current_idx+1} 出完牌！名次：第{pos}名 <<<")
            else:
                print(f"Step {step:3d} | Player{current_idx+1} ({team}) "
                      f"| [INVALID, forced PASS] | 剩余: {len(player.hand)}张")
                game.pass_round(current_idx)
                if last_play_player >= 0:
                    passed_players.add(current_idx)
        else:
            print(f"Step {step:3d} | Player{current_idx+1} ({team}) "
                  f"| PASS | Q={q_val:+.3f} | 剩余: {len(player.hand)}张")
            game.pass_round(current_idx)
            if last_play_player >= 0:
                passed_players.add(current_idx)

        # 轮次重置检测
        last_played_cards, last_play_player, passed_players = \
            trainer._update_round_state(game, last_played_cards,
                                        last_play_player, passed_players)

    # ── 结果 ──
    print("\n" + "=" * 70)
    print("游戏结果")
    print("=" * 70)

    team_scores = game.get_game_state().team_scores
    winner      = 0 if team_scores[0] > team_scores[1] else 1
    print(f"\n队伍得分:")
    print(f"  Team1 (Player1 + Player3): {team_scores[0]} 分")
    print(f"  Team2 (Player2 + Player4): {team_scores[1]} 分")
    print(f"\n获胜队伍: Team{winner + 1}")
    print(f"\n玩家排名:")
    for i, pid in enumerate(game.finished_order):
        print(f"  第{i+1}名: Player{pid+1} ({'Team1' if pid in [0,2] else 'Team2'})")
    print(f"\n未出完牌的玩家手牌:")
    for i, p in enumerate(game.players):
        if p.hand:
            hand_str = ', '.join(str(c) for c in p.hand)
            print(f"  Player{i+1}: [{hand_str}]")


if __name__ == "__main__":
    run_replay()
