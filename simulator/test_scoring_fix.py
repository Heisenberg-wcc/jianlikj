"""
计分系统修复验证测试
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from game_engine import GameEngine, GameState
from card import Card
from config import Config


def test_round_just_reset_attr():
    """测试1: GameState 新增 round_just_reset 属性"""
    gs = GameState()
    assert hasattr(gs, 'round_just_reset'), "缺少 round_just_reset 属性"
    assert gs.round_just_reset == False, "初始值应为 False"
    print("✓ 测试1通过: round_just_reset 属性存在且初始为 False")


def test_round_score_settlement():
    """测试2: pass_round() 中轮次得分结算"""
    game = GameEngine()
    game.initialize()

    # 找一张分值牌(10点)
    p0 = game.players[0]
    score_cards = [c for c in p0.hand if c.rank == 10]
    if not score_cards:
        score_cards = [c for c in p0.hand if c.rank == 5]
    if not score_cards:
        score_cards = [c for c in p0.hand if c.rank == 13]

    assert score_cards, "P1手中无分值牌，无法测试"

    card = score_cards[0]
    expected_score = card.get_score_value()

    # P1出牌
    success = game.play_card(0, [card])
    assert success, "出牌失败"

    # 其他3人pass
    game.pass_round(1)
    game.pass_round(2)
    game.pass_round(3)

    # 验证: team_scores[0] 应包含该分值牌的分数(P1属于team0)
    assert game.state.team_scores[0] >= expected_score, \
        f"轮次得分未结算! team_scores={game.state.team_scores}, expected>={expected_score}"
    assert game.state.round_just_reset == True, \
        "round_just_reset 应为 True"

    print(f"✓ 测试2通过: 出牌{card}(分值{expected_score}), 3人pass后 team_scores={game.state.team_scores}, round_just_reset=True")


def test_remaining_hand_scoring():
    """测试3: 剩余手牌计入对方队伍"""
    game = GameEngine()
    game.initialize()

    # 模拟: 手动让P1和P3出完牌(同队)
    # 为简单起见直接操纵 finished_order
    game.players[0].hand = []  # P1出完
    game.players[2].hand = []  # P3出完
    game.finished_order = [0, 2]
    game.players[0].position = 1
    game.players[2].position = 2

    # P2和P4还有手牌(含分值牌)
    # 确保P2有一张10分牌
    p2_score = sum(c.get_score_value() for c in game.players[1].hand)
    p4_score = sum(c.get_score_value() for c in game.players[3].hand)

    # 记录调用前的分数
    game.state.team_scores = [0, 0]

    # 调用计算最终得分
    game._calculate_final_scores()

    # P2(team1)和P4(team1)的剩余牌分应计入team0
    # team1的剩余牌应计入team0 (P2和P4都是team1)
    remaining_to_team0 = p2_score + p4_score

    # 上游+二游同队(team0): team0 += 60, team1 -= 60
    expected_team0 = remaining_to_team0 + Config.REWARD_1ST_2ND
    expected_team1 = -Config.REWARD_1ST_2ND

    assert game.state.team_scores[0] == expected_team0, \
        f"team0得分错误: {game.state.team_scores[0]} != {expected_team0}"
    assert game.state.team_scores[1] == expected_team1, \
        f"team1得分错误: {game.state.team_scores[1]} != {expected_team1}"

    print(f"✓ 测试3通过: 剩余手牌P2={p2_score}分+P4={p4_score}分 计入team0, "
          f"排名奖惩双向转移后 team_scores={game.state.team_scores}")


def test_get_results_no_rank_bonus():
    """测试4: _get_results() 无排名速度加权"""
    from train_q_mc import MCQTrainer

    trainer = MCQTrainer()
    game = GameEngine()
    game.initialize()

    # 模拟结束状态
    game.finished_order = [0, 2, 1, 3]
    game.state.team_scores = [100, 50]
    game.state.game_over = True

    results = trainer._get_results(game)

    # team0 (P1,P3) 获胜，所有人应为 +1.0
    assert results[0] == 1.0, f"P1 应为 +1.0, 实际 {results[0]}"
    assert results[2] == 1.0, f"P3 应为 +1.0, 实际 {results[2]}"
    # team1 (P2,P4) 失败，所有人应为 -1.0
    assert results[1] == -1.0, f"P2 应为 -1.0, 实际 {results[1]}"
    assert results[3] == -1.0, f"P4 应为 -1.0, 实际 {results[3]}"

    print(f"✓ 测试4通过: 无排名速度加权, results={results}")


def test_bidirectional_rank_transfer():
    """测试5: 排名奖惩双向转移"""
    game = GameEngine()
    game.initialize()

    # 清空手牌模拟
    for p in game.players:
        p.hand = []

    game.finished_order = [0, 2]  # P1+P3 同队前两名
    game.players[0].position = 1
    game.players[2].position = 2
    game.state.team_scores = [80, 120]

    game._calculate_final_scores()

    # 上游+二游同队(team0): team0 +60, team1 -60
    assert game.state.team_scores[0] == 80 + 60, \
        f"team0 应为 {80+60}, 实际 {game.state.team_scores[0]}"
    assert game.state.team_scores[1] == 120 - 60, \
        f"team1 应为 {120-60}, 实际 {game.state.team_scores[1]}"

    print(f"✓ 测试5通过: 排名双向转移 team_scores={game.state.team_scores}")


def test_full_episode():
    """测试6: 完整对局端到端测试"""
    from train_q_mc import MCQTrainer

    trainer = MCQTrainer()
    samples, info = trainer.play_episode(render=False)

    assert len(samples) > 0, "无样本生成"
    assert 'team_scores' in info, "缺少 team_scores"

    # 验证所有奖励值为 +1.0 或 -1.0 (无速度加权)
    unique_results = set(s.result for s in samples)
    assert unique_results.issubset({1.0, -1.0}), \
        f"奖励值应仅为 +1.0/-1.0, 实际 {unique_results}"

    print(f"✓ 测试6通过: 完整对局 steps={info['steps']}, "
          f"team_scores={info['team_scores']}, 奖励值={unique_results}")


if __name__ == '__main__':
    print("=" * 56)
    print("  计分系统修复验证测试")
    print("=" * 56)

    test_round_just_reset_attr()
    test_round_score_settlement()
    test_remaining_hand_scoring()
    test_get_results_no_rank_bonus()
    test_bidirectional_rank_transfer()
    test_full_episode()

    print()
    print("=" * 56)
    print("  ✅ 所有测试通过!")
    print("=" * 56)
