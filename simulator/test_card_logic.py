"""
出牌组合逻辑验证测试
覆盖设计文档中场景A(主动出牌)和场景B(压牌)的所有验证矩阵
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from card import Card
from rules import RulesEngine, Hand, HandType
from action_space import ActionSpace

rules = RulesEngine()
action_space = ActionSpace()

# ============================================================
# 辅助函数
# ============================================================
passed = 0
failed = 0

def cards_str(cards):
    """安全打印牌列表"""
    return [f"{c.rank}_{c.suit}" for c in cards]

def check(test_name, condition, input_info="", output_info=""):
    """验证并打印结果"""
    global passed, failed
    status = "PASS" if condition else "FAIL"
    if not condition:
        failed += 1
    else:
        passed += 1
    print(f"  [{status}] {test_name}")
    if input_info:
        print(f"    输入: {input_info}")
    if output_info:
        print(f"    输出: {output_info}")
    if not condition:
        print(f"    *** 断言失败 ***")
    return condition

def get_action_types(actions):
    """获取动作列表中每个动作的牌型"""
    return [rules.detect_hand_type(a) for a in actions]

def has_cards_with_ranks(actions, target_ranks_list):
    """检查动作列表中是否包含指定rank组合的动作"""
    for action in actions:
        action_ranks = sorted([c.rank for c in action])
        if action_ranks == sorted(target_ranks_list):
            return True
    return False

# ============================================================
# 场景A: 主动出牌验证
# ============================================================
print("=" * 60)
print("场景A: 主动出牌验证")
print("=" * 60)

# 构造一个综合测试手牌
test_hand_a = [
    Card(3, 'spade'), Card(3, 'heart'), Card(3, 'club'),
    Card(4, 'spade'), Card(4, 'heart'), Card(4, 'club'),
    Card(5, 'spade'), Card(5, 'heart'), Card(5, 'club'), Card(5, 'diamond'),
    Card(10, 'spade'), Card(10, 'heart'),
    Card(13, 'spade'), Card(13, 'heart'),
    Card(14, 'spade'),
    Card(15, 'spade'), Card(15, 'heart'),  # 2
]

print(f"\n测试手牌: {cards_str(test_hand_a)}")
actions_a = action_space.get_all_actions(test_hand_a, last_hand=None)
action_types = get_action_types(actions_a)
print(f"生成合法动作总数: {len(actions_a)}")

# A1. 单张生成 - 简化后每个rank只生成一个
print("\n--- A1. 单张生成 ---")
singles = [a for a, t in zip(actions_a, action_types) if t == HandType.SINGLE]
unique_ranks_in_hand = set(c.rank for c in test_hand_a)
check("单张数量等于唯一rank数",
      len(singles) == len(unique_ranks_in_hand),
      f"唯一rank数={len(unique_ranks_in_hand)}",
      f"单张动作{len(singles)}个")

# A2. 对子生成 - 简化后每个rank只生成一个对子
print("\n--- A2. 对子生成 ---")
pairs = [a for a, t in zip(actions_a, action_types) if t == HandType.PAIR]
# rank=3有3张->1对, rank=4有3张->1对, rank=5有4张->1对, rank=10有2张->1对, rank=13有2张->1对, rank=15有2张->1对
expected_pairs = 6  # 每个>=2张的rank只生成一个对子
check("对子组合数量正确",
      len(pairs) == expected_pairs,
      f"各rank张数: 3x3,4x3,5x4,10x2,13x2,15x2",
      f"期望{expected_pairs}个对子, 实际{len(pairs)}个")

# A3. 连对生成
print("\n--- A3. 连对生成 ---")
consec_pairs = [a for a, t in zip(actions_a, action_types) if t == HandType.CONSECUTIVE_PAIRS]
# 连续有2张以上的rank(排除2): 3,4,5,10,13,14
# 连续序列: 3-4, 3-4-5, 4-5, 13-14 (10不连续,10和13不连续)
# 所以至少应包含 3-4 连对和 4-5 连对和 3-4-5 连对
has_34 = any(sorted([c.rank for c in a]) in [[3,3,4,4]] for a in consec_pairs)
has_45 = any(sorted([c.rank for c in a]) == [4,4,5,5] for a in consec_pairs)
has_345 = any(sorted([c.rank for c in a]) == [3,3,4,4,5,5] for a in consec_pairs)
check("包含3344连对", has_34, "", f"连对总数: {len(consec_pairs)}")
check("包含4455连对", has_45)
check("包含334455连对", has_345)
# 2不参与连对
has_2_in_consec = any(any(c.rank == 15 for c in a) for a in consec_pairs)
check("2不参与连对", not has_2_in_consec)

# A4. 连三张生成
print("\n--- A4. 连三张生成 ---")
consec_threes = [a for a, t in zip(actions_a, action_types) if t == HandType.CONSECUTIVE_THREES]
# rank=3有3张, rank=4有3张, rank=5有4张(可选3张)
# 连续序列: 3-4, 3-4-5, 4-5
has_333444 = any(sorted([c.rank for c in a]) == [3,3,3,4,4,4] for a in consec_threes)
has_444555 = any(sorted([c.rank for c in a]) == [4,4,4,5,5,5] for a in consec_threes)
check("包含333444连三张", has_333444, "", f"连三张总数: {len(consec_threes)}")
check("包含444555连三张", has_444555)
has_2_in_threes = any(any(c.rank == 15 for c in a) for a in consec_threes)
check("2不参与连三张", not has_2_in_threes)

# A5. 炸弹生成
print("\n--- A5. 炸弹生成 ---")
bombs = [a for a, t in zip(actions_a, action_types) if t == HandType.BOMB]
# rank=5有4张 -> 4张炸弹
has_bomb_5 = any(sorted([c.rank for c in a]) == [5,5,5,5] for a in bombs)
check("包含5555炸弹", has_bomb_5, "", f"炸弹总数: {len(bombs)}")

# A6. 510K生成 - 简化后区分纯色/杂色
print("\n--- A6. 510K生成 ---")
ftk = [a for a, t in zip(actions_a, action_types) if t == HandType.FIVE_TEN_KING]
# 5有4张, 10有2张(spade,heart), K(13)有2张(spade,heart)
# 纯色: spade(5♠+10♠+K♠)=1, heart(5♥+10♥+K♥)=1
# 杂色: 1个代表
# 总计: 2纯色 + 1杂色 = 3
check("510K组合数量正确(纯色+杂色)",
      len(ftk) == 3,
      f"5x4, 10x2, K(13)x2 -> 2纯色+1杂色",
      f"期望3个, 实际{len(ftk)}个")
# 验证纯色510K存在
pure_ftk = [a for a in ftk if all(c.suit == a[0].suit for c in a)]
mixed_ftk = [a for a in ftk if not all(c.suit == a[0].suit for c in a)]
check("510K中有2个纯色", len(pure_ftk) == 2,
      "", f"纯色数量={len(pure_ftk)}")
check("510K中有1个杂色", len(mixed_ftk) == 1,
      "", f"杂色数量={len(mixed_ftk)}")

# A7. 非法牌型过滤 + 无四王
print("\n--- A7. 非法牌型过滤 ---")
invalid_types = [t for t in action_types if t == HandType.INVALID]
check("无INVALID牌型", len(invalid_types) == 0)
# 确保没有四王
has_four_kings = any(t.value == "four_kings" for t in action_types)
check("无四王牌型", not has_four_kings)


# ============================================================
# 场景B: 压牌验证
# ============================================================
print("\n" + "=" * 60)
print("场景B: 压牌验证")
print("=" * 60)

# B1. 单张压单张
print("\n--- B1. 单张压单张 ---")
last_single_k = Hand([Card(13, 'spade')], HandType.SINGLE)
hand_b1 = [
    Card(3, 'spade'), Card(10, 'heart'),
    Card(13, 'heart'),  # K,不能压K
    Card(14, 'spade'),  # A,能压
    Card(15, 'spade'),  # 2,能压
    Card(16, None),     # 小王,能压
    Card(17, None),     # 大王,能压
]
actions_b1 = action_space.get_all_actions(hand_b1, last_single_k)
singles_b1 = [a for a in actions_b1 if rules.detect_hand_type(a) == HandType.SINGLE]
single_ranks = sorted([a[0].rank for a in singles_b1])
print(f"  输入: 上一手=单张K(13), 手牌={cards_str(hand_b1)}")
print(f"  输出: 能压过的单张ranks={single_ranks}")
check("只生成rank>13的单张",
      all(r > 13 for r in single_ranks),
      "", f"ranks: {single_ranks}")
check("包含A(14)", 14 in single_ranks)
check("包含2(15)", 15 in single_ranks)
check("包含小王(16)", 16 in single_ranks)
check("包含大王(17)", 17 in single_ranks)
check("不包含3(3)和10(10)和K(13)", 3 not in single_ranks and 10 not in single_ranks and 13 not in single_ranks)

# B2. 对子压对子
print("\n--- B2. 对子压对子 ---")
last_pair_5 = Hand([Card(5, 'spade'), Card(5, 'heart')], HandType.PAIR)
hand_b2 = [
    Card(3, 'spade'), Card(3, 'heart'),  # rank3 < 5
    Card(5, 'club'), Card(5, 'diamond'),  # rank5 == 5, 不能压
    Card(8, 'spade'), Card(8, 'heart'),  # rank8 > 5, 能压
    Card(14, 'spade'), Card(14, 'heart'),  # rank14 > 5, 能压
]
actions_b2 = action_space.get_all_actions(hand_b2, last_pair_5)
pairs_b2 = [a for a in actions_b2 if rules.detect_hand_type(a) == HandType.PAIR]
pair_ranks_b2 = sorted(set([a[0].rank for a in pairs_b2]))
print(f"  输入: 上一手=对子55, 手牌={cards_str(hand_b2)}")
print(f"  输出: 能压过的对子ranks={pair_ranks_b2}")
check("只生成rank>5的对子", all(r > 5 for r in pair_ranks_b2))
check("包含对子88", 8 in pair_ranks_b2)
check("包含对子AA", 14 in pair_ranks_b2)
check("不包含对子33和55", 3 not in pair_ranks_b2 and 5 not in pair_ranks_b2)

# B3. 连对压连对(含边界rank) - 验证3.2修复
print("\n--- B3. 连对压连对(含边界rank, 验证3.2修复) ---")
last_consec_pair_34 = Hand([
    Card(3, 'spade'), Card(3, 'heart'),
    Card(4, 'spade'), Card(4, 'heart')
], HandType.CONSECUTIVE_PAIRS)
hand_b3 = [
    Card(4, 'spade'), Card(4, 'heart'),
    Card(5, 'spade'), Card(5, 'heart'),
    Card(6, 'spade'), Card(6, 'heart'),
]
actions_b3 = action_space.get_all_actions(hand_b3, last_consec_pair_34)
consec_pairs_b3 = [a for a in actions_b3 if rules.detect_hand_type(a) == HandType.CONSECUTIVE_PAIRS]
print(f"  输入: 上一手=3344, 手牌={cards_str(hand_b3)}")
print(f"  输出: 能压过的连对:")
for a in consec_pairs_b3:
    print(f"    {cards_str(a)} (max_rank={max(c.rank for c in a)})")
has_4455 = any(sorted([c.rank for c in a]) == [4,4,5,5] for a in consec_pairs_b3)
has_5566 = any(sorted([c.rank for c in a]) == [5,5,6,6] for a in consec_pairs_b3)
check("4455应出现(边界rank修复)", has_4455)
check("5566应出现", has_5566)

# B4. 连三张压连三张(含边界rank) - 验证3.3修复
print("\n--- B4. 连三张压连三张(含边界rank, 验证3.3修复) ---")
last_consec_three_34 = Hand([
    Card(3, 'spade'), Card(3, 'heart'), Card(3, 'club'),
    Card(4, 'spade'), Card(4, 'heart'), Card(4, 'club')
], HandType.CONSECUTIVE_THREES)
hand_b4 = [
    Card(4, 'spade'), Card(4, 'heart'), Card(4, 'club'),
    Card(5, 'spade'), Card(5, 'heart'), Card(5, 'club'),
    Card(6, 'spade'), Card(6, 'heart'), Card(6, 'club'),
]
actions_b4 = action_space.get_all_actions(hand_b4, last_consec_three_34)
consec_threes_b4 = [a for a in actions_b4 if rules.detect_hand_type(a) == HandType.CONSECUTIVE_THREES]
print(f"  输入: 上一手=333444, 手牌={cards_str(hand_b4)}")
print(f"  输出: 能压过的连三张:")
for a in consec_threes_b4:
    print(f"    {cards_str(a)} (max_rank={max(c.rank for c in a)})")
has_444555 = any(sorted([c.rank for c in a]) == [4,4,4,5,5,5] for a in consec_threes_b4)
has_555666 = any(sorted([c.rank for c in a]) == [5,5,5,6,6,6] for a in consec_threes_b4)
check("444555应出现(边界rank修复)", has_444555)
check("555666应出现", has_555666)

# B5. 连对/连三张张数约束
print("\n--- B5. 连对张数约束 ---")
# 上一手4张连对3344, 手中有6张连对334455, 不能压(张数不同)
hand_b5 = [
    Card(3, 'spade'), Card(3, 'heart'),
    Card(4, 'spade'), Card(4, 'heart'),
    Card(5, 'spade'), Card(5, 'heart'),
]
actions_b5 = action_space.get_all_actions(hand_b5, last_consec_pair_34)
# 334455(6张)不能压3344(4张), 只有4455(4张)和4556不连续...只有4455
consec_pairs_b5 = [a for a in actions_b5 if rules.detect_hand_type(a) == HandType.CONSECUTIVE_PAIRS]
lengths = [len(a) for a in consec_pairs_b5]
has_6_len = 6 in lengths
check("6张连对不能压4张连对", not has_6_len,
      "上一手=3344(4张), 手中有3-4-5各2张",
      f"连对张数: {lengths}")

# B6. 炸弹压炸弹(同张数)
print("\n--- B6. 炸弹压炸弹(同张数) ---")
last_bomb_5 = Hand([
    Card(5, 'spade'), Card(5, 'heart'), Card(5, 'club'), Card(5, 'diamond')
], HandType.BOMB)
hand_b6 = [
    Card(3, 'spade'), Card(3, 'heart'), Card(3, 'club'), Card(3, 'diamond'),  # rank3<5
    Card(8, 'spade'), Card(8, 'heart'), Card(8, 'club'), Card(8, 'diamond'),  # rank8>5
]
actions_b6 = action_space.get_all_actions(hand_b6, last_bomb_5)
bombs_b6 = [a for a in actions_b6 if rules.detect_hand_type(a) == HandType.BOMB]
bomb_ranks_b6 = [a[0].rank for a in bombs_b6]
print(f"  输入: 上一手=5555, 手牌含3333和8888")
print(f"  输出: 能压过的炸弹ranks={bomb_ranks_b6}")
check("8888应出现", 8 in bomb_ranks_b6)
check("3333不应出现", 3 not in bomb_ranks_b6)

# B7. 炸弹压炸弹(多张数) - 验证3.4修复
print("\n--- B7. 炸弹压炸弹(多张数, 验证3.4修复) ---")
hand_b7 = [
    Card(6, 'spade'), Card(6, 'heart'), Card(6, 'club'),
    Card(6, 'diamond'), Card(6, 'spade'), Card(6, 'heart'),  # 6张6 (两副牌)
]
# 修正: 两副牌的6, 需要不同的花色标识
hand_b7 = [
    Card(6, 'spade'), Card(6, 'heart'), Card(6, 'club'), Card(6, 'diamond'),  # 第一副
    Card(6, 'spade'), Card(6, 'heart'),  # 第二副(重复花色)
]
actions_b7 = action_space.get_all_actions(hand_b7, last_bomb_5)
bombs_b7 = [a for a in actions_b7 if rules.detect_hand_type(a) == HandType.BOMB]
bomb_sizes_b7 = sorted(set([len(a) for a in bombs_b7]))
print(f"  输入: 上一手=5555(4张), 手牌=6张6")
print(f"  输出: 炸弹张数集合={bomb_sizes_b7}")
check("应生成4张6的炸弹", 4 in bomb_sizes_b7)
check("应生成5张6的炸弹", 5 in bomb_sizes_b7)
check("应生成6张6的炸弹", 6 in bomb_sizes_b7)

# B8. 炸弹压非炸弹
print("\n--- B8. 炸弹压非炸弹 ---")
last_single_k2 = Hand([Card(13, 'spade')], HandType.SINGLE)
hand_b8 = [
    Card(3, 'spade'), Card(3, 'heart'), Card(3, 'club'), Card(3, 'diamond'),
    Card(7, 'spade'),
]
actions_b8 = action_space.get_all_actions(hand_b8, last_single_k2)
bombs_b8 = [a for a in actions_b8 if rules.detect_hand_type(a) == HandType.BOMB]
check("炸弹应出现在压单张的选项中",
      len(bombs_b8) > 0,
      "上一手=单张K, 手中有3333",
      f"炸弹数量: {len(bombs_b8)}")

# B9. 510K压普通牌型 - 验证3.5修复
print("\n--- B9. 510K压普通牌型(验证3.5修复) ---")
hand_510k = [
    Card(5, 'spade'), Card(10, 'heart'), Card(13, 'club'),
    Card(5, 'heart'), Card(10, 'club'), Card(13, 'spade'),
    Card(7, 'spade'),
]

# 510K 压单张
last_single_3 = Hand([Card(3, 'spade')], HandType.SINGLE)
actions_510k_s = action_space.get_all_actions(hand_510k, last_single_3)
ftk_s = [a for a in actions_510k_s if rules.detect_hand_type(a) == HandType.FIVE_TEN_KING]
check("510K能压单张",
      len(ftk_s) > 0,
      "上一手=单张3",
      f"510K组合数: {len(ftk_s)}")

# 510K 压对子
last_pair_3 = Hand([Card(3, 'spade'), Card(3, 'heart')], HandType.PAIR)
actions_510k_p = action_space.get_all_actions(hand_510k, last_pair_3)
ftk_p = [a for a in actions_510k_p if rules.detect_hand_type(a) == HandType.FIVE_TEN_KING]
check("510K能压对子",
      len(ftk_p) > 0,
      "上一手=对子33",
      f"510K组合数: {len(ftk_p)}")

# 510K 压连对
last_cp = Hand([Card(3,'s'), Card(3,'h'), Card(4,'s'), Card(4,'h')], HandType.CONSECUTIVE_PAIRS)
actions_510k_cp = action_space.get_all_actions(hand_510k, last_cp)
ftk_cp = [a for a in actions_510k_cp if rules.detect_hand_type(a) == HandType.FIVE_TEN_KING]
check("510K能压连对",
      len(ftk_cp) > 0,
      "上一手=3344",
      f"510K组合数: {len(ftk_cp)}")

# 510K 压连三张
last_ct = Hand([
    Card(3,'spade'), Card(3,'heart'), Card(3,'club'),
    Card(4,'spade'), Card(4,'heart'), Card(4,'club')
], HandType.CONSECUTIVE_THREES)
actions_510k_ct = action_space.get_all_actions(hand_510k, last_ct)
ftk_ct = [a for a in actions_510k_ct if rules.detect_hand_type(a) == HandType.FIVE_TEN_KING]
check("510K能压连三张",
      len(ftk_ct) > 0,
      "上一手=333444",
      f"510K组合数: {len(ftk_ct)}")

# B10. 510K不可压炸弹
print("\n--- B10. 510K不可压炸弹 ---")
last_bomb = Hand([
    Card(3, 'spade'), Card(3, 'heart'), Card(3, 'club'), Card(3, 'diamond')
], HandType.BOMB)
actions_510k_bomb = action_space.get_all_actions(hand_510k, last_bomb)
ftk_bomb = [a for a in actions_510k_bomb if rules.detect_hand_type(a) == HandType.FIVE_TEN_KING]
check("510K不能压炸弹",
      len(ftk_bomb) == 0,
      "上一手=3333炸弹",
      f"510K组合数: {len(ftk_bomb)}")

# B11. 510K互压规则 (杂色510K不可互压, 手中无纯色可用)
print("\n--- B11. 510K互压规则(手中无纯色510K) ---")
last_510k = Hand([Card(5, 'spade'), Card(10, 'heart'), Card(13, 'club')], HandType.FIVE_TEN_KING)
actions_510k_vs = action_space.get_all_actions(hand_510k, last_510k)
ftk_vs = [a for a in actions_510k_vs if rules.detect_hand_type(a) == HandType.FIVE_TEN_KING]
# 手牌中无法组成纯色510K (5:spade/heart, 10:heart/club, K:club/spade, 无同花色组合)
check("手中无纯色510K时无法压杂色510K",
      len(ftk_vs) == 0,
      "上一手=杂色510K, 手中无纯色510K可用",
      f"510K组合数: {len(ftk_vs)}")

# B11b. 纯色510K压杂色510K
print("\n--- B11b. 纯色510K压杂色510K ---")
hand_pure_510k = [
    Card(5, 'spade'), Card(10, 'spade'), Card(13, 'spade'),  # 纯色♠
    Card(7, 'heart'),
]
last_mixed_510k = Hand([Card(5, 'heart'), Card(10, 'club'), Card(13, 'diamond')], HandType.FIVE_TEN_KING)
actions_pure_vs_mixed = action_space.get_all_actions(hand_pure_510k, last_mixed_510k)
ftk_pure_beat = [a for a in actions_pure_vs_mixed if rules.detect_hand_type(a) == HandType.FIVE_TEN_KING]
check("纯色510K能压杂色510K",
      len(ftk_pure_beat) > 0,
      "上一手=杂色510K, 手中有5♠10♠K♠",
      f"510K组合数: {len(ftk_pure_beat)}")
# 确认压出的510K是纯色
if ftk_pure_beat:
    check("压出的510K是纯色",
          all(c.suit == ftk_pure_beat[0][0].suit for c in ftk_pure_beat[0]))

# B11c. 纯色510K不可压纯色510K
print("\n--- B11c. 纯色510K不可压纯色510K ---")
last_pure_510k = Hand([Card(5, 'heart'), Card(10, 'heart'), Card(13, 'heart')], HandType.FIVE_TEN_KING)
actions_pure_vs_pure = action_space.get_all_actions(hand_pure_510k, last_pure_510k)
ftk_pure_vs_pure = [a for a in actions_pure_vs_pure if rules.detect_hand_type(a) == HandType.FIVE_TEN_KING]
check("纯色510K不能压纯色510K",
      len(ftk_pure_vs_pure) == 0,
      "上一手=纯色510K♥, 手中有5♠10♠K♠",
      f"510K组合数: {len(ftk_pure_vs_pure)}")

# B12. 炸弹压510K
print("\n--- B12. 炸弹压510K ---")
hand_b12 = [
    Card(3, 'spade'), Card(3, 'heart'), Card(3, 'club'), Card(3, 'diamond'),
    Card(7, 'spade'),
]
actions_b12 = action_space.get_all_actions(hand_b12, last_510k)
bombs_b12 = [a for a in actions_b12 if rules.detect_hand_type(a) == HandType.BOMB]
check("炸弹能压510K",
      len(bombs_b12) > 0,
      "上一手=510K, 手中有3333",
      f"炸弹数量: {len(bombs_b12)}")

# B13. PASS选项 - 无法压过时返回空列表
print("\n--- B13. PASS选项(无法压过) ---")
last_bomb_a = Hand([
    Card(14, 'spade'), Card(14, 'heart'), Card(14, 'club'), Card(14, 'diamond')
], HandType.BOMB)
hand_b13 = [Card(3, 'spade'), Card(7, 'heart')]  # 只有单张,无法压炸弹
actions_b13 = action_space.get_all_actions(hand_b13, last_bomb_a)
check("无法压过时返回空列表",
      len(actions_b13) == 0,
      "上一手=AAAA炸弹, 手中只有3和7",
      f"动作数: {len(actions_b13)}")


# ============================================================
# 规则引擎 can_beat 直接验证
# ============================================================
print("\n" + "=" * 60)
print("规则引擎 can_beat 直接验证")
print("=" * 60)

print("\n--- 510K压牌能力 ---")
h_510k = Hand([Card(5,'spade'), Card(10,'heart'), Card(13,'club')], HandType.FIVE_TEN_KING)
h_510k_pure = Hand([Card(5,'spade'), Card(10,'spade'), Card(13,'spade')], HandType.FIVE_TEN_KING)
h_single = Hand([Card(14,'spade')], HandType.SINGLE)
h_pair = Hand([Card(8,'spade'), Card(8,'heart')], HandType.PAIR)
h_bomb4 = Hand([Card(3,'spade'), Card(3,'heart'), Card(3,'club'), Card(3,'diamond')], HandType.BOMB)

check("510K能压单张A", rules.can_beat(h_510k, h_single))
check("510K能压对子88", rules.can_beat(h_510k, h_pair))
check("杂色510K不能压杂色510K", not rules.can_beat(h_510k, h_510k))
check("纯色510K能压杂色510K", rules.can_beat(h_510k_pure, h_510k))
check("杂色510K不能压纯色510K", not rules.can_beat(h_510k, h_510k_pure))
check("纯色510K不能压纯色510K", not rules.can_beat(h_510k_pure, h_510k_pure))
check("510K不能压炸弹", not rules.can_beat(h_510k, h_bomb4))
check("炸弹能压510K", rules.can_beat(h_bomb4, h_510k))
check("炸弹能压纯色510K", rules.can_beat(h_bomb4, h_510k_pure))

print("\n--- 炸弹大小比较 ---")
bomb_5_4 = Hand([Card(5,'s'), Card(5,'h'), Card(5,'c'), Card(5,'d')], HandType.BOMB)
bomb_8_4 = Hand([Card(8,'s'), Card(8,'h'), Card(8,'c'), Card(8,'d')], HandType.BOMB)
bomb_3_5 = Hand([Card(3,'s'), Card(3,'h'), Card(3,'c'), Card(3,'d'), Card(3,'s')], HandType.BOMB)

check("同张数炸弹rank大者胜", rules.can_beat(bomb_8_4, bomb_5_4))
check("同张数炸弹rank小者不能压", not rules.can_beat(bomb_5_4, bomb_8_4))
check("张数多的炸弹能压张数少的", rules.can_beat(bomb_3_5, bomb_5_4))

print("\n--- 连对张数约束 ---")
cp_34 = Hand([Card(3,'s'), Card(3,'h'), Card(4,'s'), Card(4,'h')], HandType.CONSECUTIVE_PAIRS)
cp_345 = Hand([Card(5,'s'), Card(5,'h'), Card(6,'s'), Card(6,'h'), Card(7,'s'), Card(7,'h')], HandType.CONSECUTIVE_PAIRS)
cp_56 = Hand([Card(5,'s'), Card(5,'h'), Card(6,'s'), Card(6,'h')], HandType.CONSECUTIVE_PAIRS)

check("同张数连对rank大者胜", rules.can_beat(cp_56, cp_34))
check("不同张数连对不能压", not rules.can_beat(cp_345, cp_34))


# ============================================================
# 最终统计
# ============================================================
print("\n" + "=" * 60)
total = passed + failed
print(f"测试完成: {passed}/{total} PASSED, {failed}/{total} FAILED")
print("=" * 60)

if failed > 0:
    sys.exit(1)
else:
    print("所有测试通过!")
    sys.exit(0)
