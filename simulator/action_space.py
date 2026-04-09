"""
动作空间 - 生成所有合法的出牌组合
Action Space - generates all valid card combinations
"""

from typing import List, Set, Tuple, Optional
from itertools import combinations
from collections import Counter, defaultdict
from card import Card
from rules import RulesEngine, Hand, HandType

class ActionSpace:
    """动作空间 - 生成合法出牌组合"""

    def __init__(self):
        self.rules = RulesEngine()

    def get_all_actions(self, hand: List[Card], last_hand: Optional[Hand] = None) -> List[List[Card]]:
        """
        获取所有合法的出牌动作

        Args:
            hand: 手牌
            last_hand: 上一手牌 (None表示首个出牌)

        Returns:
            所有合法的出牌组合列表
        """
        # 添加"不要"选项
        actions = []

        if last_hand is None:
            # 首个出牌,可以出任意合法牌型
            actions.extend(self._generate_all_possible_hands(hand))
        else:
            # 需要压过上一手牌
            actions.extend(self._generate_beating_hands(hand, last_hand))

        # 去重
        actions = self._deduplicate_actions(actions)

        return actions

    def _generate_all_possible_hands(self, hand: List[Card]) -> List[List[Card]]:
        """
        生成所有可能的合法牌型
        简化: 非510K牌型每个rank只生成一种组合(rank等价)

        Args:
            hand: 手牌

        Returns:
            所有合法的牌型组合
        """
        actions = []

        # 统计手牌
        counter = Counter([c.rank for c in hand])
        rank_to_cards = defaultdict(list)
        for card in hand:
            rank_to_cards[card.rank].append(card)

        # 1. 单张 - 每个rank只生成一个
        for rank in counter:
            actions.append([rank_to_cards[rank][0]])

        # 2. 对子 - 每个rank只生成一个(取前2张)
        for rank in counter:
            if counter[rank] >= 2:
                actions.append(rank_to_cards[rank][:2])

        # 3. 连对 (至少2组)
        actions.extend(self._generate_consecutive_pairs(counter, rank_to_cards))

        # 4. 连三张 (至少2组)
        actions.extend(self._generate_consecutive_threes(counter, rank_to_cards))

        # 5. 炸弹 (4+张相同)
        actions.extend(self._generate_bombs(counter, rank_to_cards))

        # 6. 开机 (510K) - 保留花色区分(纯色+杂色)
        actions.extend(self._generate_five_ten_king(hand))

        # 过滤非法牌型并去重
        valid_actions = []
        seen = set()
        for action in actions:
            hand_type = self.rules.detect_hand_type(action)
            if hand_type != HandType.INVALID:
                # 510K需区分纯色/杂色, 纯色还需区分花色
                if hand_type == HandType.FIVE_TEN_KING:
                    is_pure = all(c.suit == action[0].suit for c in action)
                    if is_pure:
                        # 每种花色的纯色510K各保留一个
                        key = (tuple(sorted([c.rank for c in action])), True, action[0].suit)
                    else:
                        # 杂色510K只保留一个
                        key = (tuple(sorted([c.rank for c in action])), False)
                else:
                    key = tuple(sorted([c.rank for c in action]))
                if key not in seen:
                    seen.add(key)
                    valid_actions.append(action)

        return valid_actions

    def _generate_consecutive_pairs(self, counter: Counter, rank_to_cards: dict) -> List[List[Card]]:
        """生成所有连对 - 简化: 每个rank只取前2张"""
        actions = []

        # 找出所有至少有2张的rank
        pairs_ranks = sorted([rank for rank in counter if counter[rank] >= 2])

        # 2不能参与连对
        pairs_ranks = [rank for rank in pairs_ranks if rank != 15]

        if len(pairs_ranks) < 2:
            return actions

        # 生成不同长度的连对
        for length in range(2, len(pairs_ranks) + 1):
            for start_idx in range(len(pairs_ranks) - length + 1):
                end_idx = start_idx + length
                selected_ranks = pairs_ranks[start_idx:end_idx]

                # 检查是否连续
                if self._is_consecutive(selected_ranks):
                    # 每个rank只取前2张
                    cards = []
                    for rank in selected_ranks:
                        cards.extend(rank_to_cards[rank][:2])
                    actions.append(cards)

        return actions

    def _generate_consecutive_pairs_combinations(
        self,
        ranks: List[int],
        rank_to_cards: dict,
        idx: int,
        current: List[Card],
        actions: List[List[Card]],
        target_length: int
    ):
        """递归生成连对的所有组合"""
        if idx >= len(ranks):
            if len(current) == target_length:
                # 检查是否已经存在
                cards_tuple = tuple(sorted([str(c) for c in current]))
                if not any(all(c in action for c in current) for action in actions if len(action) == len(current)):
                    actions.append(current[:])
            return

        rank = ranks[idx]
        cards_list = rank_to_cards[rank]

        # 生成该rank的所有2张组合
        for pair in combinations(cards_list, 2):
            current.extend(pair)
            self._generate_consecutive_pairs_combinations(
                ranks, rank_to_cards, idx + 1, current, actions, target_length
            )
            # 回溯
            current[-2:] = []

    def _generate_consecutive_threes(self, counter: Counter, rank_to_cards: dict) -> List[List[Card]]:
        """生成所有连三张 - 简化: 每个rank只取前3张"""
        actions = []

        # 找出所有至少有3张的rank
        threes_ranks = sorted([rank for rank in counter if counter[rank] >= 3])

        # 2不能参与连三张
        threes_ranks = [rank for rank in threes_ranks if rank != 15]

        if len(threes_ranks) < 2:
            return actions

        # 生成不同长度的连三张
        for length in range(2, len(threes_ranks) + 1):
            for start_idx in range(len(threes_ranks) - length + 1):
                end_idx = start_idx + length
                selected_ranks = threes_ranks[start_idx:end_idx]

                # 检查是否连续
                if self._is_consecutive(selected_ranks):
                    # 每个rank只取前3张
                    cards = []
                    for rank in selected_ranks:
                        cards.extend(rank_to_cards[rank][:3])
                    actions.append(cards)

        return actions

    def _generate_consecutive_threes_combinations(
        self,
        ranks: List[int],
        rank_to_cards: dict,
        idx: int,
        current: List[Card],
        actions: List[List[Card]],
        target_length: int
    ):
        """递归生成连三张的所有组合"""
        if idx >= len(ranks):
            if len(current) == target_length:
                actions.append(current[:])
            return

        rank = ranks[idx]
        cards_list = rank_to_cards[rank]

        # 生成该rank的所有3张组合
        for triple in combinations(cards_list, 3):
            current.extend(triple)
            self._generate_consecutive_threes_combinations(
                ranks, rank_to_cards, idx + 1, current, actions, target_length
            )
            # 回溯
            current[-3:] = []

    def _generate_bombs(self, counter: Counter, rank_to_cards: dict) -> List[List[Card]]:
        """生成所有炸弹 - 简化: 同rank同张数只生成一个"""
        actions = []

        for rank in counter:
            count = counter[rank]
            if count >= 4:
                # 生成4, 5, 6...张的炸弹, 每种张数只取前N张
                for bomb_size in range(4, count + 1):
                    actions.append(rank_to_cards[rank][:bomb_size])

        return actions

    def _generate_five_ten_king(self, hand: List[Card]) -> List[List[Card]]:
        """生成开机(510K)组合 - 区分纯色和杂色

        纯色510K: 三张牌花色相同, 每种花色最多1个
        杂色510K: 花色不全相同, 只生成一个代表动作
        """
        actions = []

        # 找出所有5, 10, K
        fives = [c for c in hand if c.rank == 5]
        tens = [c for c in hand if c.rank == 10]
        kings = [c for c in hand if c.rank == 13]

        if not (fives and tens and kings):
            return actions

        # 1. 生成纯色510K (每种花色最多1个)
        for suit in ['spade', 'heart', 'club', 'diamond']:
            five = next((c for c in fives if c.suit == suit), None)
            ten = next((c for c in tens if c.suit == suit), None)
            king = next((c for c in kings if c.suit == suit), None)
            if five and ten and king:
                actions.append([five, ten, king])

        # 2. 生成一个杂色510K代表(花色不全相同)
        for five in fives:
            for ten in tens:
                for king in kings:
                    if not (five.suit == ten.suit == king.suit):
                        actions.append([five, ten, king])
                        return actions  # 找到一个杂色即返回

        return actions

    def _generate_beating_hands(self, hand: List[Card], last_hand: Hand) -> List[List[Card]]:
        """
        生成能压过last_hand的所有组合

        Args:
            hand: 手牌
            last_hand: 上一手牌

        Returns:
            能压过的所有组合
        """
        actions = []

        # 1. 生成炸弹 (炸弹可以压过任何非炸弹牌型,含开机510K)
        if last_hand.hand_type != HandType.BOMB:
            bombs = self._generate_bombs(
                Counter([c.rank for c in hand]),
                self._rank_to_cards(hand)
            )
            for bomb in bombs:
                actions.append(bomb)

        # 2. 生成开机510K
        if last_hand.hand_type in self.rules.NORMAL_HAND_TYPES:
            # 510K可以压过普通牌型: 单张/对子/连对/连三张
            five_ten_king = self._generate_five_ten_king(hand)
            actions.extend(five_ten_king)
        elif last_hand.hand_type == HandType.FIVE_TEN_KING and not last_hand.is_pure_suit:
            # 纯色510K可以压过杂色510K
            all_510k = self._generate_five_ten_king(hand)
            pure_510k = [a for a in all_510k if self.rules.is_pure_suit_510k(a)]
            actions.extend(pure_510k)

        # 3. 同类型,更大
        if last_hand.hand_type == HandType.SINGLE:
            actions.extend(self._generate_beating_singles(hand, last_hand))
        elif last_hand.hand_type == HandType.PAIR:
            actions.extend(self._generate_beating_pairs(hand, last_hand))
        elif last_hand.hand_type == HandType.CONSECUTIVE_PAIRS:
            actions.extend(self._generate_beating_consecutive_pairs(hand, last_hand))
        elif last_hand.hand_type == HandType.CONSECUTIVE_THREES:
            actions.extend(self._generate_beating_consecutive_threes(hand, last_hand))
        elif last_hand.hand_type == HandType.BOMB:
            actions.extend(self._generate_beating_bombs(hand, last_hand))

        return actions

    def _generate_beating_singles(self, hand: List[Card], last_hand: Hand) -> List[List[Card]]:
        """生成能压过的单张 - 每个rank只一个"""
        actions = []
        seen_ranks = set()
        for card in hand:
            if card.rank > last_hand.rank and card.rank not in seen_ranks:
                actions.append([card])
                seen_ranks.add(card.rank)
        return actions

    def _generate_beating_pairs(self, hand: List[Card], last_hand: Hand) -> List[List[Card]]:
        """生成能压过的对子 - 每个rank只一个"""
        actions = []
        counter = Counter([c.rank for c in hand])
        rank_to_cards = self._rank_to_cards(hand)

        for rank in counter:
            if counter[rank] >= 2 and rank > last_hand.rank:
                actions.append(rank_to_cards[rank][:2])

        return actions

    def _generate_beating_consecutive_pairs(self, hand: List[Card], last_hand: Hand) -> List[List[Card]]:
        """生成能压过的连对 - 每个rank只取前2张"""
        actions = []
        length = last_hand.length // 2  # 对子的数量
        counter = Counter([c.rank for c in hand])
        rank_to_cards = self._rank_to_cards(hand)

        # 找出所有至少有2张的rank
        pairs_ranks = sorted([rank for rank in counter if counter[rank] >= 2])
        pairs_ranks = [rank for rank in pairs_ranks if rank != 15]  # 2不能参与连对

        if len(pairs_ranks) < length:
            return actions

        for start_idx in range(len(pairs_ranks) - length + 1):
            end_idx = start_idx + length
            selected_ranks = pairs_ranks[start_idx:end_idx]

            if self._is_consecutive(selected_ranks) and max(selected_ranks) > last_hand.rank:
                # 每个rank只取前2张
                cards = []
                for rank in selected_ranks:
                    cards.extend(rank_to_cards[rank][:2])
                actions.append(cards)

        return actions

    def _generate_beating_consecutive_threes(self, hand: List[Card], last_hand: Hand) -> List[List[Card]]:
        """生成能压过的连三张 - 每个rank只取前3张"""
        actions = []
        length = last_hand.length // 3  # 三张的数量
        counter = Counter([c.rank for c in hand])
        rank_to_cards = self._rank_to_cards(hand)

        # 找出所有至少有3张的rank
        threes_ranks = sorted([rank for rank in counter if counter[rank] >= 3])
        threes_ranks = [rank for rank in threes_ranks if rank != 15]  # 2不能参与连三张

        if len(threes_ranks) < length:
            return actions

        for start_idx in range(len(threes_ranks) - length + 1):
            end_idx = start_idx + length
            selected_ranks = threes_ranks[start_idx:end_idx]

            if self._is_consecutive(selected_ranks) and max(selected_ranks) > last_hand.rank:
                # 每个rank只取前3张
                cards = []
                for rank in selected_ranks:
                    cards.extend(rank_to_cards[rank][:3])
                actions.append(cards)

        return actions

    def _generate_beating_bombs(self, hand: List[Card], last_hand: Hand) -> List[List[Card]]:
        """生成能压过的炸弹 - 同rank同张数只生成一个"""
        actions = []
        counter = Counter([c.rank for c in hand])
        rank_to_cards = self._rank_to_cards(hand)

        last_bomb_length = last_hand.length

        for rank in counter:
            count = counter[rank]
            if count >= 4:
                for bomb_size in range(4, count + 1):
                    if bomb_size > last_bomb_length or (bomb_size == last_bomb_length and rank > last_hand.rank):
                        actions.append(rank_to_cards[rank][:bomb_size])

        return actions

    def _is_consecutive(self, ranks: List[int]) -> bool:
        """检查是否连续"""
        if len(ranks) < 2:
            return True
        for i in range(1, len(ranks)):
            if ranks[i] - ranks[i-1] != 1:
                return False
        return True

    def _rank_to_cards(self, hand: List[Card]) -> dict:
        """将手牌按rank分组"""
        rank_to_cards = defaultdict(list)
        for card in hand:
            rank_to_cards[card.rank].append(card)
        return rank_to_cards

    def _deduplicate_actions(self, actions: List[List[Card]]) -> List[List[Card]]:
        """去重 - 基于rank去重, 510K区分纯色(含花色)/杂色"""
        seen = set()
        unique_actions = []

        for action in actions:
            hand_type = self.rules.detect_hand_type(action)
            if hand_type == HandType.FIVE_TEN_KING:
                is_pure = self.rules.is_pure_suit_510k(action)
                if is_pure:
                    key = (tuple(sorted([c.rank for c in action])), True, action[0].suit)
                else:
                    key = (tuple(sorted([c.rank for c in action])), False)
            else:
                key = tuple(sorted([c.rank for c in action]))
            if key not in seen:
                seen.add(key)
                unique_actions.append(action)

        return unique_actions

# 测试代码
if __name__ == "__main__":
    print("=== 测试动作空间 ===")

    from card import Card

    action_space = ActionSpace()

    # 测试手牌
    test_hand = [
        Card(3, 'spade'), Card(3, 'heart'),
        Card(4, 'spade'), Card(4, 'heart'),
        Card(5, 'spade'), Card(5, 'heart'),
        Card(10, 'spade'), Card(13, 'spade'),
        Card(5, 'club'), Card(5, 'diamond')
    ]

    print(f"\n测试手牌: {[str(c) for c in test_hand]}")

    # 获取所有动作
    actions = action_space.get_all_actions(test_hand)
    print(f"\n所有合法动作数量: {len(actions)}")

    # 显示前10个动作
    print("\n前10个动作:")
    for i, action in enumerate(actions[:10]):
        hand_type = action_space.rules.detect_hand_type(action)
        print(f"{i+1}. {action} ({hand_type.value})")

    # 测试压过指定牌型
    print("\n=== 测试压过牌型 ===")
    last_hand = Hand([Card(3, 'spade'), Card(3, 'heart')], HandType.PAIR)
    beating_actions = action_space.get_all_actions(test_hand, last_hand)
    print(f"能压过 {last_hand} 的动作数量: {len(beating_actions)}")
    for action in beating_actions:
        print(f"  {action}")
