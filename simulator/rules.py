"""
规则引擎 - 牌型判断、合法性检查、大小比较
Rules Engine - hand type detection, validation, comparison
"""

import sys
import io
from typing import List, Optional, Dict, Tuple
from enum import Enum
from collections import Counter
from card import Card

class HandType(Enum):
    """牌型枚举"""
    SINGLE = "single"  # 单张
    PAIR = "pair"  # 对子
    CONSECUTIVE_PAIRS = "consecutive_pairs"  # 连对
    CONSECUTIVE_THREES = "consecutive_threes"  # 连三张
    BOMB = "bomb"  # 炸弹 (4+张相同)
    FIVE_TEN_KING = "510K"  # 开机 (510K) - 弱炸弹
    INVALID = "invalid"

class Hand:
    """手牌组合"""
    def __init__(self, cards: List[Card], hand_type: HandType):
        self.cards = cards
        self.hand_type = hand_type
        self.rank = self._calculate_rank()
        self.length = len(cards)
        # 仅510K时有意义: 三张牌是否花色相同(纯色)
        self.is_pure_suit = self._check_pure_suit()

    def _check_pure_suit(self) -> bool:
        """检查510K是否为纯色(三张牌花色相同)"""
        if self.hand_type != HandType.FIVE_TEN_KING:
            return False
        if not self.cards or len(self.cards) != 3:
            return False
        return all(c.suit is not None and c.suit == self.cards[0].suit for c in self.cards)

    def _calculate_rank(self) -> int:
        """计算牌型的主要点数(用于大小比较)"""
        if self.hand_type == HandType.SINGLE:
            return self.cards[0].rank
        elif self.hand_type == HandType.PAIR:
            return self.cards[0].rank
        elif self.hand_type == HandType.CONSECUTIVE_PAIRS:
            # 取最大的一对
            counter = Counter([c.rank for c in self.cards])
            return max(counter.keys())
        elif self.hand_type == HandType.CONSECUTIVE_THREES:
            # 取最大的三张
            counter = Counter([c.rank for c in self.cards])
            return max(counter.keys())
        elif self.hand_type == HandType.BOMB:
            return self.cards[0].rank
        elif self.hand_type == HandType.FIVE_TEN_KING:
            # 开机固定为K的rank (13)
            return 13
        return 0

    def __repr__(self):
        return f"{self.hand_type.value}({self.cards})"

class RulesEngine:
    """规则引擎"""

    # 牌型优先级 (数字越大越强)
    HAND_TYPE_PRIORITY = {
        HandType.SINGLE: 1,
        HandType.PAIR: 2,
        HandType.CONSECUTIVE_PAIRS: 3,
        HandType.CONSECUTIVE_THREES: 4,
        HandType.FIVE_TEN_KING: 5,
        HandType.BOMB: 6,
    }

    def __init__(self):
        pass

    def detect_hand_type(self, cards: List[Card]) -> HandType:
        """
        检测牌型

        Args:
            cards: 牌列表

        Returns:
            牌型
        """
        if not cards:
            return HandType.INVALID

        n = len(cards)
        ranks = [c.rank for c in cards]
        counter = Counter(ranks)

        # 检查开机 (510K)
        if self._is_five_ten_king(cards):
            return HandType.FIVE_TEN_KING

        # 检查炸弹
        if n >= 4 and len(counter) == 1:
            return HandType.BOMB

        # 检查连三张 (至少2组, 每组必须是3张)
        if n >= 6:
            counts = sorted(counter.values(), reverse=True)
            # 必须每组都是3张
            if len(counts) >= 2 and all(c == 3 for c in counts) and self._is_consecutive(list(counter.keys())):
                return HandType.CONSECUTIVE_THREES
                
        # 检查连对 (至少2组, 每组必须是2张)
        if n >= 4:
            counts = sorted(counter.values(), reverse=True)
            # 必须每组都是2张
            if len(counts) >= 2 and all(c == 2 for c in counts) and self._is_consecutive(list(counter.keys())):
                return HandType.CONSECUTIVE_PAIRS

        # 检查对子
        if n == 2 and len(counter) == 1:
            return HandType.PAIR

        # 检查单张
        if n == 1:
            return HandType.SINGLE

        return HandType.INVALID

    def _is_five_ten_king(self, cards: List[Card]) -> bool:
        """检查是否为开机 (510K)"""
        if len(cards) != 3:
            return False

        ranks = sorted([c.rank for c in cards])
        # 需要有5(5), 10(10), K(13)
        return ranks == [5, 10, 13]

    @staticmethod
    def is_pure_suit_510k(cards: List[Card]) -> bool:
        """检查510K的三张牌是否花色相同(纯色)"""
        if len(cards) != 3:
            return False
        return all(c.suit is not None and c.suit == cards[0].suit for c in cards)

    def _is_consecutive(self, ranks: List[int]) -> bool:
        """检查是否为连续序列"""
        if len(ranks) < 2:
            return True

        sorted_ranks = sorted(ranks)
        # 2(15)不能参与连牌
        if 15 in sorted_ranks:
            return False

        for i in range(1, len(sorted_ranks)):
            if sorted_ranks[i] - sorted_ranks[i-1] != 1:
                return False

        return True

    def is_valid_move(self, cards: List[Card], last_hand: Optional[Hand] = None) -> bool:
        """
        检查出牌是否合法

        Args:
            cards: 要出的牌
            last_hand: 上一轮出的牌 (None表示首个出牌)

        Returns:
            是否合法
        """
        # 检测牌型
        hand_type = self.detect_hand_type(cards)

        if hand_type == HandType.INVALID:
            return False

        # 如果是第一个出牌,只要牌型合法即可
        if last_hand is None:
            return True

        current_hand = Hand(cards, hand_type)

        # 比较牌型
        return self.can_beat(current_hand, last_hand)

    # 普通牌型集合 (可被510K和炸弹压过)
    NORMAL_HAND_TYPES = {
        HandType.SINGLE,
        HandType.PAIR,
        HandType.CONSECUTIVE_PAIRS,
        HandType.CONSECUTIVE_THREES,
    }

    def can_beat(self, current: Hand, last: Hand) -> bool:
        """
        检查当前手牌是否能压过上一手牌

        压牌规则:
        - 炸弹可压过任意非炸弹牌型(含开机510K)
        - 开机(510K)可压过任意普通牌型(单张/对子/连对/连三张),但不能压过炸弹
        - 纯色开机(510K)可压杂色开机, 同类型510K不可互压
        - 同类型牌比大小(连对/连三张需张数相同)
        - 炸弹之间张数多者大,张数相同比点数

        Args:
            current: 当前手牌
            last: 上一手牌

        Returns:
            是否能压过
        """
        # 炸弹可以压过任意非炸弹 (含开机510K和普通牌型)
        if current.hand_type == HandType.BOMB and last.hand_type != HandType.BOMB:
            return True

        # 开机(510K)可以压过普通牌型 (弱炸弹)
        if current.hand_type == HandType.FIVE_TEN_KING and last.hand_type in self.NORMAL_HAND_TYPES:
            return True

        # 同类型比较
        if current.hand_type == last.hand_type:
            # 开机(510K): 纯色可压杂色, 同类型不可互压
            if current.hand_type == HandType.FIVE_TEN_KING:
                # 纯色510K可以压过杂色510K
                if current.is_pure_suit and not last.is_pure_suit:
                    return True
                # 其余情况不可互压 (纯vs纯, 杂vs杂, 杂vs纯)
                return False

            # 连对和连三张需要张数相同
            if current.hand_type in [HandType.CONSECUTIVE_PAIRS, HandType.CONSECUTIVE_THREES]:
                if current.length != last.length:
                    return False

            # 炸弹: 张数多的大
            if current.hand_type == HandType.BOMB:
                if current.length > last.length:
                    return True
                elif current.length < last.length:
                    return False

            # 比较点数
            return current.rank > last.rank

        # 其他情况不能压
        return False

    def get_all_valid_moves(self, hand: List[Card], last_hand: Optional[Hand] = None) -> List[List[Card]]:
        """
        获取所有合法的出牌组合

        Args:
            hand: 手牌
            last_hand: 上一手牌 (None表示首个出牌)

        Returns:
            所有合法的出牌组合列表
        """
        valid_moves = []

        # 如果是第一个出牌,可以出任意合法牌型
        if last_hand is None:
            # TODO: 生成所有可能的组合 (在action_space.py中实现)
            pass
        else:
            # 需要压过上一手牌
            # TODO: 生成能压过last_hand的组合 (在action_space.py中实现)
            pass

        return valid_moves

    def calculate_hand_score(self, cards: List[Card]) -> int:
        """
        计算手牌的分数

        Args:
            cards: 牌列表

        Returns:
            总分
        """
        return sum(card.get_score_value() for card in cards)

# 测试代码
if __name__ == "__main__":
    print("=== 测试规则引擎 ===")

    from card import Card

    rules = RulesEngine()

    # 测试单张
    single_card = Card(3, 'spade')
    print(f"单张: {single_card} -> {rules.detect_hand_type([single_card])}")

    # 测试对子
    pair_cards = [Card(3, 'spade'), Card(3, 'heart')]
    print(f"对子: {pair_cards} -> {rules.detect_hand_type(pair_cards)}")

    # 测试连对
    consecutive_pairs = [
        Card(3, 'spade'), Card(3, 'heart'),
        Card(4, 'spade'), Card(4, 'heart')
    ]
    print(f"连对: {consecutive_pairs} -> {rules.detect_hand_type(consecutive_pairs)}")

    # 测试连三张
    consecutive_threes = [
        Card(3, 'spade'), Card(3, 'heart'), Card(3, 'club'),
        Card(4, 'spade'), Card(4, 'heart'), Card(4, 'club')
    ]
    print(f"连三张: {consecutive_threes} -> {rules.detect_hand_type(consecutive_threes)}")

    # 测试炸弹
    bomb_cards = [
        Card(5, 'spade'), Card(5, 'heart'),
        Card(5, 'club'), Card(5, 'diamond')
    ]
    print(f"炸弹: {bomb_cards} -> {rules.detect_hand_type(bomb_cards)}")

    # 测试开机 (510K)
    five_ten_king = [
        Card(5, 'spade'), Card(10, 'heart'), Card(13, 'club')
    ]
    print(f"开机(510K): {five_ten_king} -> {rules.detect_hand_type(five_ten_king)}")

    # 测试纯色510K
    pure_510k = [
        Card(5, 'spade'), Card(10, 'spade'), Card(13, 'spade')
    ]
    print(f"纯色510K: {pure_510k} -> {rules.detect_hand_type(pure_510k)}")
    h_pure = Hand(pure_510k, HandType.FIVE_TEN_KING)
    print(f"  is_pure_suit={h_pure.is_pure_suit}")

    h_mixed = Hand(five_ten_king, HandType.FIVE_TEN_KING)
    print(f"杂色510K is_pure_suit={h_mixed.is_pure_suit}")

    # 测试大小比较
    print("\n=== 大小比较 ===")
    hand1 = Hand(pair_cards, HandType.PAIR)
    hand2 = Hand([Card(4, 'spade'), Card(4, 'heart')], HandType.PAIR)

    print(f"{hand1} vs {hand2} -> {rules.can_beat(hand2, hand1)}")

    hand_bomb = Hand(bomb_cards, HandType.BOMB)
    print(f"{hand1} vs {hand_bomb} -> {rules.can_beat(hand_bomb, hand1)}")
