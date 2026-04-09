"""
牌堆类 - 处理发牌、洗牌等操作
Deck class - handles shuffling, dealing
"""

import random
from typing import List, Optional
from card import Card
from config import Config

class Deck:
    """牌堆类"""

    def __init__(self, num_decks: int = 2):
        """
        初始化牌堆

        Args:
            num_decks: 副数,默认2副牌
        """
        self.num_decks = num_decks
        self.cards: List[Card] = []
        self._initialize()

    def _initialize(self):
        """初始化牌堆"""
        self.cards = []

        # 添加每副牌
        for _ in range(self.num_decks):
            # 普通牌 (3-A, 2)
            for suit in Config.SUITS:
                for rank in range(3, 16):
                    self.cards.append(Card(rank, suit))

            # 王牌
            self.cards.append(Card(16, None))  # 小王
            self.cards.append(Card(17, None))  # 大王

    def shuffle(self):
        """洗牌"""
        random.shuffle(self.cards)

    def deal(self, num_players: int = 4) -> List[List[Card]]:
        """
        发牌

        Args:
            num_players: 玩家数量

        Returns:
            每个玩家的手牌列表
        """
        # 计算每人分到的牌数
        total_cards = len(self.cards)
        cards_per_player = total_cards // num_players

        # 分发手牌
        hands = []
        for i in range(num_players):
            start_idx = i * cards_per_player
            end_idx = start_idx + cards_per_player
            hand = self.cards[start_idx:end_idx]

            # 排序手牌 (按点数从小到大)
            hand.sort()

            hands.append(hand)

        return hands

    def remaining(self) -> int:
        """返回剩余牌数"""
        return len(self.cards)

    def draw(self) -> Optional[Card]:
        """抽取一张牌"""
        if self.cards:
            return self.cards.pop()
        return None

    def reset(self):
        """重置牌堆"""
        self._initialize()
