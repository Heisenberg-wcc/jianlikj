"""
玩家类 - 定义玩家属性和行为
Player class - defines player attributes and behaviors
"""

from typing import List, Optional
from card import Card

class Player:
    """玩家类"""

    def __init__(self, player_id: int, name: str = ""):
        """
        初始化玩家

        Args:
            player_id: 玩家ID (0-3)
            name: 玩家名称
        """
        self.player_id = player_id
        self.name = name if name else f"Player{player_id+1}"
        self.hand: List[Card] = []  # 手牌
        self.score = 0  # 当前抓分
        self.position: Optional[int] = None  # 出牌顺序(1=一游, 2=二游, 3=三游, 4=末游)
        self.is_dealer = False  # 是否为庄家
        self.teammate_id: Optional[int] = None  # 队友ID

    def add_card(self, card: Card):
        """添加一张牌到手牌"""
        self.hand.append(card)
        # 保持手牌排序
        self.hand.sort()

    def add_cards(self, cards: List[Card]):
        """添加多张牌到手牌"""
        self.hand.extend(cards)
        self.hand.sort()

    def remove_card(self, card: Card):
        """从手牌中移除一张牌"""
        if card in self.hand:
            self.hand.remove(card)

    def remove_cards(self, cards: List[Card]):
        """从手牌中移除多张牌"""
        for card in cards:
            if card in self.hand:
                self.hand.remove(card)

    def has_card(self, card: Card) -> bool:
        """检查是否持有某张牌"""
        return card in self.hand

    def get_hand_size(self) -> int:
        """获取手牌数量"""
        return len(self.hand)

    def is_out(self) -> bool:
        """是否已出完牌"""
        return len(self.hand) == 0

    def set_teammate(self, teammate_id: int):
        """设置队友"""
        self.teammate_id = teammate_id

    def get_teammate_id(self) -> Optional[int]:
        """获取队友ID"""
        return self.teammate_id

    def add_score(self, score: int):
        """增加抓分"""
        self.score += score

    def reset(self):
        """重置玩家状态"""
        self.hand = []
        self.score = 0
        self.position = None
        self.is_dealer = False

    def __repr__(self):
        """字符串表示"""
        teammate = f", Teammate: Player{self.teammate_id+1}" if self.teammate_id is not None else ""
        dealer = " (Dealer)" if self.is_dealer else ""
        return f"{self.name}{dealer}[Cards: {len(self.hand)}, Score: {self.score}{teammate}]"

    def __str__(self):
        """友好的字符串表示"""
        return self.__repr__()
