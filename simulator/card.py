"""
扑克牌类定义
Card class definition
"""

from typing import Optional
from config import Config

class Card:
    """扑克牌类"""

    def __init__(self, rank: int, suit: Optional[str] = None):
        """
        初始化扑克牌

        Args:
            rank: 点数 (3-15为普通牌, 16为小王, 17为大王)
            suit: 花色 (spade, heart, club, diamond), 王牌为None
        """
        self.rank = rank
        self.suit = suit

    def __repr__(self):
        """字符串表示"""
        if self.rank in [16, 17]:
            return f"{Config.RANK_NAMES[self.rank]}"
        return f"{Config.SUIT_SYMBOLS[self.suit]}{Config.RANK_NAMES[self.rank]}"

    def __str__(self):
        """友好的字符串表示"""
        return self.__repr__()

    def __eq__(self, other):
        """相等判断 - 仅基于rank, 花色在大多数情况下无意义"""
        if not isinstance(other, Card):
            return False
        return self.rank == other.rank

    def __hash__(self):
        """哈希值 - 仅基于rank, 同rank不同花色视为等价"""
        return hash(self.rank)

    def __lt__(self, other):
        """小于判断 (用于排序)"""
        return self.rank < other.rank

    def __gt__(self, other):
        """大于判断 (用于排序)"""
        return self.rank > other.rank

    def __le__(self, other):
        """小于等于判断"""
        return self.rank <= other.rank

    def __ge__(self, other):
        """大于等于判断"""
        return self.rank >= other.rank

    def get_score_value(self) -> int:
        """获取分数值"""
        return Config.SCORE_CARDS.get(self.rank, 0)

    def is_score_card(self) -> bool:
        """是否为分数牌"""
        return self.rank in Config.SCORE_CARDS

    def is_joker(self) -> bool:
        """是否为王牌"""
        return self.rank in [16, 17]

    def to_dict(self) -> dict:
        """转换为字典格式 (用于序列化)"""
        return {
            'rank': self.rank,
            'suit': self.suit
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'Card':
        """从字典创建Card对象"""
        return cls(data['rank'], data.get('suit'))

    def clone(self) -> 'Card':
        """克隆当前牌"""
        return Card(self.rank, self.suit)
