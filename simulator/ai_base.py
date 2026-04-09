"""
AI基类和简单AI实现
AI Base class and simple AI implementations
"""

import random
from typing import List, Optional
from card import Card
from rules import Hand
from action_space import ActionSpace
from game_engine import GameEngine

class AIBase:
    """AI基类"""

    def __init__(self, player_id: int, game_engine: GameEngine):
        self.player_id = player_id
        self.game_engine = game_engine
        self.action_space = ActionSpace()

    def decide_action(self, last_hand: Optional[Hand] = None) -> Optional[List[Card]]:
        """
        决定出牌动作

        Args:
            last_hand: 上一手牌 (None表示首个出牌)

        Returns:
            要出的牌 (None表示不要)
        """
        raise NotImplementedError

class RandomAI(AIBase):
    """随机AI - 随机选择合法动作"""

    def decide_action(self, last_hand: Optional[Hand] = None) -> Optional[List[Card]]:
        """随机选择合法动作"""
        player = self.game_engine.get_player(self.player_id)

        # 获取所有合法动作
        actions = self.action_space.get_all_actions(player.hand, last_hand)

        if not actions:
            return None  # 不要

        # 随机选择
        return random.choice(actions)

class GreedyAI(AIBase):
    """贪心AI - 优先出能抓分的牌"""

    def decide_action(self, last_hand: Optional[Hand] = None) -> Optional[List[Card]]:
        """贪心决策"""
        player = self.game_engine.get_player(self.player_id)

        # 获取所有合法动作
        actions = self.action_space.get_all_actions(player.hand, last_hand)

        if not actions:
            return None  # 不要

        # 如果是首个出牌,优先出分数少的牌
        if last_hand is None:
            # 按分数从小到大排序
            actions_with_score = []
            for action in actions:
                score = sum(card.get_score_value() for card in action)
                actions_with_score.append((score, action))

            actions_with_score.sort(key=lambda x: (x[0], -len(x[1])))  # 分数优先,张数次之

            return actions_with_score[0][1]
        else:
            # 需要压过,选择最小的能压过的
            actions_with_length = [(len(action), action) for action in actions]
            actions_with_length.sort(key=lambda x: (x[0], sum(card.rank for card in x[1])))

            return actions_with_length[0][1]

class SmartAI(AIBase):
    """智能AI - 结合多种策略"""

    def __init__(self, player_id: int, game_engine: GameEngine):
        super().__init__(player_id, game_engine)
        self.opponent_ids = self._get_opponent_ids()

    def _get_opponent_ids(self) -> List[int]:
        """获取对手ID列表"""
        teammate_id = self.game_engine.get_player(self.player_id).get_teammate_id()
        opponent_ids = [i for i in range(4) if i != self.player_id and i != teammate_id]
        return opponent_ids

    def decide_action(self, last_hand: Optional[Hand] = None) -> Optional[List[Card]]:
        """智能决策"""
        player = self.game_engine.get_player(self.player_id)

        # 获取所有合法动作
        actions = self.action_space.get_all_actions(player.hand, last_hand)

        if not actions:
            return None  # 不要

        # 评估每个动作
        best_action = None
        best_score = float('-inf')

        for action in actions:
            score = self._evaluate_action(action, last_hand)
            if score > best_score:
                best_score = score
                best_action = action

        return best_action

    def _evaluate_action(self, action: List[Card], last_hand: Optional[Hand]) -> float:
        """评估动作的分数"""
        score = 0

        # 1. 优先出非分数牌
        has_score_card = any(card.is_score_card() for card in action)
        if not has_score_card:
            score += 5

        # 2. 优先出大牌
        avg_rank = sum(card.rank for card in action) / len(action)
        score += avg_rank / 10

        # 3. 保留炸弹和特殊牌型
        # TODO: 实现更复杂的评估

        return score
