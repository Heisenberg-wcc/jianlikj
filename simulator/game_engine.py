"""
游戏引擎 - 控制游戏流程
Game Engine - controls game flow
"""

import sys
import io
from typing import List, Tuple, Optional, Dict
from card import Card
from deck import Deck
from player import Player
from config import Config
from rules import RulesEngine, Hand, HandType

class GameState:
    """游戏状态"""

    def __init__(self):
        self.current_player_idx = 0  # 当前出牌玩家
        self.current_round_cards: List[Card] = []  # 当前轮已出的牌
        self.current_round_players: List[int] = []  # 当前轮出牌的玩家ID
        self.last_played_hand: Optional[Hand] = None  # 上一手有效出牌 (None表示主动出牌)
        self.last_played_player: Optional[int] = None  # 上一手出牌的玩家ID
        self.last_round_winner: Optional[int] = None  # 上一轮获胜玩家
        self.round_count = 0  # 当前轮次
        self.game_over = False  # 游戏是否结束
        self.history: List[Dict] = []  # 历史记录
        self.team_scores = [0, 0]  # 队伍得分 [team0, team1]
        self.round_just_reset = False  # 轮次刚刚重置标志位，供外部查询
        
        # 【新增】记牌器和游戏进程追踪
        self.played_cards_history: List[Card] = []  # 本局已出的所有牌（记牌器）
        self.pass_counts: List[int] = [0, 0, 0, 0]  # 各玩家连续PASS次数
        self.total_step = 0  # 游戏总步数（用于encode_state的step参数）

class GameEngine:
    """游戏引擎"""

    def __init__(self):
        """初始化游戏引擎"""
        self.deck = Deck(num_decks=Config.NUM_DECKS)
        self.players: List[Player] = [
            Player(0, "Player1"),
            Player(1, "Player2"),
            Player(2, "Player3"),
            Player(3, "Player4"),
        ]
        self.state = GameState()
        self.rules = RulesEngine()  # 规则引擎
        self.finished_order: List[int] = []  # 出完牌的玩家顺序

    def initialize(self):
        """初始化游戏 - 发牌和确定队友"""
        # 洗牌
        self.deck.shuffle()

        # 发牌
        hands = self.deck.deal(num_players=Config.NUM_PLAYERS)
        for i, hand in enumerate(hands):
            self.players[i].add_cards(hand)

        # 确定庄家和队友 (第一局)
        self._determine_dealer_and_teams()

    def _determine_dealer_and_teams(self):
        """
        确定庄家和队友
        (简化实现: 随机选择翻牌)
        """
        # 随机选择一张翻牌
        import random
        dealer_idx = random.randint(0, 3)
        self.players[dealer_idx].is_dealer = True

        # 获取翻牌的点数
        flip_card = self.players[dealer_idx].hand[0]
        flip_rank = flip_card.rank

        # 寻找持有同点数牌的玩家 (简化: 默认对面为队友)
        # 队伍: [0, 2] vs [1, 3]
        self.players[0].set_teammate(2)
        self.players[2].set_teammate(0)
        self.players[1].set_teammate(3)
        self.players[3].set_teammate(1)

        # 注释掉 print 以避免编码问题
        # print(f"Dealer: Player{dealer_idx+1}, Flip Card: {flip_card}")
        # print(f"Teams: [Player1, Player3] vs [Player2, Player4]")

    def play_card(self, player_idx: int, cards: List[Card]) -> bool:
        """
        玩家出牌

        Args:
            player_idx: 玩家ID
            cards: 要出的牌

        Returns:
            是否出牌成功
        """
        # 检查玩家是否持有这些牌
        player = self.players[player_idx]
        for card in cards:
            if not player.has_card(card):
                return False

        # 检查出牌是否合法 (规则引擎校验)
        if not self.rules.is_valid_move(cards, self.state.last_played_hand):
            return False

        # 计算当前出牌的牌型并更新 last_played_hand
        hand_type = self.rules.detect_hand_type(cards)
        self.state.last_played_hand = Hand(cards, hand_type)
        self.state.last_played_player = player_idx

        # 执行出牌
        player.remove_cards(cards)
        self.state.current_round_cards.extend(cards)
        self.state.current_round_players.append(player_idx)
        
        # 【新增】更新记牌器和步数
        self.state.played_cards_history.extend(cards)
        self.state.pass_counts[player_idx] = 0  # 出牌则重置PASS计数
        self.state.total_step += 1

        # 记录历史
        self._log_event("play_card", {
            "player": player_idx,
            "cards": [c.to_dict() for c in cards],
            "round": self.state.round_count
        })

        # 检查玩家是否出完牌
        if player.is_out():
            self.finished_order.append(player_idx)
            player.position = len(self.finished_order)
            # print(f"Player{player_idx+1} is out! Position: {player.position}")

            # 检查游戏是否结束 (当至少有一方两人出完)
            self._check_game_over()
        
        # 【关键修复】切换到下一个玩家
        if not self.state.game_over:
            self.state.current_player_idx = (player_idx + 1) % 4

        return True

    def pass_round(self, player_idx: int):
        """
        玩家跳过本轮 (不要)

        Args:
            player_idx: 玩家ID
        """
        self.state.current_round_players.append(player_idx)
        
        # 【新增】增加PASS计数和总步数
        self.state.pass_counts[player_idx] += 1
        self.state.total_step += 1
        
        # 检查是否其他三人都 pass 了 (当前出牌者之后3人都pass)
        # 统计从上一次出牌后连续 pass 的人数
        pass_count = 0
        for i in range(len(self.state.current_round_players) - 1, -1, -1):
            pid = self.state.current_round_players[i]
            # 检查这个玩家是否是 pass (不是最后出牌的人)
            if pid == self.state.last_played_player:
                break
            pass_count += 1
        
        # 如果其余三人都 pass，新一轮开始
        if pass_count >= 3 and self.state.last_played_player is not None:
            # 【修复】轮次得分结算：清空前计算本轮分值牌总分
            round_score = sum(card.get_score_value() for card in self.state.current_round_cards)
            if round_score > 0:
                winner_idx = self.state.last_played_player
                team_idx = 0 if winner_idx in [0, 2] else 1
                self.state.team_scores[team_idx] += round_score

            # 清空当前轮次，最后出牌者重新主动出牌
            self.state.current_round_cards = []
            self.state.current_round_players = []
            self.state.last_played_hand = None  # 重置为主动出牌
            self.state.last_played_player = None
            self.state.round_count += 1
            self.state.round_just_reset = True  # 通知外部轮次已重置
        
        # 记录历史
        self._log_event("pass", {
            "player": player_idx,
            "round": self.state.round_count
        })
        
        # 【关键修复】切换到下一个玩家
        if not self.state.game_over:
            self.state.current_player_idx = (player_idx + 1) % 4

    def resolve_round(self) -> int:
        """
        结算当前轮次,返回获胜玩家

        Returns:
            获胜玩家ID
        """
        # 如果只有一个人出牌,该玩家获胜
        if len(self.state.current_round_players) == 1:
            winner_idx = self.state.current_round_players[0]
        else:
            # TODO: 在rules.py中实现牌型比较逻辑
            # 简化: 假设最后一个出牌的玩家获胜
            winner_idx = self.state.current_round_players[-1]

        # 计算本轮得分
        round_score = sum(card.get_score_value() for card in self.state.current_round_cards)

        # 获胜玩家获得分数
        self.players[winner_idx].add_score(round_score)

        # 更新队伍得分
        team_idx = 0 if winner_idx in [0, 2] else 1
        self.state.team_scores[team_idx] += round_score

        # 记录
        self._log_event("round_end", {
            "winner": winner_idx,
            "score": round_score,
            "team_scores": self.state.team_scores,
            "round": self.state.round_count
        })

        # 清空当前轮次
        self.state.current_round_cards = []
        self.state.current_round_players = []
        self.state.last_played_hand = None  # 重置为主动出牌
        self.state.last_played_player = None
        self.state.last_round_winner = winner_idx
        self.state.round_count += 1
        self.state.current_player_idx = winner_idx

        # print(f"Round {self.state.round_count} winner: Player{winner_idx+1}, Score: {round_score}")

        return winner_idx

    def _check_game_over(self):
        """检查游戏是否结束"""
        # 统计已出完牌的玩家
        out_players = len(self.finished_order)

        # 如果至少有一方两人出完牌,游戏结束
        if out_players >= 2:
            # 检查是否同一队两人都出完了
            if (0 in self.finished_order and 2 in self.finished_order) or \
               (1 in self.finished_order and 3 in self.finished_order):
                self._calculate_final_scores()
                self.state.game_over = True

    def _calculate_final_scores(self):
        """计算最终得分"""
        # 【修复】剩余手牌中的分值牌计入对方队伍得分
        for i, player in enumerate(self.players):
            if len(player.hand) > 0:  # 未出完牌的玩家
                remaining_score = sum(card.get_score_value() for card in player.hand)
                if remaining_score > 0:
                    my_team = 0 if i in [0, 2] else 1
                    opponent_team = 1 - my_team
                    self.state.team_scores[opponent_team] += remaining_score

        # 头游/末游奖励（双向转移）
        if len(self.finished_order) >= 2:
            # 情况1: 同队前两名
            first = self.finished_order[0]
            second = self.finished_order[1]
            first_team = 0 if first in [0, 2] else 1
            second_team = 0 if second in [0, 2] else 1

            if first_team == second_team:
                # 【修复】双向转移：己方+60，对方-60
                self.state.team_scores[first_team] += Config.REWARD_1ST_2ND
                self.state.team_scores[1 - first_team] -= Config.REWARD_1ST_2ND
            elif len(self.finished_order) >= 3:
                # 情况2: 一游+三游同队
                third = self.finished_order[2]
                third_team = 0 if third in [0, 2] else 1

                if first_team == third_team:
                    # 【修复】双向转移：己方+30，对方-30
                    self.state.team_scores[first_team] += Config.REWARD_1ST_3RD
                    self.state.team_scores[1 - first_team] -= Config.REWARD_1ST_3RD

        # 记录最终结果
        self._log_event("game_over", {
            "finished_order": self.finished_order,
            "team_scores": self.state.team_scores,
            "winner": 0 if self.state.team_scores[0] > self.state.team_scores[1] else 1
        })

    def _log_event(self, event_type: str, data: Dict):
        """记录事件到历史"""
        self.state.history.append({
            "type": event_type,
            "data": data
        })

    def get_current_player(self) -> Player:
        """获取当前玩家"""
        return self.players[self.state.current_player_idx]

    def get_player(self, player_idx: int) -> Player:
        """获取指定玩家"""
        return self.players[player_idx]

    def get_game_state(self) -> GameState:
        """获取游戏状态"""
        return self.state

    def reset(self):
        """重置游戏"""
        self.deck.reset()
        for player in self.players:
            player.reset()
        self.state = GameState()
        self.finished_order = []
        
    def get_played_cards_history(self) -> List[Card]:
        """获取已出牌历史（记牌器）"""
        return self.state.played_cards_history
    
    def get_pass_counts(self) -> List[int]:
        """获取各玩家连续PASS次数"""
        return self.state.pass_counts
    
    def get_total_step(self) -> int:
        """获取游戏总步数"""
        return self.state.total_step

# 测试代码
if __name__ == "__main__":
    print("=== 测试游戏引擎 ===")
    game = GameEngine()
    game.initialize()

    print("\n=== 玩家手牌 ===")
    for i, player in enumerate(game.players):
        print(f"Player{i+1}: {player.hand}")

    print("\n=== 队友关系 ===")
    for i, player in enumerate(game.players):
        print(f"Player{i+1} -> Teammate: Player{player.get_teammate_id()+1}")

    print("\n=== 测试出牌 ===")
    # 测试出牌逻辑
    player0 = game.players[0]
    if player0.hand:
        card = player0.hand[0]
        game.play_card(0, [card])
        print(f"Player1 played: {card}")

    # 结算轮次
    winner = game.resolve_round()
    print(f"Winner: Player{winner+1}")

    print("\n=== 当前状态 ===")
    print(f"Team Scores: {game.state.team_scores}")
    print(f"Round Count: {game.state.round_count}")
