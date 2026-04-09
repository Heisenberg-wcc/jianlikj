"""
AI基类和规则决策引擎
AI Base class and Rule-based Decision Engine

核心策略:
  - 牌型拆解规划: 保留完整连对/连三张/炸弹，减少散牌
  - 出牌优先级: 小散牌 → 小对子 → 连对/连三张 → 大牌/炸弹
  - 分值牌保护: 5/10/K不主动单出，优先组合在复合牌型中
  - 炸弹使用时机: 抢控牌权/抢高分轮/清牌阶段
  - 队友配合: 队友出牌不压（除非能走完）；对手出牌积极压制
  - 跟牌策略: 对手出牌用最小能压的牌；队友出牌一般PASS
  - 终局意识: 手牌 <= 5 张进入清牌模式
"""

import random
from typing import List, Optional, Tuple, Dict
from collections import Counter, defaultdict
from card import Card
from rules import Hand, HandType, RulesEngine
from action_space import ActionSpace
from game_engine import GameEngine


class AIBase:
    """AI基类"""

    def __init__(self, player_id: int, game_engine: GameEngine):
        self.player_id = player_id
        self.game_engine = game_engine
        self.action_space = ActionSpace()

    def decide_action(self, last_hand: Optional[Hand] = None) -> Optional[List[Card]]:
        raise NotImplementedError


class RandomAI(AIBase):
    """随机AI - 随机选择合法动作"""

    def decide_action(self, last_hand: Optional[Hand] = None) -> Optional[List[Card]]:
        player = self.game_engine.get_player(self.player_id)
        actions = self.action_space.get_all_actions(player.hand, last_hand)
        if not actions:
            return None
        return random.choice(actions)


class GreedyAI(AIBase):
    """贪心AI - 优先出能抓分的牌"""

    def decide_action(self, last_hand: Optional[Hand] = None) -> Optional[List[Card]]:
        player = self.game_engine.get_player(self.player_id)
        actions = self.action_space.get_all_actions(player.hand, last_hand)
        if not actions:
            return None
        if last_hand is None:
            actions_with_score = []
            for action in actions:
                score = sum(card.get_score_value() for card in action)
                actions_with_score.append((score, action))
            actions_with_score.sort(key=lambda x: (x[0], -len(x[1])))
            return actions_with_score[0][1]
        else:
            actions_with_length = [(len(action), action) for action in actions]
            actions_with_length.sort(key=lambda x: (x[0], sum(card.rank for card in x[1])))
            return actions_with_length[0][1]


# ────────────────────────────────────────────────────────────
# 规则决策引擎 (RuleBasedAI)
# ────────────────────────────────────────────────────────────
class RuleBasedAI(AIBase):
    """
    规则决策引擎 — 通过一系列硬编码策略规则让 AI 达到普通玩家水平。
    可独立使用，也可作为混合决策机制中的"规则层"。
    """

    def __init__(self, player_id: int, game_engine: GameEngine):
        super().__init__(player_id, game_engine)
        self.rules = RulesEngine()

    # ─── 辅助: 获取队友/对手 ─────────────────────
    def _teammate_id(self) -> Optional[int]:
        return self.game_engine.get_player(self.player_id).get_teammate_id()

    def _opponent_ids(self) -> List[int]:
        tid = self._teammate_id()
        return [i for i in range(4) if i != self.player_id and i != tid]

    def _is_teammate(self, pid: int) -> bool:
        return pid == self._teammate_id()

    # ─── 辅助: 手牌分析 ──────────────────────────
    def _hand_counter(self, hand: List[Card]) -> Counter:
        return Counter(c.rank for c in hand)

    def _rank_to_cards(self, hand: List[Card]) -> Dict[int, List[Card]]:
        d = defaultdict(list)
        for c in hand:
            d[c.rank].append(c)
        return d

    def _score_of(self, cards: List[Card]) -> int:
        return sum(c.get_score_value() for c in cards)

    def _hand_type(self, cards: List[Card]) -> HandType:
        return self.rules.detect_hand_type(cards)

    def _count_bombs(self, hand: List[Card]) -> int:
        """统计手牌中炸弹数量"""
        counter = self._hand_counter(hand)
        return sum(1 for cnt in counter.values() if cnt >= 4)

    def _count_510k(self, hand: List[Card]) -> int:
        """统计手牌中510K数量(至少有1组即算1)"""
        ranks = set(c.rank for c in hand)
        return 1 if {5, 10, 13}.issubset(ranks) else 0

    # ─── 辅助: 分类动作 ──────────────────────────
    def _classify_actions(self, actions: List[List[Card]]) -> Dict[str, List[List[Card]]]:
        """
        将合法动作按牌型分类:
        singles, pairs, consecutive_pairs, consecutive_threes, bombs, five_ten_king
        """
        cats: Dict[str, List[List[Card]]] = {
            'singles': [], 'pairs': [], 'consecutive_pairs': [],
            'consecutive_threes': [], 'bombs': [], 'five_ten_king': [],
        }
        for a in actions:
            ht = self._hand_type(a)
            if ht == HandType.SINGLE:
                cats['singles'].append(a)
            elif ht == HandType.PAIR:
                cats['pairs'].append(a)
            elif ht == HandType.CONSECUTIVE_PAIRS:
                cats['consecutive_pairs'].append(a)
            elif ht == HandType.CONSECUTIVE_THREES:
                cats['consecutive_threes'].append(a)
            elif ht == HandType.BOMB:
                cats['bombs'].append(a)
            elif ht == HandType.FIVE_TEN_KING:
                cats['five_ten_king'].append(a)
        return cats

    def _sort_by_rank(self, action_list: List[List[Card]], ascending=True) -> List[List[Card]]:
        """按平均rank排序"""
        return sorted(
            action_list,
            key=lambda a: sum(c.rank for c in a) / len(a),
            reverse=not ascending
        )

    def _is_score_action(self, action: List[Card]) -> bool:
        """该动作是否包含分值牌"""
        return any(c.is_score_card() for c in action)

    # ─── 辅助: 判断是否能一次性出完 ────────────────
    def _can_finish_now(self, hand: List[Card], last_hand: Optional[Hand]) -> Optional[List[Card]]:
        """
        检查当前手牌能否一次性全部出完。
        返回能出完的那个动作, 否则返回 None。
        """
        if not hand:
            return None
        ht = self._hand_type(hand)
        if ht == HandType.INVALID:
            return None
        if last_hand is None:
            return list(hand)  # 主动出牌，只要是合法牌型即可
        h = Hand(hand, ht)
        if self.rules.can_beat(h, last_hand):
            return list(hand)
        return None

    # ─── 辅助: 场上轮次分值估计 ───────────────────
    def _current_round_score(self) -> int:
        """当前轮次场上的分值牌总分"""
        return sum(c.get_score_value() for c in self.game_engine.state.current_round_cards)

    # ════════════════════════════════════════════
    # 核心决策入口
    # ════════════════════════════════════════════
    def decide_action(self, last_hand: Optional[Hand] = None,
                      last_player: int = -1) -> Optional[List[Card]]:
        """
        规则决策入口。

        Args:
            last_hand   : 上一手牌 (None=主动出牌)
            last_player : 上一手出牌的玩家ID (-1=轮首)
        Returns:
            出牌列表 or None(PASS)
        """
        player = self.game_engine.get_player(self.player_id)
        hand = player.hand
        if not hand:
            return None

        actions = self.action_space.get_all_actions(hand, last_hand)
        if not actions:
            return None

        # ── 检查能否一次性出完 ──
        finish = self._can_finish_now(hand, last_hand)
        if finish is not None:
            return finish

        # ── 清牌模式 (手牌<=5) ──
        if len(hand) <= 5:
            return self._endgame_strategy(hand, actions, last_hand, last_player)

        # ── 主动出牌 (轮首) ──
        if last_hand is None:
            return self._lead_strategy(hand, actions)

        # ── 跟牌 (需要压过上家) ──
        return self._follow_strategy(hand, actions, last_hand, last_player)

    # ════════════════════════════════════════════
    # 策略 1: 主动出牌 (轮首)
    # ════════════════════════════════════════════
    def _lead_strategy(self, hand: List[Card],
                       actions: List[List[Card]]) -> Optional[List[Card]]:
        """
        主动出牌优先级:
          1. 非分值散牌 (从小到大)
          2. 非分值对子 (从小到大)
          3. 非分值连对/连三张 (从小到大)
          4. 含分值的连对/连三张 (将分值牌组合消耗)
          5. 含分值的小对子
          6. 含分值的散牌 (不得已)
          7. 510K (仅当手牌中有多组时)
        不主动出炸弹。
        """
        cats = self._classify_actions(actions)

        # 过滤掉炸弹和510K, 后续单独处理
        non_bomb = []
        non_bomb.extend(cats['singles'])
        non_bomb.extend(cats['pairs'])
        non_bomb.extend(cats['consecutive_pairs'])
        non_bomb.extend(cats['consecutive_threes'])

        # ── 1. 不含分值的散牌 ──
        no_score_singles = [a for a in cats['singles'] if not self._is_score_action(a)]
        if no_score_singles:
            return self._sort_by_rank(no_score_singles, ascending=True)[0]

        # ── 2. 不含分值的对子 ──
        no_score_pairs = [a for a in cats['pairs'] if not self._is_score_action(a)]
        if no_score_pairs:
            return self._sort_by_rank(no_score_pairs, ascending=True)[0]

        # ── 3. 不含分值的连对/连三张 ──
        no_score_cp = [a for a in cats['consecutive_pairs'] if not self._is_score_action(a)]
        if no_score_cp:
            return self._sort_by_rank(no_score_cp, ascending=True)[0]
        no_score_ct = [a for a in cats['consecutive_threes'] if not self._is_score_action(a)]
        if no_score_ct:
            return self._sort_by_rank(no_score_ct, ascending=True)[0]

        # ── 4. 含分值的连对/连三张 (优先用组合牌消耗分值牌) ──
        score_cp = [a for a in cats['consecutive_pairs'] if self._is_score_action(a)]
        if score_cp:
            return self._sort_by_rank(score_cp, ascending=True)[0]
        score_ct = [a for a in cats['consecutive_threes'] if self._is_score_action(a)]
        if score_ct:
            return self._sort_by_rank(score_ct, ascending=True)[0]

        # ── 5. 含分值的对子 ──
        score_pairs = [a for a in cats['pairs'] if self._is_score_action(a)]
        if score_pairs:
            return self._sort_by_rank(score_pairs, ascending=True)[0]

        # ── 6. 含分值的散牌 (不得已) ──
        score_singles = [a for a in cats['singles'] if self._is_score_action(a)]
        if score_singles:
            # 优先出5分牌(rank=5), 尽量保住10分牌(rank=10,13)
            sorted_ss = sorted(score_singles, key=lambda a: a[0].get_score_value())
            return sorted_ss[0]

        # ── 7. 510K ──
        if cats['five_ten_king']:
            return cats['five_ten_king'][0]

        # ── 8. 实在没有, 出最小的炸弹 ──
        if cats['bombs']:
            return self._sort_by_rank(cats['bombs'], ascending=True)[0]

        # 兜底: 最小动作
        return self._sort_by_rank(actions, ascending=True)[0]

    # ════════════════════════════════════════════
    # 策略 2: 跟牌 (需要压过上家)
    # ════════════════════════════════════════════
    def _follow_strategy(self, hand: List[Card],
                         actions: List[List[Card]],
                         last_hand: Hand,
                         last_player: int) -> Optional[List[Card]]:
        """
        跟牌策略:
        - 队友出牌: 一般 PASS, 除非能一次出完
        - 对手出牌: 用最小能压过的牌
        - 炸弹: 仅在抢分/清牌/必要夺回控牌权时使用
        """
        is_teammate_play = self._is_teammate(last_player) if last_player >= 0 else False

        # ── 队友出牌: 一般 PASS ──
        if is_teammate_play:
            # 唯一例外: 能一次出完
            finish = self._can_finish_now(hand, last_hand)
            if finish is not None:
                return finish
            return None  # PASS 配合队友

        # ── 对手出牌: 积极压制 ──
        cats = self._classify_actions(actions)

        # 同类型压制 (非炸弹非510K)
        same_type_actions = []
        for a in actions:
            ht = self._hand_type(a)
            if ht == last_hand.hand_type and ht not in (HandType.BOMB, HandType.FIVE_TEN_KING):
                same_type_actions.append(a)

        if same_type_actions:
            # 优先选不含分值牌的最小牌
            no_score = [a for a in same_type_actions if not self._is_score_action(a)]
            if no_score:
                return self._sort_by_rank(no_score, ascending=True)[0]
            # 全含分值, 选最小
            return self._sort_by_rank(same_type_actions, ascending=True)[0]

        # 没有同类型, 考虑510K
        if cats['five_ten_king']:
            round_score = self._current_round_score()
            # 场上有分值牌 >= 15 或手牌较少时, 使用 510K
            if round_score >= 15 or len(hand) <= 8:
                return cats['five_ten_king'][0]

        # 考虑炸弹
        if cats['bombs']:
            round_score = self._current_round_score()
            # 使用炸弹条件: (1) 场上分值>=20 (2) 手牌<=8需要夺控牌权 (3) 清牌阶段
            if round_score >= 20 or len(hand) <= 8:
                return self._sort_by_rank(cats['bombs'], ascending=True)[0]

        # 无法压过, PASS
        return None

    # ════════════════════════════════════════════
    # 策略 3: 终局/清牌模式 (手牌<=5)
    # ════════════════════════════════════════════
    def _endgame_strategy(self, hand: List[Card],
                          actions: List[List[Card]],
                          last_hand: Optional[Hand],
                          last_player: int) -> Optional[List[Card]]:
        """
        清牌模式: 尽快出完牌。
        优先选能出最多张数的动作, 减少手牌轮次。
        """
        # 能一次出完, 直接出
        finish = self._can_finish_now(hand, last_hand)
        if finish is not None:
            return finish

        if last_hand is None:
            # 主动出牌: 优先出张数最多的牌型
            actions_sorted = sorted(actions, key=lambda a: (-len(a), sum(c.rank for c in a)))
            return actions_sorted[0]

        # 跟牌: 队友出牌也可以考虑压 (清牌阶段)
        # 但如果队友出牌且自己不能出完, 还是PASS
        is_teammate_play = self._is_teammate(last_player) if last_player >= 0 else False
        if is_teammate_play:
            # 看自己出完后剩余的手牌能否一手出完
            for a in actions:
                remain = [c for c in hand if c not in a]
                if not remain:
                    return a  # 直接出完
                rht = self._hand_type(remain)
                if rht != HandType.INVALID:
                    return a  # 出完后剩一手, 值得出
            return None  # 不能有效清牌, PASS

        # 对手出牌: 尽量压, 选出最多张数的
        actions_sorted = sorted(actions, key=lambda a: (-len(a), sum(c.rank for c in a)))
        return actions_sorted[0]


# ────────────────────────────────────────────────────────────
# 混合决策 AI: 规则引擎 + Q 网络 (旧版, 保留兼容)
# ────────────────────────────────────────────────────────────
class HybridAI:
    """
    混合决策 AI (旧版兼容):
    - 规则引擎先做预判, 如果触发高置信度规则(队友配合/炸弹时机/清牌)则直接采用规则决策
    - 否则使用 Q 网络选择动作
    """

    def __init__(self, player_id: int, game_engine: GameEngine):
        self.player_id = player_id
        self.game_engine = game_engine
        self.rule_ai = RuleBasedAI(player_id, game_engine)
        self.action_space = ActionSpace()
        self.rules = RulesEngine()

    def decide_action(self, last_hand: Optional[Hand],
                      last_player: int,
                      q_select_fn=None) -> Tuple[Optional[List[Card]], str]:
        """
        混合决策入口。

        Args:
            last_hand    : 上一手牌 (None=轮首)
            last_player  : 上一手出牌者 (-1=轮首)
            q_select_fn  : Q网络选择函数, 签名 () -> (action, action_enc, q_val)
                           如果为 None 则纯规则模式
        Returns:
            (action, source)  source='rule'或'qnet'
        """
        player = self.game_engine.get_player(self.player_id)
        hand = player.hand

        # ── 高置信度规则: 直接走规则 ──

        # 规则1: 手牌为空
        if not hand:
            return None, 'rule'

        # 规则2: 能一次出完 → 必出
        finish = self.rule_ai._can_finish_now(hand, last_hand)
        if finish is not None:
            return finish, 'rule'

        # 规则3: 清牌模式 (<=5张) → 规则引擎
        if len(hand) <= 5:
            action = self.rule_ai.decide_action(last_hand, last_player)
            return action, 'rule'

        # 规则4: 队友出牌 → PASS (除非能出完, 已在规则2处理)
        is_teammate_play = self.rule_ai._is_teammate(last_player) if last_player >= 0 else False
        if last_hand is not None and is_teammate_play:
            return None, 'rule'

        # 规则5: 主动出牌且没有Q网络 → 规则引擎
        if q_select_fn is None:
            action = self.rule_ai.decide_action(last_hand, last_player)
            return action, 'rule'

        # ── 中等/低置信度场景: 使用Q网络 ──
        action, action_enc, q_val = q_select_fn()

        # 安全网: 如果Q网络选了一个"明显差"的动作, 用规则兜底
        if action is not None:
            # 检查: 主动出牌时不应先出大牌(rank>=14)如果有小牌(rank<=8)
            if last_hand is None:
                avg_rank = sum(c.rank for c in action) / len(action)
                has_small = any(c.rank <= 8 for c in hand)
                ht = self.rules.detect_hand_type(action)
                if avg_rank >= 14 and has_small and ht == HandType.SINGLE:
                    # Q网络想出大牌, 但有小牌可以出 → 用规则
                    rule_action = self.rule_ai.decide_action(last_hand, last_player)
                    if rule_action is not None:
                        return rule_action, 'rule'

            # 检查: 不必要地浪费炸弹 (场上无分值牌且手牌多)
            ht = self.rules.detect_hand_type(action)
            if ht == HandType.BOMB and len(hand) > 8:
                round_score = self.rule_ai._current_round_score()
                if round_score < 10 and last_hand is not None:
                    # 场上分少, 不值得炸 → 规则兜底
                    rule_action = self.rule_ai.decide_action(last_hand, last_player)
                    return rule_action, 'rule'

        return action, 'qnet'


# ────────────────────────────────────────────────────────────
# 三层混合决策 AI: 规则层 + 合法动作掩码 + 策略网络 (阶段三)
# ────────────────────────────────────────────────────────────
class ThreeLayerHybridAI:
    """
    三层混合决策架构 (终极形态):

    第一层 - 规则层:
      处理高确定性决策: 一手出完 / 清牌阶段 / 队友配合 / 必炸场景

    第二层 - 合法动作过滤:
      枚举合法动作, 非法动作掩码, 确保策略网络只能选合法动作

    第三层 - 策略网络:
      在合法动作集中按概率选择最优动作

    增强奖励:
      团队奖励 / 控牌权奖励 / 节奏奖励 / 对手接近出完惩罚
    """

    def __init__(self, player_id: int, game_engine: GameEngine):
        self.player_id = player_id
        self.game_engine = game_engine
        self.rule_ai = RuleBasedAI(player_id, game_engine)
        self.action_space = ActionSpace()
        self.rules = RulesEngine()

    # ════════════════════════════════════════════
    # 第一层: 规则层
    # ════════════════════════════════════════════
    def _rule_layer(
        self,
        hand: List[Card],
        last_hand: Optional[Hand],
        last_player: int,
    ) -> Tuple[Optional[List[Card]], bool]:
        """
        规则层: 处理高确定性决策。

        Returns:
            (action, handled)
            handled=True 表示规则层已做出决策, 无需进入策略网络。
        """
        # 规则1: 手牌为空
        if not hand:
            return None, True

        # 规则2: 能一次出完 → 必出
        finish = self.rule_ai._can_finish_now(hand, last_hand)
        if finish is not None:
            return finish, True

        # 规则3: 清牌模式 (手牌 <= 5 张) → 规则引擎决策
        if len(hand) <= 5:
            action = self.rule_ai.decide_action(last_hand, last_player)
            return action, True

        # 规则4: 队友出牌且不能出完 → PASS
        is_teammate_play = self.rule_ai._is_teammate(last_player) if last_player >= 0 else False
        if last_hand is not None and is_teammate_play:
            return None, True

        # 规则5: 必炸场景 — 场上分值 >= 30 且对手即将获胜
        if last_hand is not None and not is_teammate_play:
            round_score = self.rule_ai._current_round_score()
            opp_ids = self.rule_ai._opponent_ids()
            opp_close = any(
                len(self.game_engine.get_player(oid).hand) <= 3
                for oid in opp_ids
                if not self.game_engine.get_player(oid).is_out()
            )
            if round_score >= 30 or opp_close:
                actions = self.action_space.get_all_actions(hand, last_hand)
                cats = self.rule_ai._classify_actions(actions)
                if cats['bombs']:
                    return self.rule_ai._sort_by_rank(cats['bombs'], ascending=True)[0], True

        # 规则层未触发 → 交给策略网络
        return None, False

    # ════════════════════════════════════════════
    # 第二层: 合法动作过滤 (掩码)
    # ════════════════════════════════════════════
    def _mask_layer(
        self,
        hand: List[Card],
        last_hand: Optional[Hand],
        last_played_cards: list,
    ) -> Tuple[List[Optional[List[Card]]], List]:
        """
        合法动作过滤层: 生成所有合法动作及其编码。

        Returns:
            (actions, action_encs)
            actions[0] 可能是 None (表示 pass)
        """
        legal = self.action_space.get_all_actions(hand, last_hand)
        can_pass = len(last_played_cards) > 0

        actions = []
        action_encs = []
        if can_pass:
            actions.append(None)
            # encode_action 延迟导入以避免循环依赖
            from q_net.q_network import encode_action
            action_encs.append(encode_action(None))

        from q_net.q_network import encode_action
        for a in legal:
            actions.append(a)
            action_encs.append(encode_action(a))

        if not actions:
            from q_net.q_network import encode_action
            actions.append(None)
            action_encs.append(encode_action(None))

        return actions, action_encs

    # ════════════════════════════════════════════
    # 第三层: 策略网络 (或 Q 网络)
    # ════════════════════════════════════════════
    def decide_action(
        self,
        last_hand: Optional[Hand],
        last_player: int,
        last_played_cards: list,
        policy_select_fn=None,
    ) -> Tuple[Optional[List[Card]], str]:
        """
        三层混合决策入口。

        Args:
            last_hand         : 上一手牌 (None=轮首)
            last_player       : 上一手出牌者 (-1=轮首)
            last_played_cards : 上一手出的原始牌列表
            policy_select_fn  : 策略网络选择函数,
                签名 (actions, action_encs) -> (action, action_enc, score)
                如果为 None 则纯规则模式
        Returns:
            (action, source)  source='rule'或'policy'
        """
        player = self.game_engine.get_player(self.player_id)
        hand = player.hand

        # ── 第一层: 规则层 ──
        rule_action, handled = self._rule_layer(hand, last_hand, last_player)
        if handled:
            return rule_action, 'rule'

        # ── 第二层: 合法动作掩码 ──
        actions, action_encs = self._mask_layer(hand, last_hand, last_played_cards)

        # 无策略网络 → 回退规则引擎
        if policy_select_fn is None:
            action = self.rule_ai.decide_action(last_hand, last_player)
            return action, 'rule'

        # ── 第三层: 策略网络 ──
        action, action_enc, score = policy_select_fn(actions, action_encs)

        # 安全网: 策略网络选了明显差的动作时, 规则兜底
        if action is not None:
            # 主动出牌时不应先出大单牌
            if last_hand is None:
                avg_rank = sum(c.rank for c in action) / len(action)
                has_small = any(c.rank <= 8 for c in hand)
                ht = self.rules.detect_hand_type(action)
                if avg_rank >= 14 and has_small and ht == HandType.SINGLE:
                    rule_action = self.rule_ai.decide_action(last_hand, last_player)
                    if rule_action is not None:
                        return rule_action, 'rule'

            # 不必要浪费炸弹
            ht = self.rules.detect_hand_type(action)
            if ht == HandType.BOMB and len(hand) > 8:
                round_score = self.rule_ai._current_round_score()
                if round_score < 10 and last_hand is not None:
                    rule_action = self.rule_ai.decide_action(last_hand, last_player)
                    return rule_action, 'rule'

        return action, 'policy'


# ────────────────────────────────────────────────────────────
# 增强奖励计算器 (阶段三)
# ────────────────────────────────────────────────────────────
class EnhancedRewardCalculator:
    """
    阶段三增强奖励体系:
      - 团队奖励: 队友赢得轮次分值 → +分值/400
      - 控牌权奖励: 赢得一轮 → +0.02
      - 节奏奖励: 出牌张数 → +0.01 * 张数
      - 对手接近出完惩罚: 对手手牌<=3 且未压制 → -0.03
      - 基础即时奖励 (同阶段一)
    """

    @staticmethod
    def compute(
        game: GameEngine,
        player_idx: int,
        action: Optional[List[Card]],
        action_success: bool,
        score_before: list,
        game_over_before: bool,
        last_play_player_before: int,
        last_play_player_after: int,
    ) -> float:
        reward = 0.0
        gs = game.get_game_state()
        my_team = 0 if player_idx in [0, 2] else 1
        opp_team = 1 - my_team

        # 非法出牌惩罚
        if action is not None and not action_success:
            return -0.05

        new_scores = gs.team_scores

        # ── 基础: 轮次分值变化 ──
        my_delta  = new_scores[my_team]  - score_before[my_team]
        opp_delta = new_scores[opp_team] - score_before[opp_team]
        if my_delta > 0:
            reward += my_delta / 200.0
        if opp_delta > 0:
            reward -= opp_delta / 200.0

        # ── 出完牌奖励 ──
        player = game.get_player(player_idx)
        if player.is_out() and player.position is not None:
            finish_bonus = {1: 0.5, 2: 0.3, 3: 0.1, 4: 0.0}
            reward += finish_bonus.get(player.position, 0.0)

        # ── 对局最终胜负 ──
        if gs.game_over and not game_over_before:
            if new_scores[my_team] > new_scores[opp_team]:
                reward += 1.0
            elif new_scores[my_team] < new_scores[opp_team]:
                reward -= 1.0

        # ── 增强: 团队奖励 ──
        teammate_idx = 2 - player_idx if player_idx in [0, 2] else 4 - player_idx
        teammate_team = 0 if teammate_idx in [0, 2] else 1
        if teammate_team == my_team:
            teammate_delta = new_scores[my_team] - score_before[my_team]
            if teammate_delta > 0 and action is None:
                # 队友赢分时给正反馈
                reward += teammate_delta / 400.0

        # ── 增强: 控牌权奖励 ──
        if (action is not None and action_success
                and last_play_player_after == player_idx
                and last_play_player_before != player_idx):
            reward += 0.02

        # ── 增强: 节奏奖励 ──
        if action is not None and action_success:
            reward += 0.01 * len(action)

        # ── 增强: 对手接近出完惩罚 ──
        if action is None:  # PASS
            opp_ids = [1, 3] if my_team == 0 else [0, 2]
            for oid in opp_ids:
                opp = game.get_player(oid)
                if not opp.is_out() and len(opp.hand) <= 3:
                    reward -= 0.03
                    break

        return reward


# ── 兼容旧代码: SmartAI 指向 RuleBasedAI ──
SmartAI = RuleBasedAI
