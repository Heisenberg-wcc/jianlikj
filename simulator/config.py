"""
监利K机模拟器配置文件
Configuration file for the Kaji Poker Simulator
"""

import os
from typing import List, Dict

class Config:
    """全局配置类"""

    # 项目根目录
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

    # 数据目录
    DATA_DIR = os.path.join(PROJECT_ROOT, "data")

    # 训练数据目录
    TRAINING_DIR = os.path.join(PROJECT_ROOT, "training_data")
    GAMES_LOG_DIR = os.path.join(TRAINING_DIR, "game_logs")
    MODELS_DIR = os.path.join(TRAINING_DIR, "models")
    REPLAY_BUFFER_DIR = os.path.join(TRAINING_DIR, "replay_buffers")

    # 报告输出目录
    REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")

    # 游戏配置
    NUM_PLAYERS = 4  # 4人游戏
    NUM_DECKS = 2  # 使用两副牌
    TOTAL_CARDS = 108  # 2副牌 × 54张
    INITIAL_CARDS_PER_PLAYER = 27  # 每人初始27张牌

    # 分数配置
    SCORE_CARDS = {5: 5, 10: 10, 13: 10}  # 5=5分, 10=10分, K(13)=10分
    FOUR_KINGS_SCORE = 100  # 四王直接兑换100分
    TEAM_TOTAL_SCORE = 200  # 全局总分

    # 头游/末游奖励
    REWARD_1ST_2ND = 60  # 同队前两名 +60分
    REWARD_1ST_3RD = 30  # 一游+三游同队 +30分
    REWARD_1ST_3RD_V2 = 40  # 可选版本
    PENALTY_LAST = 0  # 末游扣分(可根据规则调整)

    # AI配置
    AI_DEFAULT_DEPTH = 3  # 搜索深度
    AI_EXPLORATION_RATE = 0.1  # 探索率
    AI_LEARNING_RATE = 0.001  # 学习率

    # 模拟配置
    SIMULATION_BATCH_SIZE = 100  # 每批次模拟的游戏数量
    SIMULATION_MAX_GAMES = 10000  # 总模拟游戏数上限

    # Pygame配置
    PYGAME_WINDOW_WIDTH = 1280
    PYGAME_WINDOW_HEIGHT = 768
    PYGAME_FPS = 60

    # 颜色定义 (中国股市涨跌颜色)
    COLOR_WIN = (255, 0, 0)    # 红色 - 胜利
    COLOR_LOSE = (0, 255, 0)  # 绿色 - 失败
    COLOR_DRAW = (255, 255, 0) # 黄色 - 平局

    # 棋盘/桌面颜色
    TABLE_COLOR = (34, 139, 34)  # 森林绿
    CARD_BACK_COLOR = (50, 50, 150)
    CARD_FACE_COLOR = (255, 255, 255)

    # 日志级别
    LOG_LEVEL = "INFO"
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # 扑克牌配置
    SUITS = ['spade', 'heart', 'club', 'diamond']  # 黑桃, 红桃, 梅花, 方块
    SUIT_SYMBOLS = {'spade': '♠', 'heart': '♥', 'club': '♣', 'diamond': '♦'}
    RANKS = list(range(3, 15))  # 3-A, A=14, 2=15, 小王=16, 大王=17
    RANK_NAMES = {
        3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9', 10: '10',
        11: 'J', 12: 'Q', 13: 'K', 14: 'A', 15: '2', 16: '小王', 17: '大王'
    }

def setup_directories():
    """创建项目所需的目录结构"""
    dirs = [
        Config.DATA_DIR,
        Config.TRAINING_DIR,
        Config.GAMES_LOG_DIR,
        Config.MODELS_DIR,
        Config.REPLAY_BUFFER_DIR,
        Config.REPORTS_DIR
    ]
    for directory in dirs:
        os.makedirs(directory, exist_ok=True)
