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

    # ── 阶段一: TD Q-Learning 超参数 ──
    TD_GAMMA           = 0.99       # 折扣系数 (0.98~0.995)
    TD_LR              = 1e-4       # 学习率 (配合余弦退火)
    TD_BATCH_SIZE      = 512        # 批大小
    TD_BUFFER_MAX      = 200_000    # 经验缓冲区容量
    TD_EPSILON_START   = 1.0        # 初始探索率
    TD_EPSILON_MIN     = 0.05       # 最小探索率
    TD_EPSILON_DECAY   = 0.997      # 每局衰减系数
    TD_TARGET_SYNC     = 1000       # 目标网络同步频率 (每N次梯度更新)
    TD_GRAD_CLIP       = 1.0        # 梯度裁剪阈值
    TD_TRAIN_PER_EP    = 4          # 每局训练步数
    TD_SAVE_INTERVAL   = 500        # 模型保存间隔 (每N局)
    TD_SAVE_DIR        = "q_models" # 模型保存目录

    # ── 阶段二: 策略梯度超参数 ──
    PG_LR_ACTOR        = 3e-4       # Actor 学习率
    PG_LR_CRITIC       = 1e-3       # Critic 学习率
    PG_GAMMA           = 0.99       # 折扣系数
    PG_ENTROPY_COEF    = 0.02       # 熵奖励系数 (初期较大保持探索)
    PG_VALUE_COEF      = 0.5        # 价值损失权重
    PG_GRAD_CLIP       = 1.0        # 梯度裁剪
    PG_SAVE_DIR        = "pg_models"# 策略模型保存目录
    PG_SAVE_INTERVAL   = 500        # 保存间隔

    # ── 阶段三: 混合决策系统超参数 ──
    HY_MODEL_POOL_SIZE = 20         # 历史模型池容量
    HY_POOL_SAVE_FREQ  = 2000       # 入池频率 (每N局)
    HY_LATEST_RATIO    = 0.7        # 对手采样: 最新模型占比
    HY_ENTROPY_START   = 0.05       # 初始熵系数
    HY_ENTROPY_END     = 0.005      # 最终熵系数
    HY_ENTROPY_DECAY   = 50_000     # 熵衰减周期 (局数)
    HY_SAVE_DIR        = "hybrid_models"  # 混合模型保存目录
    HY_SAVE_INTERVAL   = 500        # 保存间隔

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
