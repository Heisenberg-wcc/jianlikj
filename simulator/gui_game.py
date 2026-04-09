"""
监利K · AI四人对战可视化界面
"""

import os
import sys
import glob
import tkinter as tk
from tkinter import ttk, scrolledtext

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from game_engine import GameEngine
from train_q_mc import MCQTrainer

BG_TABLE   = '#1a6b2a'
BG_TEAM1   = '#2a5a1a'
BG_TEAM2   = '#1a1a5a'
BG_ACTIVE  = '#4aab3c'
BG_TOPBAR  = '#0d3d12'

SUIT_COLOR = {
    'spade': '#111111', 'club': '#111111',
    'heart': '#cc0000', 'diamond': '#cc0000', None: '#7700bb',
}
RANK_NAMES = {
    3:'3', 4:'4', 5:'5', 6:'6', 7:'7', 8:'8', 9:'9', 10:'10',
    11:'J', 12:'Q', 13:'K', 14:'A', 15:'2', 16:'小王', 17:'大王'
}
CW, CH = 34, 50
GAP    = 18


class CardCanvas(tk.Canvas):
    def __init__(self, parent, **kwargs):
        kwargs.setdefault('height', CH + 8)
        kwargs.setdefault('highlightthickness', 0)
        super().__init__(parent, **kwargs)

    def draw_cards(self, cards):
        self.delete('all')
        if not cards:
            return
        n = len(cards)
        w = self.winfo_width()
        if w < 20:
            w = int(self.cget('width') or 400)
        gap = min(GAP, max(4, (w - CW - 10) // max(n - 1, 1))) if n > 1 else GAP
        for i, card in enumerate(cards):
            self._draw_one(5 + i * gap, 4, card)

    def _draw_one(self, x, y, card):
        score_bg = '#fff8f8' if card.rank in [5, 10, 13] else '#fffef8'
        if card.rank == 17:   score_bg = '#fff8e0'
        elif card.rank == 16: score_bg = '#e8f4ff'
        self.create_rectangle(x, y, x+CW, y+CH, fill=score_bg, outline='#999', width=1)
        rank = RANK_NAMES.get(card.rank, str(card.rank))
        if card.rank in [16, 17]:
            color = '#cc5500' if card.rank == 17 else '#0055cc'
            self.create_text(x+CW//2, y+CH//2, text=rank, fill=color, font=('SimHei', 8, 'bold'))
        else:
            sym = {'spade':'♠','heart':'♥','club':'♣','diamond':'♦'}.get(card.suit, '?')
            color = SUIT_COLOR.get(card.suit, '#333')
            self.create_text(x+3, y+7, anchor='nw', text=rank, fill=color, font=('Arial', 7, 'bold'))
            self.create_text(x+CW//2, y+CH//2, text=sym, fill=color, font=('Arial', 13))


class GameGUI:
    SAVE_DIR = 'q_models'

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title('监利K · AI四人对战可视化')
        self.root.configure(bg=BG_TABLE)
        self.root.resizable(True, True)

        self.game: GameEngine = None
        self.trainers = [None, None, None, None]   # 每位玩家独立模型
        self.last_played_cards = []
        self.last_play_player  = -1
        self.passed_players    = set()
        self.step      = 0
        self.game_over = False
        self.auto_id   = None
        self.p_last_action = ['', '', '', '']
        self.p_last_cards  = [[], [], [], []]

        self._build_ui()
        self._refresh_model_list()
        self._load_all_models()
        self.new_game()

    # ── 模型管理 ──────────────────────────────
    def _get_model_files(self):
        files = sorted(glob.glob(os.path.join(self.SAVE_DIR, '*.pth')))
        names = [os.path.basename(f) for f in files] if files else []
        return ['【随机】'] + names

    def _refresh_model_list(self):
        files = self._get_model_files()
        for pid in range(4):
            self.p_model_combo[pid]['values'] = files
            cur = self.p_model_var[pid].get()
            if cur not in files:
                # 默认：找 model1.0.pth / q_net_final.pth，否则随机
                if 'model1.0.pth' in files:
                    self.p_model_var[pid].set('model1.0.pth')
                elif 'q_net_final.pth' in files:
                    self.p_model_var[pid].set('q_net_final.pth')
                elif len(files) > 1:
                    self.p_model_var[pid].set(files[-1])
                else:
                    self.p_model_var[pid].set('【随机】')

    def _load_one_model(self, pid: int) -> MCQTrainer:
        name = self.p_model_var[pid].get()
        t = MCQTrainer()
        if name == '【随机】' or not name:
            self._log(f'  P{pid+1}: 随机策略')
            return t
        path = os.path.join(self.SAVE_DIR, name)
        if os.path.exists(path):
            t.load(path)
            self._log(f'  P{pid+1}: {name}')
        else:
            self._log(f'  P{pid+1}: 文件不存在 {name}，使用随机')
        return t

    def _load_all_models(self):
        self._log('[模型分配]')
        for pid in range(4):
            self.trainers[pid] = self._load_one_model(pid)

    def _on_load_models(self):
        self._load_all_models()
        self.new_game()

    # ── 界面构建 ───────────────────────────────
    def _build_ui(self):
        # 顶部状态栏
        top = tk.Frame(self.root, bg=BG_TOPBAR)
        top.pack(fill='x')
        tk.Label(top, text='监利K · AI对战', bg=BG_TOPBAR, fg='#ffd700',
                 font=('SimHei', 13, 'bold')).pack(side='left', padx=10, pady=5)
        self.lbl_scores = tk.Label(top, text='队1: 0  |  队2: 0',
                                   bg=BG_TOPBAR, fg='white', font=('SimHei', 11))
        self.lbl_scores.pack(side='left', padx=20)
        self.lbl_step = tk.Label(top, text='步数: 0', bg=BG_TOPBAR,
                                 fg='#aaffaa', font=('Arial', 10))
        self.lbl_step.pack(side='right', padx=10)

        # 每玩家模型选择栏
        mf = tk.Frame(self.root, bg='#0a2e0d')
        mf.pack(fill='x')
        self.p_model_var   = [tk.StringVar() for _ in range(4)]
        self.p_model_combo = [None] * 4
        p_colors = ['#aaffaa', '#aaaaff', '#aaffaa', '#aaaaff']  # T1绿 T2蓝
        for pid in range(4):
            team = 'T1' if pid in [0, 2] else 'T2'
            fg = p_colors[pid]
            tk.Label(mf, text=f'P{pid+1}({team}):', bg='#0a2e0d', fg=fg,
                     font=('SimHei', 9)).pack(side='left', padx=(8,1), pady=4)
            cb = ttk.Combobox(mf, textvariable=self.p_model_var[pid],
                              width=16, state='readonly', font=('Consolas', 8))
            cb.pack(side='left', padx=(0,4), pady=4)
            self.p_model_combo[pid] = cb
        tk.Button(mf, text='加载并开始', command=self._on_load_models,
                  bg='#2d6b3a', fg='white', font=('SimHei', 9),
                  relief='flat', padx=8, cursor='hand2').pack(side='left', padx=6)
        tk.Button(mf, text='刷新列表', command=self._refresh_model_list,
                  bg='#1a3d20', fg='#aaffaa', font=('SimHei', 9),
                  relief='flat', padx=6, cursor='hand2').pack(side='left', padx=2)

        # 主游戏区
        gf = tk.Frame(self.root, bg=BG_TABLE)
        gf.pack(fill='both', expand=True, padx=6, pady=4)
        gf.columnconfigure(0, weight=1, minsize=170)
        gf.columnconfigure(1, weight=3, minsize=420)
        gf.columnconfigure(2, weight=1, minsize=170)
        gf.rowconfigure(0, weight=1)
        gf.rowconfigure(1, weight=2)
        gf.rowconfigure(2, weight=1)

        self.pf = {}; self.lbl_info = {}; self.lbl_action = {}
        self.cv_played = {}; self.cv_hand = {}

        self._mk_player(gf, pid=2, row=0, col=1)
        self._mk_player(gf, pid=3, row=1, col=0)
        self._mk_player(gf, pid=1, row=1, col=2)
        self._mk_player(gf, pid=0, row=2, col=1)
        self._mk_table(gf, row=1, col=1)

        # 日志
        lf = tk.Frame(self.root, bg='#071e0b')
        lf.pack(fill='x', padx=6, pady=2)
        tk.Label(lf, text='对局日志', bg='#071e0b', fg='#88ff88',
                 font=('SimHei', 9)).pack(side='left', padx=4)
        self.log_box = scrolledtext.ScrolledText(
            lf, height=5, bg='#071e0b', fg='#ccffcc',
            font=('Consolas', 9), state='disabled', wrap='word')
        self.log_box.pack(fill='x', padx=4, pady=2)

        # 控制栏
        cf = tk.Frame(self.root, bg=BG_TOPBAR)
        cf.pack(fill='x', padx=6, pady=4)

        self.btn_step = tk.Button(cf, text='▶ 下一步', command=self.do_step,
                                  bg='#2d8a3a', fg='white', font=('SimHei', 10, 'bold'),
                                  relief='flat', padx=10, pady=4, cursor='hand2')
        self.btn_step.pack(side='left', padx=4)

        self.btn_auto = tk.Button(cf, text='⏩ 自动播放', command=self.toggle_auto,
                                  bg='#1a5c28', fg='white', font=('SimHei', 10),
                                  relief='flat', padx=10, pady=4, cursor='hand2')
        self.btn_auto.pack(side='left', padx=4)

        tk.Label(cf, text='速度:', bg=BG_TOPBAR, fg='white',
                 font=('SimHei', 10)).pack(side='left', padx=(12,2))
        self.speed_var = tk.DoubleVar(value=0.6)
        self.lbl_spd = tk.Label(cf, text='0.6秒/步', bg=BG_TOPBAR,
                                 fg='#aaaaaa', font=('Arial', 9))
        ttk.Scale(cf, from_=0.05, to=3.0, variable=self.speed_var,
                  orient='horizontal', length=110,
                  command=lambda v: self.lbl_spd.config(
                      text=f'{float(v):.1f}秒/步')).pack(side='left', padx=2)
        self.lbl_spd.pack(side='left', padx=2)

        tk.Button(cf, text='🔄 新对局', command=self.new_game,
                  bg='#4a3a10', fg='#ffd700', font=('SimHei', 10),
                  relief='flat', padx=10, pady=4, cursor='hand2').pack(side='left', padx=10)

        self.lbl_status = tk.Label(cf, text='就绪', bg=BG_TOPBAR,
                                   fg='#ffdd88', font=('SimHei', 10))
        self.lbl_status.pack(side='right', padx=10)

    def _mk_player(self, parent, pid, row, col):
        team = 1 if pid in [0, 2] else 2
        bg   = BG_TEAM1 if team == 1 else BG_TEAM2
        fg_t = '#aaffaa' if team == 1 else '#aaaaff'
        frm = tk.LabelFrame(parent, text=f' P{pid+1} · {"Team1" if team==1 else "Team2"} ',
                             bg=bg, fg=fg_t, font=('SimHei', 9, 'bold'),
                             relief='groove', bd=2, labelanchor='n')
        frm.grid(row=row, column=col, padx=4, pady=4, sticky='nsew')
        lbl_info = tk.Label(frm, text='手牌: 27张', bg=bg, fg='#ccffcc', font=('SimHei', 9))
        lbl_info.pack(pady=(2,0))
        lbl_act = tk.Label(frm, text='等待...', bg=bg, fg='#ffff88', font=('SimHei', 9))
        lbl_act.pack()
        tk.Label(frm, text='── 最近出牌 ──', bg=bg, fg='#88ddff', font=('SimHei', 8)).pack()
        cv_played = CardCanvas(frm, bg=bg, height=CH+8, width=300)
        cv_played.pack(fill='x', padx=4, pady=2)
        tk.Label(frm, text='── 手  牌 ──', bg=bg, fg='#88ddff', font=('SimHei', 8)).pack()
        cv_hand = CardCanvas(frm, bg=bg, height=CH+8, width=300)
        cv_hand.pack(fill='x', padx=4, pady=(2,4))
        self.pf[pid]=frm; self.lbl_info[pid]=lbl_info; self.lbl_action[pid]=lbl_act
        self.cv_played[pid]=cv_played; self.cv_hand[pid]=cv_hand

    def _mk_table(self, parent, row, col):
        frm = tk.Frame(parent, bg=BG_TABLE, relief='sunken', bd=2)
        frm.grid(row=row, column=col, padx=4, pady=4, sticky='nsew')
        self.lbl_cur = tk.Label(frm, text='当前出牌: P1',
                                bg=BG_TABLE, fg='#ffd700', font=('SimHei', 12, 'bold'))
        self.lbl_cur.pack(pady=(8,2))
        tk.Label(frm, text='── 场上最近出牌 ──', bg=BG_TABLE, fg='#88ff88', font=('SimHei', 9)).pack()
        self.cv_table = CardCanvas(frm, bg=BG_TABLE, height=CH+12, width=380)
        self.cv_table.pack(padx=8, pady=4)
        self.lbl_htype = tk.Label(frm, text='牌型: -', bg=BG_TABLE, fg='#aaddff', font=('SimHei', 9))
        self.lbl_htype.pack()
        tsf = tk.Frame(frm, bg=BG_TABLE); tsf.pack(pady=6)
        self.lbl_t1 = tk.Label(tsf, text='Team1(P1+P3): 0分', bg='#143d0a', fg='#aaffaa',
                               font=('SimHei', 9), relief='groove', padx=8, pady=2)
        self.lbl_t1.pack(side='left', padx=4)
        self.lbl_t2 = tk.Label(tsf, text='Team2(P2+P4): 0分', bg='#0a0a3d', fg='#aaaaff',
                               font=('SimHei', 9), relief='groove', padx=8, pady=2)
        self.lbl_t2.pack(side='left', padx=4)
        self.lbl_rank = tk.Label(frm, text='出完顺序: -', bg=BG_TABLE, fg='#ffdd88', font=('SimHei', 9))
        self.lbl_rank.pack(pady=2)

    # ── 游戏逻辑 ──────────────────────────────
    def new_game(self):
        if self.auto_id:
            self.root.after_cancel(self.auto_id)
            self.auto_id = None
            self.btn_auto.config(text='⏩ 自动播放', bg='#1a5c28')
        self.game = GameEngine()
        self.game.initialize()
        self.last_played_cards = []
        self.last_play_player  = -1
        self.passed_players    = set()
        self.step = 0; self.game_over = False
        self.p_last_action = ['开局','开局','开局','开局']
        self.p_last_cards  = [[],[],[],[]]
        self.cv_table.delete('all')
        self.lbl_htype.config(text='牌型: -')
        self.lbl_rank.config(text='出完顺序: -')
        self.btn_step.config(state='normal')
        self.lbl_status.config(text='新对局')
        self._log('=' * 52)
        self._log('  新对局开始！')
        self._update_display()

    def do_step(self):
        if self.game_over or not self.game or not all(self.trainers):
            return
        gs = self.game.get_game_state()
        if gs.game_over:
            self._on_game_over(); return
        current_idx = gs.current_player_idx
        player  = self.game.get_player(current_idx)
        trainer = self.trainers[current_idx]   # ← 当前玩家专属模型

        if player.is_out():
            self.game.pass_round(current_idx)
            if self.last_play_player >= 0:
                self.passed_players.add(current_idx)
                (self.last_played_cards, self.last_play_player, self.passed_players) = \
                    trainer._update_round_state(self.game, self.last_played_cards,
                                                self.last_play_player, self.passed_players)
            self.step += 1; self._update_display(); return

        action, _, q_val = trainer.select_action(
            self.game, current_idx, self.last_played_cards, explore=False)

        if action is not None:
            success = self.game.play_card(current_idx, action)
            if not success:
                self.game.pass_round(current_idx)
                if self.last_play_player >= 0:
                    self.passed_players.add(current_idx)
                self.p_last_action[current_idx] = 'PASS (非法)'
                self.p_last_cards[current_idx]  = []
                self._log(f'  P{current_idx+1} 非法→PASS  剩余:{len(player.hand)}张')
            else:
                self.last_played_cards = list(action)
                self.last_play_player  = current_idx
                self.passed_players    = set()
                self.p_last_cards[current_idx] = list(action)
                s = self._cs(action)
                self.p_last_action[current_idx] = f'出牌: {s}'
                ht = self.game.rules.detect_hand_type(action)
                self.lbl_htype.config(text=f'牌型: {ht.name}')
                self.cv_table.draw_cards(action)
                self._log(f'  P{current_idx+1} ▶ [{s}]  Q={q_val:.2f}  剩余:{len(player.hand)}张')
        else:
            self.game.pass_round(current_idx)
            if self.last_play_player >= 0:
                self.passed_players.add(current_idx)
            self.p_last_action[current_idx] = 'PASS'
            self.p_last_cards[current_idx]  = []
            self._log(f'  P{current_idx+1} — PASS  剩余:{len(player.hand)}张')

        (self.last_played_cards, self.last_play_player, self.passed_players) = \
            trainer._update_round_state(self.game, self.last_played_cards,
                                        self.last_play_player, self.passed_players)
        self.step += 1
        self._update_display()
        if self.game.get_game_state().game_over:
            self._on_game_over()

    def _on_game_over(self):
        self.game_over = True
        self.btn_step.config(state='disabled')
        if self.auto_id:
            self.root.after_cancel(self.auto_id)
            self.auto_id = None
            self.btn_auto.config(text='⏩ 自动播放', bg='#1a5c28')
        scores = self.game.get_game_state().team_scores
        order  = self.game.finished_order
        winner = 'Team1(P1+P3)' if scores[0] > scores[1] else 'Team2(P2+P4)'
        rank_s = ' → '.join(f'P{p+1}' for p in order)
        self._log('=' * 52)
        self._log(f'  🎉 游戏结束！  获胜：{winner}')
        self._log(f'  Team1: {scores[0]}分  |  Team2: {scores[1]}分')
        self._log(f'  出完顺序: {rank_s}   总步数: {self.step}')
        self.lbl_status.config(text=f'游戏结束！{winner} 获胜')
        self.lbl_rank.config(text=f'出完顺序: {rank_s}')
        self._update_display()

    def toggle_auto(self):
        if self.auto_id:
            self.root.after_cancel(self.auto_id)
            self.auto_id = None
            self.btn_auto.config(text='⏩ 自动播放', bg='#1a5c28')
        else:
            if not self.game_over:
                self.btn_auto.config(text='⏸ 暂停', bg='#8b2020')
                self._auto_loop()

    def _auto_loop(self):
        if self.game_over:
            self.btn_auto.config(text='⏩ 自动播放', bg='#1a5c28')
            self.auto_id = None; return
        self.do_step()
        if not self.game_over:
            self.auto_id = self.root.after(int(self.speed_var.get() * 1000), self._auto_loop)

    def _update_display(self):
        if not self.game: return
        gs = self.game.get_game_state()
        scores = gs.team_scores; cur = gs.current_player_idx
        self.lbl_scores.config(text=f'队1(P1+P3): {scores[0]}分  |  队2(P2+P4): {scores[1]}分')
        self.lbl_t1.config(text=f'Team1(P1+P3): {scores[0]}分')
        self.lbl_t2.config(text=f'Team2(P2+P4): {scores[1]}分')
        self.lbl_step.config(text=f'步数: {self.step}')
        if not gs.game_over:
            self.lbl_cur.config(text=f'当前出牌: P{cur+1}')
            self.lbl_status.config(text=f'P{cur+1} 思考中...')
        if self.game.finished_order:
            self.lbl_rank.config(text=f'已出完: {" → ".join(f"P{p+1}" for p in self.game.finished_order)}')
        for pid in range(4):
            player = self.game.get_player(pid)
            team = 1 if pid in [0, 2] else 2
            base_bg = BG_TEAM1 if team == 1 else BG_TEAM2
            bg = BG_ACTIVE if (pid == cur and not gs.game_over) else base_bg
            self.pf[pid].config(bg=bg)
            self.lbl_info[pid].config(bg=bg)
            self.lbl_action[pid].config(bg=bg)
            self.cv_played[pid].config(bg=bg)
            self.cv_hand[pid].config(bg=bg)
            pos_s = f' [第{player.position}游]' if player.is_out() and player.position else ''
            self.lbl_info[pid].config(text=f'手牌: {len(player.hand)}张{pos_s}')
            self.lbl_action[pid].config(text=self.p_last_action[pid])
            self.cv_played[pid].draw_cards(self.p_last_cards[pid])
            self.cv_hand[pid].draw_cards(player.hand)

    def _cs(self, cards):
        try: return ' '.join(str(c) for c in cards) if cards else '无'
        except: return f'{len(cards)}张'

    def _log(self, msg):
        self.log_box.config(state='normal')
        self.log_box.insert('end', msg + '\n')
        self.log_box.see('end')
        self.log_box.config(state='disabled')


def main():
    root = tk.Tk()
    root.geometry('1080x800')
    GameGUI(root)
    root.mainloop()


if __name__ == '__main__':
    main()
