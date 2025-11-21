import numpy as np
import pygame as pg
import time
import random
import math
# 确保 utility_func.py 在同一目录下
from utility_func import draw_board, click2index, draw_stone, draw_highlighted_stone, print_text, print_winner, print_turn

BOARD_SIZE = 15

#############################################################################
# 1. 核心逻辑：检查胜负 (check_winner)
#############################################################################

def check_winner(board):
    """
    检查棋盘上是否有获胜者。
    规则:
    - 必须是连续 5 颗同色棋子 (Horizontal, Vertical, Main Diagonal)。
    - 刚好 5 颗 (Freestyle 规则，长连不算赢，但在五子棋简单实现中通常 5 个以上也算赢，
      为了符合题目“exactly 5”的要求，我们需要检查前后是否被同色棋子包围)
    
    返回:
    1: 黑棋胜
    -1: 白棋胜
    2: 平局 (棋盘满了且无胜者)
    0: 游戏继续
    """
    directions = [(0, 1), (1, 0), (1, 1)] # 横向，纵向，主对角线 (三角形棋盘只有这三个有效方向)
    
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            stone = board[r, c]
            
            # 跳过空位(0)和无效位(5)
            if stone == 0 or stone == 5:
                continue
                
            for dr, dc in directions:
                # 检查是否有足够的空间放下5子
                if 0 <= r + 4*dr < BOARD_SIZE and 0 <= c + 4*dc < BOARD_SIZE:
                    # 提取这5个位置的棋子
                    chain = [board[r + i*dr, c + i*dc] for i in range(5)]
                    
                    # 如果全色相同
                    if all(s == stone for s in chain):
                        # --- 检查 "Exactly 5" 规则 (排除长连) ---
                        
                        # 检查连线头部的前一个位置
                        prev_r, prev_c = r - dr, c - dc
                        if 0 <= prev_r < BOARD_SIZE and 0 <= prev_c < BOARD_SIZE:
                            if board[prev_r, prev_c] == stone:
                                continue # 如果前面还是同色，说明是6连或更多，跳过
                        
                        # 检查连线尾部的后一个位置
                        next_r, next_c = r + 5*dr, c + 5*dc
                        if 0 <= next_r < BOARD_SIZE and 0 <= next_c < BOARD_SIZE:
                            if board[next_r, next_c] == stone:
                                continue # 如果后面还是同色，跳过
                                
                        return stone

    # 检查平局：如果没有 0 了（只剩下 1, -1, 5）
    if not np.any(board == 0):
        return 2 # 平局
        
    return 0

#############################################################################
# 2. 增强版 AI (Smart MCTS)
#############################################################################

def get_legal_moves_optimized(board):
    """
    优化点1：局部搜索。
    只获取棋盘上现有棋子周围 2 格范围内的空位。
    这解决了“AI下得很分散”的问题，强制它在有棋子的地方纠缠。
    """
    # 获取所有非空位置 (黑棋1, 白棋-1)
    occupied = list(zip(*np.where((board == 1) | (board == -1))))
    
    # 如果棋盘是空的（开局），直接下天元附近，不要遍历全图
    if not occupied:
        return [(7, 7)]
    
    legal_moves = set()
    rows, cols = board.shape
    
    # 遍历所有已存在的棋子，把它们周围的空位加入候选
    for r, c in occupied:
        # 检查周围 2 格的范围 (Range -2 to +3)
        r_min = max(0, r - 2)
        r_max = min(rows, r + 3)
        c_min = max(0, c - 2)
        c_max = min(cols, c + 3)
        
        for i in range(r_min, r_max):
            for j in range(c_min, c_max):
                # 必须是空位(0)且不是无效区域(5)
                if board[i, j] == 0:
                    legal_moves.add((i, j))
                    
    return list(legal_moves)

def find_immediate_threat(board, color):
    """
    优化点2：规则层（直觉）。
    在运行 MCTS 之前，先扫描是否存在“一步致胜”或“必须防守”的点。
    """
    # 获取局部范围内的空位
    possible_moves = get_legal_moves_optimized(board)
    
    # 1. 进攻：检查自己是否能一步赢
    for move in possible_moves:
        board[move] = color
        if check_winner(board) == color:
            board[move] = 0 # 恢复棋盘
            return move
        board[move] = 0 # 恢复棋盘

    # 2. 防守：检查对手是否下一步能赢（冲四/活四）
    opponent = -1 * color
    for move in possible_moves:
        board[move] = opponent # 假设对手下在这里
        if check_winner(board) == opponent:
            board[move] = 0 # 恢复棋盘
            return move # 必须堵住！
        board[move] = 0 # 恢复棋盘
        
    return None

class MCTSNode:
    def __init__(self, board_state, parent=None, move=None, player_color=1):
        self.board = board_state
        self.parent = parent
        self.move = move 
        self.player_color = player_color # 刚刚下这一步棋的玩家
        self.children = []
        self.wins = 0
        self.visits = 0
        # 使用优化的获取函数，减少分支数量
        self.untried_moves = get_legal_moves_optimized(self.board)

    def is_terminal(self):
        return check_winner(self.board) != 0

    def expand(self):
        move = self.untried_moves.pop()
        new_board = self.board.copy()
        # 下一步轮到对手下
        next_player = -1 * self.player_color 
        new_board[move] = next_player
        
        child_node = MCTSNode(new_board, parent=self, move=move, player_color=next_player)
        self.children.append(child_node)
        return child_node

    def best_child(self, c_param=1.4):
        # UCB1 公式
        choices_weights = [
            (child.wins / child.visits) + c_param * np.sqrt((2 * np.log(self.visits) / child.visits))
            for child in self.children
        ]
        return self.children[np.argmax(choices_weights)]

    def rollout(self):
        current_board = self.board.copy()
        current_player = -1 * self.player_color # 模拟开始，轮到下一个人
        
        winner = 0
        depth = 0
        max_depth = 15 # 限制深度，让 AI 能模拟更多次局势，而不是深究某一种
        
        while True:
            winner = check_winner(current_board)
            if winner != 0 or depth > max_depth:
                break
            
            # 快速走子：只在局部随机，不全图随机
            valid_moves = get_legal_moves_optimized(current_board)
            if not valid_moves:
                winner = 2
                break
            
            # 随机选一步
            move = random.choice(valid_moves)
            current_board[move] = current_player
            current_player *= -1
            depth += 1
            
        return winner

    def backpropagate(self, result):
        self.visits += 1
        if result == 2: 
            self.wins += 0.5
        elif result == self.player_color:
            self.wins += 1
        elif result != 0: 
            # 优化点3：加大输棋的惩罚，让 AI 极力避免让对手赢的路径
            self.wins -= 10 
            
        if self.parent:
            self.parent.backpropagate(result)


def computer_move(board, color):
    """
    AI 主函数。
    """
    # --- 1. 规则判断 (Reflex) ---
    # 只要有必胜或必救点，直接返回，不浪费时间做 MCTS
    urgent_move = find_immediate_threat(board, color)
    if urgent_move:
        print(f"AI 规则层触发 (进攻/防守): {urgent_move}")
        return urgent_move

    # --- 2. MCTS 搜索 (Strategy) ---
    start_time = time.time()
    time_limit = 4.8 # 预留 0.2 秒防止超时
    
    # 根节点的状态是“对手刚下完”，所以 player_color 是 opponent
    root = MCTSNode(board, player_color=-1 * color)
    
    iter_count = 0
    while time.time() - start_time < time_limit:
        node = root
        
        # Selection (选择)
        while node.children and not node.untried_moves:
            node = node.best_child()
            
        # Expansion (扩展)
        if node.untried_moves:
            node = node.expand()
            
        # Simulation (模拟)
        result = node.rollout()
        
        # Backpropagation (回溯)
        node.backpropagate(result)
        iter_count += 1
        
    # 容错：如果没有生成子节点（极少见，除非棋盘满了）
    if not root.children:
        legal = get_legal_moves_optimized(board)
        return random.choice(legal) if legal else None
        
    # 选择访问次数最多的节点作为最佳移动（最稳健）
    best_move_node = max(root.children, key=lambda c: c.visits)
    
    print(f"AI 思考次数: {iter_count}, 最佳落子: {best_move_node.move}, 胜率评分: {best_move_node.wins}/{best_move_node.visits}")
    return best_move_node.move

#############################################################################
# 3. 游戏主循环
#############################################################################

def main(player_is_black=True):
    
    # 初始化 Pygame
    pg.init()
    surface = draw_board()
    
    # 初始化棋盘
    board = np.zeros((15,15), dtype=int)
    # 标记上半三角为无效区域(5)
    board[np.triu_indices(15, k=1)] = 5
    board = np.flipud(board)
    
    # 1 = 黑棋, -1 = 白棋
    human_color = 1 if player_is_black else -1
    ai_color = -1 if player_is_black else 1
    
    current_turn = 1 # 黑棋先手
    
    indx_mem = (-10, -10) # 记录上一步位置
    running = True
    gameover = False
    winner = 0
    
    print_turn(surface, current_turn)
    
    # 如果电脑执黑，直接先手
    if not player_is_black:
        print("电脑正在思考...")
        # 电脑第一步一般下天元或者靠近中心
        move = (7, 7)
        
        board[move] = ai_color
        indx_mem = move
        draw_stone(surface, move, ai_color)
        draw_highlighted_stone(surface, move, ai_color)
        
        current_turn *= -1
        print_turn(surface, current_turn)
    
    while running:
        # 处理事件
        for event in pg.event.get():
            if event.type == pg.QUIT:
                running = False
                
            # --- 玩家回合 ---
            if event.type == pg.MOUSEBUTTONDOWN and not gameover and current_turn == human_color:
                
                idx = click2index(event.pos)
                
                # 如果点击有效且该位置为空
                if idx and board[idx] == 0:
                    
                    # 清除上一步的高亮
                    if indx_mem != (-10, -10):
                        prev_color = board[indx_mem]
                        draw_stone(surface, indx_mem, prev_color)
                    
                    # 玩家落子
                    board[idx] = human_color
                    indx_mem = idx
                    draw_highlighted_stone(surface, idx, human_color)
                    
                    # 检查玩家是否获胜
                    winner = check_winner(board)
                    if winner != 0:
                        gameover = True
                        print_winner(surface, winner)
                    else:
                        # 切换到电脑回合
                        current_turn *= -1
                        print_turn(surface, current_turn)
                        # 强制刷新界面，防止电脑思考时界面卡死
                        pg.display.update() 
                        
                        # --- 电脑回合 ---
                        if not gameover:
                            ai_move = computer_move(board, ai_color)
                            
                            if ai_move:
                                # 清除玩家的高亮
                                draw_stone(surface, indx_mem, human_color)
                                
                                # 电脑落子
                                board[ai_move] = ai_color
                                indx_mem = ai_move
                                draw_highlighted_stone(surface, ai_move, ai_color)
                                
                                # 检查电脑是否获胜
                                winner = check_winner(board)
                                if winner != 0:
                                    gameover = True
                                    print_winner(surface, winner)
                                else:
                                    current_turn *= -1
                                    print_turn(surface, current_turn)

    pg.quit()

if __name__ == '__main__':
    # 默认玩家执黑先手，改为 False 则电脑先手
    main(player_is_black=True)