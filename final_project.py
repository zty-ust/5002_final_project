import numpy as np
import pygame as pg
import time
import random
import math
import copy
# 导入项目提供的绘图工具 (必须确保 utility_func.py 在同一目录下)
from utility_func import draw_board, click2index, draw_stone, draw_highlighted_stone, print_text, print_winner, print_turn

BOARD_SIZE = 15

#############################################################################
# 1. 基础逻辑：胜负判断 & 威胁检测
#############################################################################

def check_winner(board):
    """
    检查是否有玩家获胜。
    返回: 1(黑胜), -1(白胜), 2(平局), 0(继续)
    """
    # 三角形棋盘的有效方向：横向、纵向、主对角线
    directions = [(0, 1), (1, 0), (1, 1)]
    
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            stone = board[r, c]
            if stone == 0 or stone == 5: # 跳过空位和无效区域
                continue
                
            for dr, dc in directions:
                if 0 <= r + 4*dr < BOARD_SIZE and 0 <= c + 4*dc < BOARD_SIZE:
                    chain = [board[r + i*dr, c + i*dc] for i in range(5)]
                    
                    if all(s == stone for s in chain):
                        # --- Exactly 5 规则 (排除长连) ---
                        prev_r, prev_c = r - dr, c - dc
                        if 0 <= prev_r < BOARD_SIZE and 0 <= prev_c < BOARD_SIZE:
                            if board[prev_r, prev_c] == stone:
                                continue 
                        
                        next_r, next_c = r + 5*dr, c + 5*dc
                        if 0 <= next_r < BOARD_SIZE and 0 <= next_c < BOARD_SIZE:
                            if board[next_r, next_c] == stone:
                                continue 
                                
                        return stone

    if not np.any(board == 0):
        return 2 # 平局
    return 0

def creates_open_four(board, r, c, color):
    """
    核心防守逻辑：检查如果在 (r,c) 落子，是否会形成“活四”(两头空的四连)。
    活四是必胜棋型，必须提前拦截（即在对手形成活三时就要堵）。
    """
    board[r, c] = color # 模拟落子
    directions = [(0, 1), (1, 0), (1, 1)]
    is_threat = False
    rows, cols = board.shape

    for dr, dc in directions:
        # 寻找连线起点
        start_r, start_c = r, c
        while 0 <= start_r - dr < rows and 0 <= start_c - dc < cols and board[start_r - dr, start_c - dc] == color:
            start_r -= dr
            start_c -= dc
        
        # 计算连线长度
        length = 0
        curr_r, curr_c = start_r, start_c
        while 0 <= curr_r < rows and 0 <= curr_c < cols and board[curr_r, curr_c] == color:
            length += 1
            curr_r += dr
            curr_c += dc
        
        # 如果连成4个，检查两头是否为空 (活四)
        if length == 4:
            head_r, head_c = start_r - dr, start_c - dc
            head_ok = (0 <= head_r < rows and 0 <= head_c < cols and board[head_r, head_c] == 0)
            
            tail_r, tail_c = curr_r, curr_c
            tail_ok = (0 <= tail_r < rows and 0 <= tail_c < cols and board[tail_r, tail_c] == 0)
            
            if head_ok and tail_ok:
                is_threat = True
                break
                
    board[r, c] = 0 # 恢复棋盘
    return is_threat

#############################################################################
# 2. 搜索优化：局部候选点生成
#############################################################################

def get_legal_moves_optimized(board):
    """
    只获取现有棋子周围 2 格范围内的空位。
    防止 AI 在无关紧要的角落落子，提高 MCTS 搜索深度。
    """
    occupied = list(zip(*np.where((board == 1) | (board == -1))))
    
    if not occupied:
        return [(7, 7)] # 开局下天元
    
    legal_moves = set()
    rows, cols = board.shape
    
    for r, c in occupied:
        # 搜索半径范围
        r_min, r_max = max(0, r - 2), min(rows, r + 3)
        c_min, c_max = max(0, c - 2), min(cols, c + 3)
        
        for i in range(r_min, r_max):
            for j in range(c_min, c_max):
                if board[i, j] == 0:
                    legal_moves.add((i, j))
                    
    return list(legal_moves)

#############################################################################
# 3. 规则层 (Reflex Layer)
#############################################################################

def find_immediate_threat(board, color):
    """
    在 MCTS 之前运行，处理“必胜”和“必救”局面。
    """
    possible_moves = get_legal_moves_optimized(board)
    opponent = -1 * color

    # 优先级 1: 我能赢吗？(5连)
    for move in possible_moves:
        board[move] = color
        if check_winner(board) == color:
            board[move] = 0
            return move
        board[move] = 0

    # 优先级 2: 对手要赢了吗？(堵冲四/活四)
    for move in possible_moves:
        board[move] = opponent
        if check_winner(board) == opponent:
            board[move] = 0
            print(f"防守：发现对手绝杀点 {move}")
            return move
        board[move] = 0

    # 优先级 3: 对手要形成活四了吗？(堵活三) -> 解决“连了四个才来堵”的问题
    for move in possible_moves:
        if creates_open_four(board, move[0], move[1], opponent):
            print(f"防守：拦截对手活三 {move}")
            return move
            
    return None

#############################################################################
# 4. MCTS (Strategy Layer)
#############################################################################

class MCTSNode:
    def __init__(self, board_state, parent=None, move=None, player_color=1):
        self.board = board_state
        self.parent = parent
        self.move = move
        self.player_color = player_color 
        self.children = []
        self.wins = 0
        self.visits = 0
        self.untried_moves = get_legal_moves_optimized(self.board)

    def is_terminal(self):
        return check_winner(self.board) != 0

    def expand(self):
        move = self.untried_moves.pop()
        new_board = self.board.copy()
        next_player = -1 * self.player_color
        new_board[move] = next_player
        
        child_node = MCTSNode(new_board, parent=self, move=move, player_color=next_player)
        self.children.append(child_node)
        return child_node

    def best_child(self, c_param=1.4):
        choices_weights = [
            (child.wins / child.visits) + c_param * np.sqrt((2 * np.log(self.visits) / child.visits))
            for child in self.children
        ]
        return self.children[np.argmax(choices_weights)]

    def rollout(self):
        current_board = self.board.copy()
        current_player = -1 * self.player_color
        
        winner = 0
        depth = 0
        max_depth = 15 # 深度限制，换取更多模拟次数
        
        while True:
            winner = check_winner(current_board)
            if winner != 0 or depth > max_depth:
                break
            
            valid_moves = get_legal_moves_optimized(current_board)
            if not valid_moves:
                winner = 2
                break
            
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
            self.wins -= 10 # 提高输棋惩罚，让AI更畏惧失败
            
        if self.parent:
            self.parent.backpropagate(result)

def computer_move(board, color):
    # 1. 规则判断 (Reflex)
    urgent_move = find_immediate_threat(board, color)
    if urgent_move:
        return urgent_move

    # 2. MCTS 搜索 (Strategy)
    start_time = time.time()
    time_limit = 4.8 # 5秒限制
    
    root = MCTSNode(board, player_color=-1 * color)
    
    iter_count = 0
    while time.time() - start_time < time_limit:
        node = root
        
        # Selection
        while node.children and not node.untried_moves:
            node = node.best_child()
            
        # Expansion
        if node.untried_moves:
            node = node.expand()
            
        # Simulation
        result = node.rollout()
        
        # Backpropagation
        node.backpropagate(result)
        iter_count += 1
        
    if not root.children:
        legal = get_legal_moves_optimized(board)
        return random.choice(legal) if legal else None
        
    best_move_node = max(root.children, key=lambda c: c.visits)
    print(f"AI 模拟次数: {iter_count}, 最佳落子: {best_move_node.move}")
    return best_move_node.move

#############################################################################
# 5. 主程序
#############################################################################

def main(player_is_black=True):
    pg.init()
    surface = draw_board()
    
    # 初始化三角形棋盘 (使用 15x15 矩阵，右上角标为 5)
    board = np.zeros((15,15), dtype=int)
    board[np.triu_indices(15, k=1)] = 5
    board = np.flipud(board)
    
    human_color = 1 if player_is_black else -1
    ai_color = -1 if player_is_black else 1
    
    current_turn = 1 # 黑棋先行
    
    indx_mem = (-10, -10)
    running = True
    gameover = False
    winner = 0
    
    print_turn(surface, current_turn)
    
    # 如果电脑执黑，直接第一手
    if not player_is_black:
        print_text(surface, "Thinking...", color=[0,0,0])
        pg.display.update()
        
        move = (7, 7)
        board[move] = ai_color
        indx_mem = move
        draw_stone(surface, move, ai_color)
        draw_highlighted_stone(surface, move, ai_color)
        current_turn *= -1
        print_turn(surface, current_turn)
    
    while running:
        for event in pg.event.get():
            if event.type == pg.QUIT:
                running = False
                
            if event.type == pg.MOUSEBUTTONDOWN and not gameover and current_turn == human_color:
                idx = click2index(event.pos)
                
                if idx and board[idx] == 0:
                    # 1. 玩家落子
                    if indx_mem != (-10, -10):
                        prev_color = board[indx_mem]
                        draw_stone(surface, indx_mem, prev_color)
                    
                    board[idx] = human_color
                    indx_mem = idx
                    draw_highlighted_stone(surface, idx, human_color)
                    
                    winner = check_winner(board)
                    if winner != 0:
                        gameover = True
                        print_winner(surface, winner)
                    else:
                        current_turn *= -1
                        print_turn(surface, current_turn)
                        print_text(surface, "Thinking...", color=[100,100,100])
                        pg.display.update() 
                        
                        # 2. 电脑落子
                        if not gameover:
                            ai_move = computer_move(board, ai_color)
                            if ai_move:
                                draw_stone(surface, indx_mem, human_color)
                                board[ai_move] = ai_color
                                indx_mem = ai_move
                                draw_highlighted_stone(surface, ai_move, ai_color)
                                
                                winner = check_winner(board)
                                if winner != 0:
                                    gameover = True
                                    print_winner(surface, winner)
                                else:
                                    current_turn *= -1
                                    print_turn(surface, current_turn)
    pg.quit()

if __name__ == '__main__':
    main(player_is_black=True)