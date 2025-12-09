import math
import os
import random
import time
import numpy as np
import pygame as pg

# 设置窗口位置
os.environ['SDL_VIDEO_WINDOW_POS'] = "%d,%d" % (20, 80)

# =============================================================================
# 1. 全局配置
# =============================================================================
COLOR_LINE = [153, 153, 153]
COLOR_BOARD = [241, 196, 15]
COLOR_BLACK = [0, 0, 0]
COLOR_DARK_GRAY = [75, 75, 75]
COLOR_WHITE = [255, 255, 255]
COLOR_LIGHT_GRAY = [235, 235, 235]
COLOR_RED = [255, 0, 0]
COLOR_GREEN = [0, 255, 0]
COLOR_BLUE = [0, 0, 255]

WINDOW_SIZE = 720
PADDING = 36
TRIANGLE_SPAN = 15

SEP_X = (WINDOW_SIZE - 2 * PADDING) / (TRIANGLE_SPAN - 1)
SEP_Y = SEP_X * np.sqrt(3) / 2
PAD_X = PADDING
PAD_Y = WINDOW_SIZE / 2 - (WINDOW_SIZE - 2 * PADDING) * np.sqrt(3) / 4
PIECE_RADIUS = SEP_X * 0.3


# =============================================================================
# 2. 辅助绘图函数
# =============================================================================

def draw_board():
    surface = pg.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
    pg.display.set_caption("Gomuku (Triangle Board)")
    surface.fill(COLOR_BOARD)

    for i in range(TRIANGLE_SPAN - 1):
        # 水平线
        pg.draw.line(surface, COLOR_LINE,
                     (PADDING + i * SEP_X / 2, PAD_Y + i * SEP_Y),
                     (WINDOW_SIZE - PADDING - i * SEP_X / 2, PAD_Y + i * SEP_Y), 3)
        # 左斜线 (TL-BR)
        pg.draw.line(surface, COLOR_LINE,
                     (PAD_X + i * SEP_X, PAD_Y),
                     (WINDOW_SIZE - PADDING - (TRIANGLE_SPAN - i - 1) * SEP_X / 2,
                      PAD_Y + (TRIANGLE_SPAN - i - 1) * SEP_Y), 3)
        # 右斜线 (TR-BL)
        pg.draw.line(surface, COLOR_LINE,
                     (WINDOW_SIZE - PAD_X - i * SEP_X, PAD_Y),
                     (PADDING + (TRIANGLE_SPAN - i - 1) * SEP_X / 2,
                      PAD_Y + (TRIANGLE_SPAN - i - 1) * SEP_Y), 3)
    pg.display.update()
    return surface


def click2index(pos):
    x, y = pos
    if ((y > PAD_Y - PIECE_RADIUS) and
            (x - PAD_X) > (y - PAD_Y - PIECE_RADIUS) / np.sqrt(3) and
            (x - WINDOW_SIZE + PAD_X) < (PAD_Y + PIECE_RADIUS - y) / np.sqrt(3)):
        u = round((y - PAD_Y) / SEP_Y)
        v = round((x - PAD_X - u * SEP_X / 2) / SEP_X)
        return (u, v)
    return False


def get_stone_screen_pos(pos):
    u, v = pos
    x = PAD_X + u * SEP_X / 2 + v * SEP_X
    y = PAD_Y + u * SEP_Y
    return int(x), int(y)


def draw_stone(surface, pos, color=0):
    x, y = get_stone_screen_pos(pos)
    if color == 1:
        pg.draw.circle(surface, COLOR_BLACK, [x, y], PIECE_RADIUS, 0)
        pg.draw.circle(surface, COLOR_DARK_GRAY, [x, y], PIECE_RADIUS, 2)
    elif color == -1:
        pg.draw.circle(surface, COLOR_WHITE, [x, y], PIECE_RADIUS, 0)
        pg.draw.circle(surface, COLOR_LIGHT_GRAY, [x, y], PIECE_RADIUS, 2)
    pg.display.update()


def draw_highlighted_stone(surface, pos, color=0):
    x, y = get_stone_screen_pos(pos)
    border_color = COLOR_RED
    if color == 1:
        pg.draw.circle(surface, COLOR_BLACK, [x, y], PIECE_RADIUS, 0)
        pg.draw.circle(surface, border_color, [x, y], PIECE_RADIUS, 3)
    elif color == -1:
        pg.draw.circle(surface, COLOR_WHITE, [x, y], PIECE_RADIUS, 0)
        pg.draw.circle(surface, border_color, [x, y], PIECE_RADIUS, 3)
    pg.display.update()


def print_text(surface, msg, color=COLOR_BLACK):
    pg.draw.rect(surface, COLOR_BOARD, pg.Rect(0, 0, 350, 60), 0)
    font = pg.font.SysFont('arial', 32, bold=True)
    text = font.render(msg, True, color)
    surface.blit(text, (10, 10))
    pg.display.update()


def update_status_display(surface, winner=0, turn=1, thinking=False):
    if thinking:
        print_text(surface, "AI Thinking...", COLOR_RED)
        return

    if winner == 1:
        print_text(surface, "Black Wins!", COLOR_BLACK)
    elif winner == -1:
        print_text(surface, "White Wins!", COLOR_BLUE)
    elif winner == 2:
        print_text(surface, "Draw!", COLOR_RED)
    else:
        if turn == 1:
            print_text(surface, "Black's Turn", COLOR_BLACK)
        else:
            print_text(surface, "White's Turn", COLOR_WHITE)


# =============================================================================
# 3. 游戏核心逻辑
# =============================================================================

def check_winner(board):
    rows, cols = board.shape
    for u in range(rows):
        for v in range(cols):
            color = board[u][v]
            if color == 1 or color == -1:
                if v + 4 < cols and all(board[u][v + i] == color for i in range(1, 5)):
                    return color
                if u + 4 < rows and all(board[u + i][v] == color for i in range(1, 5)):
                    return color
                if u + 4 < rows and v - 4 >= 0 and all(board[u + i][v - i] == color for i in range(1, 5)):
                    return color
    if not np.any(board == 0):
        return 2
    return 0


def get_neighboring_moves(board, distance=2):
    rows, cols = board.shape
    occupied = np.argwhere((board == 1) | (board == -1))
    if len(occupied) == 0: return [(6, 4)] # 默认开局天元

    candidates = set()
    for ux, uy in occupied:
        r_min, r_max = max(0, ux - distance), min(rows, ux + distance + 1)
        c_min, c_max = max(0, uy - distance), min(cols, uy + distance + 1)
        sub_board = board[r_min:r_max, c_min:c_max]
        local_empties = np.argwhere(sub_board == 0)
        for er, ec in local_empties:
            candidates.add((r_min + er, c_min + ec))
    return list(candidates)


def quick_check_win_at_move(board, move, color):
    u, v = move
    rows, cols = board.shape
    directions = [(0, 1), (1, 0), (1, -1)]

    for dr, dc in directions:
        count = 1
        for i in range(1, 5):
            r, c = u + dr * i, v + dc * i
            if 0 <= r < rows and 0 <= c < cols and board[r][c] == color:
                count += 1
            else:
                break
        for i in range(1, 5):
            r, c = u - dr * i, v - dc * i
            if 0 <= r < rows and 0 <= c < cols and board[r][c] == color:
                count += 1
            else:
                break
        if count >= 5: return True
    return False

def evaluate_pos(board, r, c, color):
    """
    [功能增强] 评估单个空位的棋型价值。
    既可以评估自己的进攻（color=my_color），也可以评估对手的威胁（color=opponent_color）。
    返回: (活三数量, 冲四数量, 活四数量)
    """
    rows, cols = board.shape
    directions = [(0, 1), (1, 0), (1, -1)]
    
    live_threes = 0
    sleep_fours = 0 # 冲四 (被堵一头)
    live_fours = 0  # 活四 (两头空)

    for dr, dc in directions:
        # 统计该方向上加上 (r,c) 后能连多少子
        # 向前搜
        len_pos = 0
        blocked_pos = False
        for k in range(1, 5):
            nr, nc = r + k*dr, c + k*dc
            if 0 <= nr < rows and 0 <= nc < cols:
                if board[nr, nc] == color:
                    len_pos += 1
                elif board[nr, nc] == 0:
                    break
                else: # 遇到异色或边界
                    blocked_pos = True
                    break
            else:
                blocked_pos = True
                break
                
        # 向后搜
        len_neg = 0
        blocked_neg = False
        for k in range(1, 5):
            nr, nc = r - k*dr, c - k*dc
            if 0 <= nr < rows and 0 <= nc < cols:
                if board[nr, nc] == color:
                    len_neg += 1
                elif board[nr, nc] == 0:
                    break
                else:
                    blocked_neg = True
                    break
            else:
                blocked_neg = True
                break
        
        total_len = 1 + len_pos + len_neg
        
        if total_len == 4:
            if not blocked_pos and not blocked_neg:
                live_fours += 1
            elif not (blocked_pos and blocked_neg):
                sleep_fours += 1
        elif total_len == 3:
            # 活三必须两头都空
            if not blocked_pos and not blocked_neg:
                live_threes += 1
                
    return live_threes, sleep_fours, live_fours


def check_critical_threats(board, opponent_color):
    """
    [策略优化] 防守逻辑：
    1. 必须防冲四 (下一步输)
    2. 必须防活三 (下一步变活四)
    3. [新增] 必须防双三/四三 (双杀点)
    """
    candidates = get_neighboring_moves(board, distance=1)

    # 1. 检查致命一击 (对方下这一步就赢) - 优先级最高
    for move in candidates:
        board[move] = opponent_color
        if quick_check_win_at_move(board, move, opponent_color):
            board[move] = 0
            print(f"AI 紧急防守 (阻挡冲四): {move}")
            return move
        board[move] = 0

    # 2. 检查危险棋型 (双杀、活三) - 优先级次高
    best_defense_score = -1
    best_defense_move = None

    for move in candidates:
        # 评估对手下在这里的威力
        l3, s4, l4 = evaluate_pos(board, move[0], move[1], opponent_color)
        
        score = 0
        if l4 >= 1: score = 20000 # 活四 = 必死
        elif s4 >= 2: score = 15000 # 双冲四 = 必死
        elif s4 >= 1 and l3 >= 1: score = 10000 # 四三 = 必死
        elif l3 >= 2: score = 8000 # 双三 = 必死
        elif l3 >= 1: score = 2000 # 单活三 (必须堵)
        elif s4 >= 1: score = 1000 # 单冲四 (通常第一步检测已覆盖，这里作为漏网补充)
        
        if score > best_defense_score:
            best_defense_score = score
            best_defense_move = move
            
    # 只有当威胁足够大时才强制防守
    if best_defense_score >= 2000:
        print(f"AI 战术防守 (分值{best_defense_score}): {best_defense_move}")
        return best_defense_move

    return None

def find_best_attack_move(board, my_color):
    """
    [新增策略] 主动进攻逻辑：
    寻找能形成 活四、冲四、双三 的点。
    """
    candidates = get_neighboring_moves(board, distance=1)
    best_score = -1
    best_move = None

    for move in candidates:
        # 评估我方下在这里的威力
        l3, s4, l4 = evaluate_pos(board, move[0], move[1], my_color)
        
        score = 0
        if l4 >= 1: score = 10000 # 活四 (下完就赢了)
        elif s4 >= 2: score = 8000 # 双冲四
        elif s4 >= 1 and l3 >= 1: score = 7000 # 四三
        elif l3 >= 2: score = 5000 # 双三
        elif l3 >= 1: score = 1000 # 活三 (好棋)
        elif s4 >= 1: score = 500 # 冲四 (进攻性)
        
        if score > best_score:
            best_score = score
            best_move = move

    # 如果发现了有价值的进攻点 (至少是冲四或活三)，返回该点
    if best_score >= 500:
        print(f"AI 主动进攻 (分值{best_score}): {best_move}")
        return best_move
    
    return None

class MCTSNode:
    def __init__(self, board, parent=None, move=None, player=0):
        self.board = board
        self.parent = parent
        self.move = move
        self.player = player
        self.children = []
        self.wins = 0
        self.visits = 0
        self.untried_moves = get_neighboring_moves(board, distance=1)

    def ucb1(self, exploration_weight=1.41):
        if self.visits == 0: return float('inf')
        return (self.wins / self.visits) + exploration_weight * math.sqrt(math.log(self.parent.visits) / self.visits)

    def is_fully_expanded(self):
        return len(self.untried_moves) == 0

    def is_terminal_node(self):
        return check_winner(self.board) != 0

    def best_child(self):
        return max(self.children, key=lambda node: node.ucb1())


def computer_move(board, turn):
    """优化后的 AI (攻守平衡版)"""
    legal_moves = get_neighboring_moves(board, distance=2)
    if not legal_moves: return (7, 7) # 中心开局

    # 1. 必胜判断 (直接赢)
    for move in legal_moves:
        board[move] = turn
        if quick_check_win_at_move(board, move, turn):
            board[move] = 0
            print(f"AI 绝杀: {move}")
            return move
        board[move] = 0

    # 2. 关键防守 (防对手必胜/双杀) - 优先级高
    opponent = -turn
    defense_move = check_critical_threats(board, opponent)
    if defense_move:
        return defense_move

    # 3. 关键进攻 (如果没有防守压力，尝试造杀) - 新增
    attack_move = find_best_attack_move(board, turn)
    if attack_move:
        return attack_move

    # 4. MCTS (局面平稳时，通过搜索找最优解)
    time_limit = 2.8
    start_time = time.time()
    root = MCTSNode(board=board.copy(), parent=None, move=None, player=-turn)
    sim_count = 0

    while time.time() - start_time < time_limit:
        if sim_count % 50 == 0: # 稍微降低 pump 频率以提高速度
            pg.event.pump()

        sim_count += 1
        node = root

        # A. Selection
        while node.is_fully_expanded() and len(node.children) > 0:
            node = node.best_child()

        # B. Expansion
        if not node.is_fully_expanded():
            if check_winner(node.board) == 0:
                # 随机选择一个未尝试的节点扩展
                idx = random.randint(0, len(node.untried_moves) - 1)
                move = node.untried_moves.pop(idx)
                
                new_board = node.board.copy()
                new_board[move] = -node.player
                new_node = MCTSNode(new_board, parent=node, move=move, player=-node.player)
                node.children.append(new_node)
                node = new_node

        # C. Simulation
        sim_board = node.board.copy()
        sim_turn = node.player
        winner = 0
        
        # 检查当前节点是否已分胜负
        if quick_check_win_at_move(sim_board, node.move, node.player) if node.move else False:
            winner = node.player

        depth = 0
        max_depth = 12 # 稍微减小深度，增加模拟次数
        if winner == 0:
            candidates = get_neighboring_moves(sim_board, distance=1)
            while depth < max_depth and candidates:
                sim_turn = -sim_turn
                # 随机落子
                idx = random.randint(0, len(candidates) - 1)
                mv = candidates.pop(idx)
                
                if sim_board[mv] == 0:
                    sim_board[mv] = sim_turn
                    if quick_check_win_at_move(sim_board, mv, sim_turn):
                        winner = sim_turn
                        break
                depth += 1

        # D. Backpropagation
        while node is not None:
            node.visits += 1
            if winner == node.player:
                node.wins += 1
            elif winner == 0:
                node.wins += 0.1 # 平局给微小分值，鼓励不输
            else:
                node.wins -= 1 # 输了扣分 (可选)
            node = node.parent

    if not root.children: return random.choice(legal_moves)
    
    # 选择访问次数最多的
    best_move = max(root.children, key=lambda n: n.visits).move
    print(f"AI MCTS 推荐: {best_move} (Sims: {sim_count})")
    return best_move


# =============================================================================
# 4. 主程序
# =============================================================================

def main(player_is_black=True):
    pg.init()
    surface = draw_board()

    board = np.zeros((15, 15), dtype=int)
    board[np.triu_indices(15, k=1)] = 2
    board = np.flipud(board)

    running = True
    gameover = False

    if player_is_black:
        player_color = 1
        computer_color = -1
    else:
        player_color = -1
        computer_color = 1

    current_turn = 1
    indx_mem = (-10, -10)

    update_status_display(surface, winner=0, turn=current_turn)

    while running:
        if not gameover and current_turn == computer_color:
            update_status_display(surface, thinking=True)
            pg.event.pump()
            pg.display.update()

            ai_move = computer_move(board, computer_color)

            if board[ai_move] == 0:
                draw_stone(surface, indx_mem, -current_turn)
                board[ai_move] = computer_color
                indx_mem = ai_move
                draw_highlighted_stone(surface, ai_move, computer_color)

                winner = check_winner(board)
                if winner != 0:
                    update_status_display(surface, winner=winner)
                    gameover = True
                else:
                    current_turn = player_color
                    update_status_display(surface, winner=0, turn=current_turn)

        for event in pg.event.get():
            if event.type == pg.QUIT:
                running = False

            if event.type == pg.MOUSEBUTTONDOWN and not gameover:
                if current_turn == player_color:
                    indx = click2index(event.pos)

                    if indx and board[indx] == 0:
                        draw_stone(surface, indx_mem, -current_turn)
                        board[indx] = player_color
                        indx_mem = indx
                        draw_highlighted_stone(surface, indx, player_color)

                        winner = check_winner(board)
                        if winner != 0:
                            update_status_display(surface, winner=winner)
                            gameover = True
                        else:
                            current_turn = computer_color
                            pass

    pg.quit()


if __name__ == '__main__':
    main(player_is_black=False)