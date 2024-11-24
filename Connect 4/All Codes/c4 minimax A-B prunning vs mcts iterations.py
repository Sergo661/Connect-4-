import numpy as np
import random
import pygame
import sys
import math
import time

BLUE = (0, 0, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
YELLOW = (255, 255, 0)

ROW_COUNT = 6
COLUMN_COUNT = 7

PLAYER = 0
AI = 1

EMPTY = 0
PLAYER_PIECE = 1
AI_PIECE = 2

WINDOW_LENGTH = 4


def create_board():
    board = np.zeros((ROW_COUNT, COLUMN_COUNT))
    return board


def drop_piece(board, row, col, piece):
    board[row][col] = piece


def is_valid_location(board, col):
    return board[ROW_COUNT - 1][col] == 0


def get_next_open_row(board, col):
    for r in range(ROW_COUNT):
        if board[r][col] == 0:
            return r


def print_board(board):
    print(np.flip(board, 0))


def winning_move(board, piece):
    for c in range(COLUMN_COUNT - 3):
        for r in range(ROW_COUNT):
            if board[r][c] == piece and board[r][c + 1] == piece and board[r][c + 2] == piece and board[r][
                c + 3] == piece:
                return True

    for c in range(COLUMN_COUNT):
        for r in range(ROW_COUNT - 3):
            if board[r][c] == piece and board[r + 1][c] == piece and board[r + 2][c] == piece and board[r + 3][
                c] == piece:
                return True

    for c in range(COLUMN_COUNT - 3):
        for r in range(ROW_COUNT - 3):
            if board[r][c] == piece and board[r + 1][c + 1] == piece and board[r + 2][c + 2] == piece and board[r + 3][
                c + 3] == piece:
                return True

    for c in range(COLUMN_COUNT - 3):
        for r in range(3, ROW_COUNT):
            if board[r][c] == piece and board[r - 1][c + 1] == piece and board[r - 2][c + 2] == piece and board[r - 3][
                c + 3] == piece:
                return True


def evaluate_window(window, piece):
    score = 0
    opp_piece = PLAYER_PIECE if piece == AI_PIECE else AI_PIECE

    if window.count(piece) == 4:
        score += 100
    elif window.count(piece) == 3 and window.count(EMPTY) == 1:
        score += 5
    elif window.count(piece) == 2 and window.count(EMPTY) == 2:
        score += 2

    if window.count(opp_piece) == 3 and window.count(EMPTY) == 1:
        score -= 4

    return score


def score_position(board, piece):
    score = 0

    center_array = [int(i) for i in list(board[:, COLUMN_COUNT // 2])]
    center_count = center_array.count(piece)
    score += center_count * 3

    for r in range(ROW_COUNT):
        row_array = [int(i) for i in list(board[r, :])]
        for c in range(COLUMN_COUNT - 3):
            window = row_array[c:c + WINDOW_LENGTH]
            score += evaluate_window(window, piece)

    for c in range(COLUMN_COUNT):
        col_array = [int(i) for i in list(board[:, c])]
        for r in range(ROW_COUNT - 3):
            window = col_array[r:r + WINDOW_LENGTH]
            score += evaluate_window(window, piece)

    for r in range(ROW_COUNT - 3):
        for c in range(COLUMN_COUNT - 3):
            window = [board[r + i][c + i] for i in range(WINDOW_LENGTH)]
            score += evaluate_window(window, piece)

    for r in range(ROW_COUNT - 3):
        for c in range(COLUMN_COUNT - 3):
            window = [board[r + 3 - i][c + i] for i in range(WINDOW_LENGTH)]
            score += evaluate_window(window, piece)

    return score


def is_terminal_node(board):
    return winning_move(board, PLAYER_PIECE) or winning_move(board, AI_PIECE) or len(get_valid_locations(board)) == 0


class MonteCarloTreeSearch:
    def __init__(self, board, ai_piece):
        self.board = board
        self.ai_piece = ai_piece
        self.player_piece = 3 - ai_piece

    def get_best_move(self):
        valid_locations = get_valid_locations(self.board)
        if not valid_locations:
            return None

        best_move = None
        best_score = -math.inf

        for col in valid_locations:
            row = get_next_open_row(self.board, col)
            temp_board = self.board.copy()
            drop_piece(temp_board, row, col, self.ai_piece)
            score = self.monte_carlo_simulation(temp_board, self.ai_piece, self.player_piece)

            if score > best_score:
                best_score = score
                best_move = col

        return best_move

    def monte_carlo_simulation(self, board, ai_piece, player_piece):
        sim_count = 1000
        wins = 0

        for _ in range(sim_count):
            sim_board = board.copy()
            current_piece = ai_piece
            is_game_over = False

            while not is_game_over:
                valid_moves = get_valid_locations(sim_board)
                if not valid_moves:
                    break

                if current_piece == ai_piece:
                    col = random.choice(valid_moves)
                else:
                    col = random.choice(valid_moves)

                row = get_next_open_row(sim_board, col)
                drop_piece(sim_board, row, col, current_piece)

                if winning_move(sim_board, current_piece):
                    if current_piece == ai_piece:
                        wins += 1
                    break

                current_piece = player_piece if current_piece == ai_piece else ai_piece
                is_game_over = len(get_valid_locations(sim_board)) == 0

        return wins


def get_valid_locations(board):
    valid_locations = []
    for col in range(COLUMN_COUNT):
        if is_valid_location(board, col):
            valid_locations.append(col)
    return valid_locations


def minimax(board, depth, alpha, beta, maximizingPlayer):
    valid_locations = get_valid_locations(board)
    is_terminal = is_terminal_node(board)

    if depth == 0 or is_terminal:
        if is_terminal:
            if winning_move(board, AI_PIECE):
                return (None, 100000000000000)
            elif winning_move(board, PLAYER_PIECE):
                return (None, -10000000000000)
            else:  # Game is over, no more valid moves
                return (None, 0)
        else:  # Depth is zero
            return (None, score_position(board, AI_PIECE))

    if maximizingPlayer:
        value = -math.inf
        column = random.choice(valid_locations)
        for col in valid_locations:
            row = get_next_open_row(board, col)
            b_copy = board.copy()
            drop_piece(b_copy, row, col, AI_PIECE)
            new_score = minimax(b_copy, depth - 1, alpha, beta, False)[1]
            if new_score > value:
                value = new_score
                column = col
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        return column, value

    else:  # Minimizing player
        value = math.inf
        column = random.choice(valid_locations)
        for col in valid_locations:
            row = get_next_open_row(board, col)
            b_copy = board.copy()
            drop_piece(b_copy, row, col, PLAYER_PIECE)
            new_score = minimax(b_copy, depth - 1, alpha, beta, True)[1]
            if new_score < value:
                value = new_score
                column = col
            beta = min(beta, value)
            if alpha >= beta:
                break
        return column, value


def draw_board(board):
    for c in range(COLUMN_COUNT):
        for r in range(ROW_COUNT):
            pygame.draw.rect(screen, BLUE, (c * SQUARESIZE, r * SQUARESIZE + SQUARESIZE, SQUARESIZE, SQUARESIZE))
            pygame.draw.circle(screen, BLACK, (
                int(c * SQUARESIZE + SQUARESIZE / 2), int(r * SQUARESIZE + SQUARESIZE + SQUARESIZE / 2)), RADIUS)

    for c in range(COLUMN_COUNT):
        for r in range(ROW_COUNT):
            if board[r][c] == PLAYER_PIECE:
                pygame.draw.circle(screen, RED, (
                    int(c * SQUARESIZE + SQUARESIZE / 2), height - int(r * SQUARESIZE + SQUARESIZE / 2)), RADIUS)
            elif board[r][c] == AI_PIECE:
                pygame.draw.circle(screen, YELLOW, (
                    int(c * SQUARESIZE + SQUARESIZE / 2), height - int(r * SQUARESIZE + SQUARESIZE / 2)), RADIUS)
    pygame.display.update()

def mcts_move(board, piece, n_simulations=100):
    valid_locations = get_valid_locations(board)
    best_score = -float('inf')
    best_col = random.choice(valid_locations)

    for col in valid_locations:
        score = 0
        for _ in range(n_simulations):
            row = get_next_open_row(board, col)
            temp_board = board.copy()
            drop_piece(temp_board, row, col, piece)
            score += simulate(temp_board, piece)
        if score > best_score:
            best_score = score
            best_col = col

    return best_col

def simulate(board, piece):
    temp_board = board.copy()
    turn = piece
    while not is_terminal_node(temp_board):
        valid_locations = get_valid_locations(temp_board)
        if len(valid_locations) == 0:
            break
        col = random.choice(valid_locations)
        row = get_next_open_row(temp_board, col)
        drop_piece(temp_board, row, col, turn)
        turn = PLAYER_PIECE if turn == AI_PIECE else AI_PIECE
    return 1 if winning_move(temp_board, piece) else -1 if winning_move(temp_board, turn) else 0


board = create_board()
print_board(board)
game_over = False

pygame.init()

SQUARESIZE = 100

width = COLUMN_COUNT * SQUARESIZE
height = (ROW_COUNT + 1) * SQUARESIZE

size = (width, height)

RADIUS = int(SQUARESIZE / 2 - 5)

screen = pygame.display.set_mode(size)
draw_board(board)
pygame.display.update()

myfont = pygame.font.SysFont("monospace", 28)

turn = random.randint(PLAYER, AI)
minimax_ab_starts = 0
mcts_starts = 0

while not game_over:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            sys.exit()

    if turn == AI and not game_over:
        # AI makes a move using Minimax with Alpha-Beta Pruning
        col, _ = minimax(board, 5, -math.inf, math.inf, True)


        if col is not None and is_valid_location(board, col):
            row = get_next_open_row(board, col)
            drop_piece(board, row, col, AI_PIECE)
            if winning_move(board, AI_PIECE):
                label = myfont.render("AI Minimax A-B prunning wins!", 1, YELLOW)
                screen.blit(label, (40, 10))
                game_over = True
            turn = PLAYER

    elif turn == PLAYER and not game_over:
        # AI makes a random move
        col = mcts_move(board, PLAYER_PIECE)
        if col is not None and is_valid_location(board, col):
            row = get_next_open_row(board, col)
            drop_piece(board, row, col, PLAYER_PIECE)
            if winning_move(board, PLAYER_PIECE):
                label = myfont.render("AI MCTS wins!", 1, RED)
                screen.blit(label, (40, 10))
                game_over = True
            turn = AI

    draw_board(board)
    pygame.display.update()

    if game_over:
        pygame.time.wait(3000)

def run_games(number_of_games):
    minimax_ab_wins = 0
    mcts_wins = 0
    minimax_ab_total_time = 0.0
    mcts_total_time = 0.0
    minimax_ab_starts = 0
    mcts_starts = 0

    for _ in range(number_of_games):
        board = create_board()
        game_over = False
        turn = random.randint(PLAYER, AI)

        if turn == AI:
            minimax_ab_starts += 1
        else:
            mcts_starts += 1

        while not game_over:
            if turn == AI:
                start_time = time.perf_counter()
                col, _ = minimax(board, 5, -math.inf, math.inf, True)
                minimax_ab_time = time.perf_counter() - start_time
                minimax_ab_total_time += minimax_ab_time

                if col is not None and is_valid_location(board, col):
                    row = get_next_open_row(board, col)
                    drop_piece(board, row, col, AI_PIECE)
                    if winning_move(board, AI_PIECE):
                        minimax_ab_wins += 1
                        game_over = True
                turn = PLAYER

            elif turn == PLAYER:
                start_time = time.perf_counter()
                col = mcts_move(board, PLAYER_PIECE)
                mcts_time = time.perf_counter() - start_time
                mcts_total_time += mcts_time

                if col is not None and is_valid_location(board, col):
                    row = get_next_open_row(board, col)
                    drop_piece(board, row, col, PLAYER_PIECE)
                    if winning_move(board, PLAYER_PIECE):
                        mcts_wins += 1
                        game_over = True
                turn = AI

    print(f"Minimax A-B: {minimax_ab_wins} wins, {minimax_ab_starts} starts, Total Time: {minimax_ab_total_time:.6f} seconds")
    print(f"MCTS: {mcts_wins} wins, {mcts_starts} starts, Total Time: {mcts_total_time:.6f} seconds")

# Run the games
run_games(100)
