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


board = create_board()
print_board(board)
game_over = False



turn = random.randint(PLAYER, AI)
def random_move(board):
    valid_locations = get_valid_locations(board)
    return random.choice(valid_locations) if valid_locations else None
mcts = MonteCarloTreeSearch(board, AI_PIECE)




def play_game():
    board = create_board()
    turn = random.randint(PLAYER, AI)
    ai_mcts_time = 0.0
    ai_random_time = 0.0
    starting_player = turn
    game_over = False
    mcts = MonteCarloTreeSearch(board, AI_PIECE)

    while not game_over:
        start_time = time.time()

        if turn == AI:
            # AI using Monte Carlo Tree Search makes a move
            col = mcts.get_best_move()
            ai_mcts_time += time.time() - start_time
        else:
            # AI using random move makes a move
            col = random_move(board)
            ai_random_time += time.time() - start_time

        if col is not None and is_valid_location(board, col):
            row = get_next_open_row(board, col)
            drop_piece(board, row, col, AI_PIECE if turn == AI else PLAYER_PIECE)

            if winning_move(board, AI_PIECE if turn == AI else PLAYER_PIECE):
                game_over = True

        turn = 1 - turn  # Switch player

    ai_mcts_win = winning_move(board, AI_PIECE)
    ai_random_win = winning_move(board, PLAYER_PIECE)
    return ai_mcts_win, ai_random_win, ai_mcts_time, ai_random_time, starting_player

# Main game loop and results tracking
ai_mcts_wins = 0
ai_mcts_start_wins = 0
ai_random_wins = 0
ai_random_start_wins = 0
ai_mcts_total_time = 0.0
ai_random_total_time = 0.0

for _ in range(100):
    ai_mcts_win, ai_random_win, ai_mcts_time, ai_random_time, starter = play_game()
    ai_mcts_total_time += ai_mcts_time
    ai_random_total_time += ai_random_time

    if ai_mcts_win:
        ai_mcts_wins += 1
        if starter == AI:
            ai_mcts_start_wins += 1
    elif ai_random_win:
        ai_random_wins += 1
        if starter == PLAYER:
            ai_random_start_wins += 1

print("AI MCTS Wins:", ai_mcts_wins, "Wins when started:", ai_mcts_start_wins, "Total time taken:", round(ai_mcts_total_time,2), "seconds")
print("AI Random Wins:", ai_random_wins, "Wins when started:", ai_random_start_wins, "Total time taken:", round(ai_random_total_time,2), "seconds")
print("Total time taken:", round(ai_mcts_total_time + ai_random_total_time,2), "seconds")