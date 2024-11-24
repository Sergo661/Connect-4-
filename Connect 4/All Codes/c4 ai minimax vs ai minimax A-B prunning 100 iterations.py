import numpy as np
import random
import math
import time

ROW_COUNT = 6
COLUMN_COUNT = 7

PLAYER = 0
AI = 1

EMPTY = 0
PLAYER_PIECE = 1
AI_PIECE = 2

WINDOW_LENGTH = 4

# Memoization dictionary to store computed board states
memoization_dict = {}


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


def winning_move(board, piece):
    # Check horizontal locations for win
    for c in range(COLUMN_COUNT - 3):
        for r in range(ROW_COUNT):
            if board[r][c] == piece and board[r][c + 1] == piece and board[r][c + 2] == piece and board[r][
                c + 3] == piece:
                return True

    # Check vertical locations for win
    for c in range(COLUMN_COUNT):
        for r in range(ROW_COUNT - 3):
            if board[r][c] == piece and board[r + 1][c] == piece and board[r + 2][c] == piece and board[r + 3][
                c] == piece:
                return True

    # Check positively sloped diagonals
    for c in range(COLUMN_COUNT - 3):
        for r in range(ROW_COUNT - 3):
            if board[r][c] == piece and board[r + 1][c + 1] == piece and board[r + 2][c + 2] == piece and board[r + 3][
                c + 3] == piece:
                return True

    # Check negatively sloped diagonals
    for c in range(COLUMN_COUNT - 3):
        for r in range(3, ROW_COUNT):
            if board[r][c] == piece and board[r - 1][c + 1] == piece and board[r - 2][c + 2] == piece and board[r - 3][
                c + 3] == piece:
                return True


def evaluate_window(window, piece):
    score = 0
    opp_piece = PLAYER_PIECE
    if piece == PLAYER_PIECE:
        opp_piece = AI_PIECE

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

    ## Score center column
    center_array = [int(i) for i in list(board[:, COLUMN_COUNT // 2])]
    center_count = center_array.count(piece)
    score += center_count * 3

    ## Score Horizontal
    for r in range(ROW_COUNT):
        row_array = [int(i) for i in list(board[r, :])]
        for c in range(COLUMN_COUNT - 3):
            window = row_array[c:c + WINDOW_LENGTH]
            score += evaluate_window(window, piece)

    ## Score Vertical
    for c in range(COLUMN_COUNT):
        col_array = [int(i) for i in list(board[:, c])]
        for r in range(ROW_COUNT - 3):
            window = col_array[r:r + WINDOW_LENGTH]
            score += evaluate_window(window, piece)

    ## Score positive sloped diagonal
    for r in range(ROW_COUNT - 3):
        for c in range(COLUMN_COUNT - 3):
            window = [board[r + i][c + i] for i in range(WINDOW_LENGTH)]
            score += evaluate_window(window, piece)

    ## Score negative sloped diagonal
    for r in range(ROW_COUNT - 3):
        for c in range(COLUMN_COUNT - 3):
            window = [board[r + 3 - i][c + i] for i in range(WINDOW_LENGTH)]
            score += evaluate_window(window, piece)

    return score


def is_terminal_node(board):
    return winning_move(board, PLAYER_PIECE) or winning_move(board, AI_PIECE) or len(get_valid_locations(board)) == 0


def minimax(board, depth, alpha, beta, maximizingPlayer):
    valid_locations = get_valid_locations(board)
    is_terminal = is_terminal_node(board)
    if depth == 0 or is_terminal:
        if is_terminal:
            if winning_move(board, AI_PIECE):
                return (None, 100000000000000)
            elif winning_move(board, PLAYER_PIECE):
                return (None, -10000000000000)
            else:
                return (None, score_position(board, AI_PIECE) + random.uniform(-0.1, 0.1))  # Add randomness here
        else:
            return (None, score_position(board, AI_PIECE) + random.uniform(-0.1, 0.1))  # Add randomness here
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
    else:
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


def minimax_alpha_beta(board, depth, alpha, beta, maximizingPlayer):
    valid_locations = get_valid_locations(board)
    is_terminal = is_terminal_node(board)
    if depth == 0 or is_terminal:
        if is_terminal:
            if winning_move(board, AI_PIECE):
                return (None, 100000000000000)
            elif winning_move(board, PLAYER_PIECE):
                return (None, -10000000000000)
            else:
                return (None, score_position(board, AI_PIECE) + random.uniform(-0.1, 0.1))  # Add randomness here
        else:
            return (None, score_position(board, AI_PIECE) + random.uniform(-0.1, 0.1))  # Add randomness here
    if maximizingPlayer:
        value = -math.inf
        column = random
        column = random.choice(valid_locations)
        for col in valid_locations:
            row = get_next_open_row(board, col)
            b_copy = board.copy()
            drop_piece(b_copy, row, col, AI_PIECE)
            new_score = minimax_alpha_beta(b_copy, depth - 1, alpha, beta, False)[1]
            if new_score > value:
                value = new_score
                column = col
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        return column, value
    else:
        value = math.inf
        column = random.choice(valid_locations)
        for col in valid_locations:
            row = get_next_open_row(board, col)
            b_copy = board.copy()
            drop_piece(b_copy, row, col, PLAYER_PIECE)
            new_score = minimax_alpha_beta(b_copy, depth - 1, alpha, beta, True)[1]
            if new_score < value:
                value = new_score
                column = col
            beta = min(beta, value)
            if alpha >= beta:
                break
        return column, value


def get_valid_locations(board):
    valid_locations = []
    for col in range(COLUMN_COUNT):
        if is_valid_location(board, col):
            valid_locations.append(col)
    return valid_locations


def random_move_with_minimax(board, depth_threshold):
    if depth_threshold >= 0:
        # Use the minimax algorithm with alpha-beta pruning
        col, minimax_score = minimax_alpha_beta(board, 5, -math.inf, math.inf, True)
        if col is not None:
            return col
    # Fallback to random move
    valid_locations = get_valid_locations(board)
    return random.choice(valid_locations) if len(valid_locations) > 0 else None


import random
import math
import time

# Constants
PLAYER = 0
AI = 1


# Add your function definitions for:
# create_board, is_terminal_node, minimax, random_move_with_minimax,
# is_valid_location, get_next_open_row, drop_piece, winning_move, etc.

def play_game():
    board = create_board()
    initial_turn = random.randint(PLAYER, AI)
    turn = initial_turn

    # Initialize time counters for each AI
    ai1_time = 0
    ai2_time = 0

    while not is_terminal_node(board):
        start_time = time.time()  # Start timing

        if turn == AI:
            # AI 1 (Minimax) makes a move
            col, minimax_score = minimax(board, 5, -math.inf, math.inf, True)
            ai1_time += time.time() - start_time  # Update AI 1 time
        else:
            # AI 2 (Random with Minimax) makes a move
            col = random_move_with_minimax(board, depth_threshold=2)
            ai2_time += time.time() - start_time  # Update AI 2 time

        if col is not None and is_valid_location(board, col):
            row = get_next_open_row(board, col)
            drop_piece(board, row, col, AI_PIECE if turn == AI else PLAYER_PIECE)

        turn = 1 - turn  # Switch player

    # Return the result, the time taken by each AI, and who started the game
    result = None
    if winning_move(board, AI_PIECE):
        result = AI
    elif winning_move(board, PLAYER_PIECE):
        result = PLAYER

    return result, ai1_time, ai2_time, initial_turn


# Initialize counters and total times
ai1_wins = 0
ai2_wins = 0
ai1_total_time = 0
ai2_total_time = 0
ai1_starts_wins = 0
ai2_starts_wins = 0

for _ in range(100):
    result, ai1_time, ai2_time, initial_turn = play_game()

    if result == AI:
        ai1_wins += 1
        if initial_turn == AI:
            ai1_starts_wins += 1
    elif result == PLAYER:
        ai2_wins += 1
        if initial_turn == PLAYER:
            ai2_starts_wins += 1

    ai1_total_time += ai1_time
    ai2_total_time += ai2_time


print("AI 1 (Minimax) wins:", ai1_wins,"Wins when starting:", ai1_starts_wins, "Minimax total time:", round(ai1_total_time,2), "seconds")
print("AI 2 (Minimax A-B) wins:", ai2_wins,"Wins when starting:", ai2_starts_wins,"Minimax A-B total time:", round(ai2_total_time,2), "seconds")
print("Total time taken:", round(ai1_total_time + ai2_total_time,2), "seconds")


