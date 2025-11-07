from src.util.types import Board
from heapq import heappush, heappop, nlargest
from src.util.util import generate_non_adjacent_masks
from src.astar.state import State

MASKS = generate_non_adjacent_masks(4)


def h_reward(board: Board, col_index: int, cards_left: int) -> int:
    if cards_left <= 0:
        return 0
    flat_board_remaining_cols_only = [
        cell for row in board[col_index:] for cell in row if cell > 0
    ]
    biggest_values = nlargest(cards_left, flat_board_remaining_cols_only)
    return sum(biggest_values[:cards_left])


def should_continue(
    current_state: "State",
    board: Board,
    num_of_cards: int,
    queue: list["State"],
) -> bool:
    return (
        current_state.col_index < len(board)
        and current_state.cards_used <= num_of_cards
        and queue
    )


def a_star(board: Board, num_of_cards: int) -> int:
    current_state = State(0, h_reward(board, 0, num_of_cards), 0, 0, 0)
    queue = [current_state]
    visited = {}
    best_profit = 0
    while should_continue(current_state, board, num_of_cards, queue):
        current_state = heappop(queue)
        visited[
            (
                current_state.col_index,
                current_state.previous_mask,
                current_state.cards_used,
            )
        ] = current_state.g_reward
        generate_children(current_state, board, num_of_cards, queue, visited)

        if current_state.g_reward > best_profit:
            best_profit = current_state.g_reward
    return max(best_profit, current_state.g_reward)


def generate_children(
    state: "State",
    board: Board,
    num_of_cards: int,
    queue: list["State"],
    visited: dict[tuple[int, int, int], int],
) -> None:
    if state.col_index >= len(board):
        return None
    col = board[state.col_index]
    for mask in MASKS:
        if mask & state.previous_mask:
            continue
        delta_profit = 0
        cards_used = 0
        for row in range(4):
            if (mask >> row) & 1:
                delta_profit += col[row]
                cards_used += 1
        if cards_used + state.cards_used > num_of_cards:
            continue
        new_state = State(
            state.g_reward + delta_profit,
            h_reward(
                board,
                state.col_index + 1,
                num_of_cards - state.cards_used - cards_used,
            ),
            state.col_index + 1,
            mask,
            state.cards_used + cards_used,
        )
        old = visited.get(
            (new_state.col_index, new_state.previous_mask, new_state.cards_used), -1
        )
        if old >= new_state.g_reward:
            continue

        heappush(queue, new_state)
