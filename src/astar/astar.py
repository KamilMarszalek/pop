from src.util.types import Board
from heapq import heappop
from src.util.util import generate_non_adjacent_masks
from src.astar.state import State
from src.astar.heuristic import Heuristic
from src.astar.success_generator import SuccessorGenerator


class AStar:
    def __init__(
        self,
        board: Board,
        num_of_cards: int,
        heuristics: Heuristic,
        rows=4,
    ) -> None:
        self.masks = generate_non_adjacent_masks(rows)
        self.board = board
        self.num_of_cards = num_of_cards
        self.heuristics = heuristics
        self.current_state = self._get_initial_state()
        self.queue = [self.current_state]
        self.visited = {}
        self.best_profit = float("-inf")
        self.successor_generator = SuccessorGenerator(
            num_of_cards,
            board,
            heuristics,
            self.visited,
            self.queue,
        )

    def _get_initial_state(self) -> "State":
        return State(0, self.heuristics(0, 0, self.num_of_cards), 0, 0, 0)

    def run(self) -> int:
        while self.queue:
            self.current_state = heappop(self.queue)
            if (
                self.current_state.col_index == len(self.board)
                or self.current_state.cards_used == self.num_of_cards
            ):
                self.best_profit = max(self.best_profit, self.current_state.g_reward)
                continue
            self.visited[self.current_state.key()] = self.current_state.g_reward
            self.successor_generator.generate_successors(
                self.current_state, self.best_profit
            )
        return self.best_profit
