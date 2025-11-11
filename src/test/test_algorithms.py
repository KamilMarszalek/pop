from dataclasses import dataclass

from src.dp.bottom_up import mwis_bottom_up
from src.dp.top_down import mwis_top_down
from src.util.types import Board
from src.astar.astar import AStar


def run_astar(board: Board, max_cards: int) -> int:
    astar = AStar(board, max_cards)
    return astar.run()


@dataclass
class TestCase:
    board: list[list[int]]
    max_cards: int
    result: int
    __test__: bool = False


def test_algorithms():
    algorithms = [mwis_bottom_up, mwis_top_down, run_astar]
    test_cases = [
        # No values
        TestCase(board=[[]], max_cards=0, result=0),
        TestCase(board=[[]], max_cards=1, result=0),
        # Single value
        TestCase(board=[[1]], max_cards=1, result=1),
        TestCase(board=[[-1]], max_cards=1, result=0),
        # 1D array
        TestCase(board=[[1, 1, 1, 1]], max_cards=0, result=0),
        TestCase(board=[[1, 1, 1, 1]], max_cards=1, result=1),
        TestCase(board=[[1, 2, 3, 4]], max_cards=1, result=4),
        TestCase(board=[[1, 1, 1, 1]], max_cards=4, result=2),
        TestCase(board=[[2, 1, 2, 1]], max_cards=4, result=4),
        TestCase(board=[[1, -1, 1, -1]], max_cards=4, result=2),
        TestCase(board=[[-1, -1, -1, -1]], max_cards=4, result=0),
        TestCase(board=[[1, 2, 3, 4]], max_cards=4, result=6),
        # 2D array
        TestCase(board=[[1, 1, 1, 1], [1, 1, 1, 1]], max_cards=0, result=0),
        TestCase(board=[[1, 1, 1, 1], [1, 1, 1, 1]], max_cards=1, result=1),
        TestCase(board=[[1, 1, 1, 4], [1, 1, 1, 1]], max_cards=1, result=4),
        TestCase(board=[[1, 1, 1, 1], [1, 1, 1, 1]], max_cards=8, result=4),
        TestCase(board=[[2, 1, 2, 1], [1, 2, 1, 2]], max_cards=8, result=8),
        TestCase(board=[[-1, -1, -1, -1], [-1, -1, -1, -1]], max_cards=8, result=0),
        TestCase(board=[[1, -1, -1, -1], [1, -1, -1, -1]], max_cards=8, result=1),
        TestCase(board=[[1, -1, 1, -1], [1, -1, 1, -1]], max_cards=8, result=2),
        TestCase(board=[[1, -1, -1, 1], [1, -1, 1, 1]], max_cards=8, result=3),
        TestCase(board=[[1, -1, 1, -1], [-1, 1, -1, 1]], max_cards=8, result=4),
        TestCase(board=[[5, 3, 10, -1], [0, -3, 9, -2]], max_cards=8, result=15),
    ]
    for algo in algorithms:
        for tc in test_cases:
            result, _ = algo(tc.board, tc.max_cards)
            assert result == tc.result, (
                f"Alghortim {algo.__name__} failed for board={tc.board}",
                f"max_cards={tc.max_cards}; expected={tc.result} got={result}",
            )
