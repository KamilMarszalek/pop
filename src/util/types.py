from typing import Any, Protocol

type Board = list[list[int]]
type Log = tuple[list[int], list[float]]

type MWISBase = tuple[int, list[int]]
type MWISResult = MWISBase | tuple[*MWISBase, Log]


class MWISSolver(Protocol):
    def __call__(
        self, board: Board, max_cards: int, *args: Any, **kwargs: Any
    ) -> tuple[MWISResult, float]: ...
