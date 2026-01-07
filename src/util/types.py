from typing import Any, Protocol

type Board = list[list[int]]
type MWISResult = tuple[int, list[int]]


class MWISSolver(Protocol):
    def __call__(
        self, board: Board, max_cards: int, *args: Any, **kwargs: Any
    ) -> tuple[MWISResult, float]: ...
