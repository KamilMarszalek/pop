class State:
    def __init__(
        self,
        g_reward: int,
        h_reward: int,
        col_index: int,
        previous_mask: int,
        cards_used: int,
    ) -> None:
        self.col_index = col_index
        self.previous_mask = previous_mask
        self.cards_used = cards_used
        self.g_reward = g_reward
        self.h_reward = h_reward

    def f_reward(self) -> int:
        return self.g_reward + self.h_reward

    def __lt__(self, other: "State") -> bool:
        return self.f_reward() > other.f_reward()
