from random import randint

COLUMNS = 4
ROWS = 20
LOW = 0
HIGH = 10


def generate_valid_masks(num_of_rows: int) -> list[int]:
    valid_masks: list[int] = []

    for mask in range(1 << num_of_rows):
        if mask & (mask << 1):
            continue
        valid_masks.append(mask)
    return valid_masks


def calc_values_for_all_masks(
    masks: list[int], board: list[list[int]]
) -> list[list[int]]:
    values: list[list[int]] = [[0] * len(masks) for _ in range(len(board))]
    for i in range(COLUMNS):
        for j, mask in enumerate(masks):
            value = 0
            for r in range(len(board)):
                if (mask >> r) & 1:
                    value += board[r][i]
            values[i][j] = value
    return values


if __name__ == "__main__":
    num_of_cards = 7
    board = [[randint(LOW, HIGH) for _ in range(4)] for _ in range(ROWS)]
    print(board)
    valid_masks = generate_valid_masks(ROWS)
    values = calc_values_for_all_masks(valid_masks, board)

    dp: list[list[list[float]]] = [
        [[-float("inf")] * (num_of_cards + 1) for _ in valid_masks]
        for _ in range(COLUMNS)
    ]

    for i, mask in enumerate(valid_masks):
        count = bin(mask).count("1")
        if count <= num_of_cards:
            dp[0][i][count] = values[0][i]

    for j in range(1, COLUMNS):
        for i, mask in enumerate(valid_masks):
            count = bin(mask).count("1")
            for prev_i, prev_mask in enumerate(valid_masks):
                if mask & prev_mask:
                    continue
                for k_prev in range(num_of_cards + 1 - count):
                    if dp[j - 1][prev_i][k_prev] == -float("inf"):
                        continue
                    new_k = k_prev + count
                    dp[j][i][new_k] = max(
                        dp[j][i][new_k],
                        dp[j - 1][prev_i][k_prev] + values[j][i],
                    )

    best = max(
        dp[COLUMNS - 1][i][k]
        for i in range(len(valid_masks))
        for k in range(num_of_cards + 1)
    )
    print(best)
