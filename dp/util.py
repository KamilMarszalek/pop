def generate_non_adjacent_masks(size: int) -> list[int]:
    valid_masks: list[int] = []
    for mask in range(1 << size):
        if mask & (mask << 1):
            continue
        valid_masks.append(mask)
    return valid_masks


def calculate_row_sum(row: list[int], mask: int) -> int:
    sum: int = 0
    for i, number in enumerate(row):
        if mask >> (len(row) - i - 1) & 1:
            sum += number
    return sum
