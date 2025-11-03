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


def get_masks_bit_count(masks: list[int]) -> dict[int, int]:
    return {m: m.bit_count() for m in masks}


def get_masks_compatibility(masks: list[int]) -> dict[int, list[int]]:
    return {m2: [m1 for m1 in masks if not (m1 & m2)] for m2 in masks}
