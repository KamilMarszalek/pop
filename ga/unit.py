from type_definitions import Genes
from random import choice, uniform
from itertools import combinations
from util.util import generate_non_adjacent_masks


class Unit:
    def __init__(
        self,
        num_of_columns: int,
        num_of_cards: int,
        genes: Genes = None,
    ) -> None:
        self.choices = generate_non_adjacent_masks(4)
        self.genes = (
            [choice(self.choices) for _ in range(num_of_columns)]
            if not genes
            else genes
        )
        self.num_of_cards = num_of_cards
        self.repair()

    def repair(self) -> None:
        for i in range(1, len(self.genes)):
            self.genes[i] &= ~self.genes[i - 1]
        num_of_cards_used = sum(x.bit_count() for x in self.genes)
        while num_of_cards_used > self.num_of_cards:
            column = choice([i for i, g in enumerate(self.genes) if g != 0])
            ones = [bit for bit in range(4) if (self.genes[column] >> bit) & 1]
            bit_to_remove = choice(ones)
            self.genes[column] &= ~(1 << bit_to_remove)
            num_of_cards_used -= 1

    def __str__(self) -> str:
        return str(self.genes)

    def __repr__(self) -> str:
        return str(self.genes)

    def cross(self, other: "Unit") -> tuple["Unit", "Unit"]:
        num_genes = len(self.genes)
        if num_genes <= 2:
            k = 1
        else:
            k = 2

        points = choice(list(combinations(range(1, num_genes), k)))

        def build_child(a, b, points):
            if len(points) == 1:
                p = points[0]
                return Unit(num_genes, self.num_of_cards, a[:p] + b[p:])
            else:
                p1, p2 = points
                return Unit(
                    num_genes,
                    self.num_of_cards,
                    a[:p1] + b[p1:p2] + a[p2:],
                )

        child1 = build_child(self.genes, other.genes, points)
        child2 = build_child(other.genes, self.genes, points)

        return child1, child2

    def mutate(self, probability_of_mutation: float) -> None:
        for i in range(len(self.genes)):
            if uniform(0, 1) < probability_of_mutation:
                self.genes[i] = choice(self.choices)
        self.repair()
