from src.ga.crossover import CrossoverFunc
from src.ga.mutation import MutationFunc
from src.ga.q import QFunc
from src.ga.reproduction import ReproducitionFunc
from src.ga.succession import SuccesionFunc
from src.ga.type_definitions import Board, Population
from src.ga.unit import Unit
from src.util.time_measure import measure_time
from src.util.types import MWISResult


class GeneticAlgorithm:
    def __init__(
        self,
        q: QFunc,
        mutation: MutationFunc,
        reproduction: ReproducitionFunc,
        crossover: CrossoverFunc,
        succession: SuccesionFunc,
        population_count: int,
        probability_of_mutation: float,
        probability_of_crossover: float,
        fes: int,
        num_of_cards: int,
        board: Board,
        num_of_best_survivors: int = 0,
        starting_population: Population | None = None,
    ):
        self._q = q
        self._mutation = mutation
        self._reproduction = reproduction
        self._crossover = crossover
        self._succession = succession
        self.population_count = population_count
        self.probability_of_mutation = probability_of_mutation
        self.probability_of_crossover = probability_of_crossover
        self.t_max = fes // population_count
        self.t = 0
        self.board = board
        self.num_of_cards = num_of_cards
        self.num_of_best_survivors = num_of_best_survivors
        self.population = (
            starting_population if starting_population else self._generate_starting_population()
        )
        self.evaluation = self._get_population_evaluation(self.population)
        self.best_unit, self.best_value = self._find_best_unit(self.population, self.evaluation)

    def _generate_starting_population(self) -> Population:
        return [Unit(len(self.board), self.num_of_cards) for _ in range(self.population_count)]

    def _get_population_evaluation(self, population: Population) -> dict[Unit, int]:
        evaluation: dict[Unit, int] = {}
        for u in population:
            evaluation[u] = self._q(u, self.board)
        return evaluation

    def _find_best_unit(
        self, population: Population, evaluations: dict[Unit, int]
    ) -> tuple[Unit, int]:
        best_unit = population[0]
        best_value = evaluations[best_unit]
        for unit in population:
            evaluation = evaluations[unit]
            if evaluation > best_value:
                best_unit = unit
                best_value = evaluation
        return best_unit, best_value

    def _stop(self) -> bool:
        return self.t >= self.t_max

    def reproduction(self) -> None:
        self.r_population = self._reproduction(
            self.population, self.evaluation, self.population_count
        )

    def crossover(self) -> None:
        self.c_population = self._crossover(
            self.r_population, self.probability_of_crossover, self.population_count
        )

    def mutation(self) -> None:
        self.m_population = self._mutation(self.c_population, self.probability_of_mutation)

    def succession(self) -> None:
        self.population, self.evaluation = self._succession(
            self.m_population,
            self.m_evaluation,
            self.population,
            self.evaluation,
            self.num_of_best_survivors,
        )

    @measure_time()
    def run(self) -> MWISResult:
        while not self._stop():
            self.reproduction()
            self.crossover()
            self.mutation()
            self.m_evaluation = self._get_population_evaluation(self.m_population)
            best_candidate, best_candidate_evaluation = self._find_best_unit(
                self.m_population, self.m_evaluation
            )
            if best_candidate_evaluation > self.best_value:
                self.best_unit = best_candidate
                self.best_value = best_candidate_evaluation
                # print(self.best_value)
            self.succession()
            self.t += 1
        return self.best_value, self.best_unit.genes
