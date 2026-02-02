# Search and Optimization Project — Cards on a 4×n Board

## Topic

In each cell of a rectangular board of size **4×n**, an integer **zᵢⱼ** is written.  
You are given **m** cards that may be placed on the board.  
A valid placement requires that **no two cards occupy cells adjacent vertically or horizontally**.

Your task is to find such a placement of the cards on the board that the **sum of the numbers in the occupied cells is maximized**.  
You **do not have to use all** of the cards.

---

## Core idea

Because the board height is fixed (4), each column can be represented as a **4-bit mask**.  
This enables efficient exact algorithms (DP, A*) and fast approximate methods (greedy + local repair, GA).

A placement is valid if:
- inside each column: no vertical adjacency (no consecutive 1s in the mask),
- between adjacent columns: no horizontal adjacency (`mask_current & mask_prev == 0`),
- total number of placed cards ≤ **m**.

---

## Implemented algorithms

### Dynamic Programming (exact)
Located in: `src/dp/`
- `top_down.py` — memoized recursion
- `bottom_up.py` — iterative DP

DP state follows the idea:
- current column index,
- previous mask,
- number of used cards,
- maximize the sum of chosen cells.

✅ Always optimal.

---

### A* Search (exact)
Located in: `src/astar/` (`astar.py`, `state.py`)

A* searches over states representing partial solutions and uses:
- `g(s)` — profit accumulated so far,
- `h(s)` — admissible heuristic estimating remaining maximum profit.

Heuristic is built using **block dynamic programming** and an upper-bound propagation outside the current block.  
✅ Always optimal, typically expands far fewer states than full DP.

---

### Greedy initialization + stochastic local repair (approximate)
Located in: `src/greedy/`
- greedy picks the best available cells while respecting conflicts,
- then repeatedly improves randomly chosen windows of size `(k × 4)` by re-optimizing locally using DP.

This is the “fast compromise” method:
✅ very fast in practice  
⚠️ not guaranteed optimal

---

### Genetic Algorithm (approximate; limited usefulness)
Located in: `src/ga/`
- 2-point crossover, mutation (replace a column mask),
- Lamarckian repair (fix invalid adjacency and card-limit violations).

Implemented for completeness, but in our benchmarks it is significantly slower and typically worse for large boards.

---

## Repository structure

```
.
├── src/                    # main implementation
│   ├── main.py              # experiment entrypoint (runs benchmark suites)
│   ├── astar/               # exact A* solver
│   ├── dp/                  # exact dynamic programming solvers
│   ├── greedy/              # greedy + stochastic local repair
│   ├── ga/                  # genetic algorithm
│   ├── experiment/          # configs, board generation, runners, aggregation
│   ├── util/                # helpers (timing, types, utilities)
│   └── test/                # pytest tests
├── plots/                   # plots used in the report / generated figures
├── results/                 # experiment outputs (csv logs, aggregated tables)
├── data/                    # (optional) input data / generated boards (if used)
├── pyproject.toml           # project config (uv / packaging)
└── uv.lock                  # pinned dependencies for uv
```

---

## Setup

This project uses `uv` (recommended) — dependencies are pinned in `uv.lock`.

### Install with uv (recommended)

```bash
uv sync
```

Run commands via:

```bash
uv run python -m src.main
```
---

## Running experiments

The main entrypoint for reproducing results is:

```bash
uv run python -m src.main
```

This runs the prepared experiment suites sequentially (board generation → algorithm runs → logs → aggregation).

Outputs are written to:
- `results/**/logs/*.csv` — raw per-run logs
- `results/**/tables/*.csv` — per-suite tables (also aggregated variants)

Plots shown in the report are stored in:
- `plots/*.png`

---

## Results summary (high level)

- **A\*** is usually the best choice among exact methods (optimal + fast).
- **DP (top-down / bottom-up)** is always optimal, but slower as the long dimension grows.
- **Greedy + local repair** gives a strong speed/quality tradeoff and often reaches near-optimal values on wide distributions.
- **Genetic algorithm** underperforms on large instances in this task setting.

---

## Reproducibility

Experiments were designed to be reproducible:
- each configuration is evaluated on multiple generated boards (e.g., 5 boards),
- boards are identified via `board_id`,
- generation uses a fixed seed (same boards across runs),
- stochastic methods also use seeded RNGs.

---

## Testing

Tests are in `src/test/`. Run with:

```bash
uv run pytest -q
```

---

## Notes

- Internally, the implementation relies on precomputing valid masks for the 4-row column, their popcounts, and per-column mask weights.  
  This greatly reduces overhead in DP / A* / repair steps.

---

## Authors
- Michał Szwejk
- Kamil Marszałek
