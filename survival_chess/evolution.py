"""
Population management, fitness aggregation, selection, and reproduction.

Population structure:
  7 populations per side:
    - 1 shared voting population (key: "voting")
    - 6 per-type moving populations (keys: (piece_type, "moving"))
  Each population is a list of `population_size` individuals (weight dicts).

Training now uses SEPARATE white and black populations that alternate training
each generation.  The idle side supplies a frozen opponent drawn from its
hall-of-fame (rolling window of elite snapshots from past generations).

Generation loop (for the training side):
  1. Play games: training side vs. opponent sampled from the other side's HoF.
  2. Aggregate fitness using only the training side's piece results.
  3. Rank training-side individuals by mean fitness.
  4. Elites (top 3) pass through unmutated.
  5. 9 mutated children from each elite → 27 children.
  6. New population = 3 elites + 27 children.
  7. Snapshot elites → append to training side's HoF.
  8. Switch training side.

Early stopping (#3):
  Compute within-population weight variance for each side after its training
  generation.  Stop when BOTH sides' variance falls below convergence_threshold.
"""

import chess
import numpy as np
import os
import pickle
from typing import Dict, List, Optional, Tuple

from config import Config
from neural_net import make_network, mutate, copy_weights

ALL_PIECE_TYPES = [
    chess.PAWN,
    chess.KNIGHT,
    chess.BISHOP,
    chess.ROOK,
    chess.QUEEN,
    chess.KING,
]
PIECE_TYPE_NAMES = {
    chess.PAWN: "pawn",
    chess.KNIGHT: "knight",
    chess.BISHOP: "bishop",
    chess.ROOK: "rook",
    chess.QUEEN: "queen",
    chess.KING: "king",
}
PHASES = ["voting", "moving"]


# ── Population initialisation ──────────────────────────────────────────────────

def init_populations(config: Config) -> Dict:
    """
    Initialise 7 populations per side, each containing `population_size` individuals.
    Keys:
      "voting"          — single shared voting network (all piece types use this)
      (pt, "moving")    — one moving network per piece type
    Call twice to get separate white and black populations.
    """
    populations: Dict = {}
    # Single shared voting network
    populations["voting"] = [
        make_network(config.input_size, config.hidden1, config.hidden2,
                     config.voting_output_size)
        for _ in range(config.population_size)
    ]
    # Per-type moving networks
    for pt in ALL_PIECE_TYPES:
        populations[(pt, "moving")] = [
            make_network(config.input_size, config.hidden1, config.hidden2,
                         config.moving_output_size)
            for _ in range(config.population_size)
        ]
    return populations


# ── Fitness function (standalone, used by app sliders) ────────────────────────

def piece_fitness(
    turns_survived: int,
    max_turns: int,
    won: bool,
    alpha: float = 1.0,
    gamma: float = 1.5,
    beta: float = 1.5,
    kappa: float = 1.0,
    capture_value: int = 0,
) -> float:
    """
    Fitness for a single piece instance: absolute survival + win bonus + capture bonus.

    abs_survival uses max_turns (fixed ceiling = 200) rather than game_length so
    that a piece surviving 100 plies in a 100-ply game gets (100/200)^γ, not 1.0.

    alpha:         weight on absolute survival fraction
    gamma:         curvature >1 rewards late survival more than early survival
    beta:          win bonus weight
    kappa:         capture bonus weight
    capture_value: sum of standard point values of pieces captured by this piece
    """
    abs_survival = (turns_survived / max(max_turns, 1)) ** gamma
    win_bonus = 1.0 if won else 0.0
    capture_bonus = capture_value / 39.0
    return alpha * abs_survival + beta * win_bonus + kappa * capture_bonus


# ── Fitness aggregation (per-side) ────────────────────────────────────────────

def aggregate_fitness_for_side(
    game_results: List[Dict],
    populations: Dict,
    color: chess.Color,
) -> Dict:
    """
    Aggregate mean fitness per individual, considering only pieces of `color`.
    Only the training side's results are used to evolve the training side's populations.

    Population keys:
      "voting"       — shared voting network; all pieces contribute to it
      (pt, "moving") — per-piece-type moving network

    Returns {key: np.array of mean fitness, shape (population_size,)}
    """
    records: Dict = {
        key: {i: [] for i in range(len(populations[key]))}
        for key in populations
    }

    for result in game_results:
        for pr in result["piece_results"]:
            if pr["color"] != color:
                continue
            pt = pr["original_piece_type"]
            # Shared voting network
            records["voting"][pr["voting_individual_idx"]].append(pr["fitness"])
            # Per-type moving network
            records[(pt, "moving")][pr["moving_individual_idx"]].append(pr["fitness"])

    mean_fitness: Dict = {}
    for key, rec in records.items():
        n = len(populations[key])
        means = np.array([
            float(np.mean(rec[i])) if rec[i] else 0.0
            for i in range(n)
        ])
        mean_fitness[key] = means

    return mean_fitness


# ── Within-population variance (convergence metric) ───────────────────────────

def population_weight_variance(populations: Dict) -> float:
    """
    Mean within-population weight variance across all 12 populations.

    For each population, flatten every individual's weights into a single vector,
    compute per-parameter variance across the population_size individuals, then
    average.  Result near 0 means all individuals are nearly identical → converged.
    """
    all_vars: List[float] = []
    for key, pop in populations.items():
        if len(pop) < 2:
            continue
        flat = np.array([
            np.concatenate([w.ravel() for w in ind.values()])
            for ind in pop
        ])  # shape (pop_size, total_params)
        all_vars.append(float(np.var(flat, axis=0).mean()))
    return float(np.mean(all_vars)) if all_vars else 0.0


# ── Hall-of-fame helpers ───────────────────────────────────────────────────────

def snapshot_elites(populations: Dict, k: int = 1) -> Dict:
    """
    Copy the first k individuals from each population into a compact dict
    suitable for hall-of-fame storage.

    After evolve_populations(), elites are always placed at indices 0..elite_count-1,
    so we don't need a separate fitness array for ranking here.
    k defaults to 1 to match the new elite_count=1 default.
    """
    return {
        key: [copy_weights(pop[i]) for i in range(min(k, len(pop)))]
        for key, pop in populations.items()
    }


def sample_opponent(
    hall_of_fame: List[Dict],
    fallback_populations: Dict,
) -> Dict:
    """
    Return a random snapshot from the hall of fame, or `fallback_populations`
    if the hall of fame is empty (first generation).
    """
    if not hall_of_fame:
        return fallback_populations
    return hall_of_fame[np.random.randint(len(hall_of_fame))]


# ── Evolution step ─────────────────────────────────────────────────────────────

def evolve_populations(
    populations: Dict,
    mean_fitness: Dict,
    ema_fitness: Dict,
    config: Config,
    generation: int,
) -> Tuple[Dict, Dict, Dict]:
    """
    Produce next generation for ONE side's populations.

    Fix C — EMA smoothing:
      effective_fitness[i] = ema_decay * ema_fitness[i] + (1-ema_decay) * raw_fitness[i]
    Selection is based on effective_fitness, not raw single-generation mean.
    Children inherit their parent elite's effective fitness discounted by ema_decay,
    giving them a warm start rather than starting from zero.

    Returns:
      new_populations  : evolved Dict (same structure)
      elite_fitness    : {(pt, phase): float} — EMA-smoothed best fitness per pop
      new_ema_fitness  : updated EMA dict to pass into the next generation
    """
    mutation_std = config.mutation_std_initial * (
        0.9 ** (generation // config.mutation_anneal_interval)
    )

    new_populations = {}
    elite_fitness = {}
    new_ema_fitness = {}

    for key in populations:
        pop = populations[key]
        raw_fit = mean_fitness[key]
        ema_fit = ema_fitness.get(key, np.zeros(len(pop)))
        n = len(pop)

        # Blend raw fitness with historical EMA
        effective = config.ema_decay * ema_fit + (1.0 - config.ema_decay) * raw_fit

        ranked = np.argsort(effective)[::-1]
        elite_idx = ranked[: config.elite_count]

        # Elites pass through unmutated
        elites = [copy_weights(pop[i]) for i in elite_idx]
        elite_fitness[key] = float(effective[ranked[0]])

        # All remaining slots are children of the elites (round-robin)
        n_children = n - config.elite_count
        children_per_elite = n_children // config.elite_count
        remainder = n_children % config.elite_count

        children = []
        for ei, (idx, elite) in enumerate(zip(elite_idx, elites)):
            nc = children_per_elite + (1 if ei < remainder else 0)
            for _ in range(nc):
                children.append(mutate(elite, config.mutation_rate, mutation_std))

        new_populations[key] = elites + children

        # Build new EMA array:
        #   elites:   keep their current effective fitness
        #   children: warm-start from parent's effective fitness, discounted one step
        new_ema = np.zeros(n)
        for slot, idx in enumerate(elite_idx):
            new_ema[slot] = effective[idx]
        child_slot = config.elite_count
        for ei, idx in enumerate(elite_idx):
            nc = children_per_elite + (1 if ei < remainder else 0)
            for _ in range(nc):
                new_ema[child_slot] = config.ema_decay * effective[idx]
                child_slot += 1
        new_ema_fitness[key] = new_ema

    return new_populations, elite_fitness, new_ema_fitness


# ── Model persistence ──────────────────────────────────────────────────────────

SAVE_DIR = os.path.join(os.path.dirname(__file__), "saved_models")


def save_model(name: str, white_populations: Dict, black_populations: Dict) -> str:
    """Save white + black populations to a single pickle file."""
    os.makedirs(SAVE_DIR, exist_ok=True)
    path = os.path.join(SAVE_DIR, f"{name}.pkl")
    with open(path, "wb") as f:
        pickle.dump({"white": white_populations, "black": black_populations}, f)
    return path


def load_model(name: str) -> Tuple[Dict, Dict]:
    """
    Load white and black populations from a saved file.
    Returns (white_populations, black_populations).
    Handles old single-population format for backward compatibility.
    """
    path = os.path.join(SAVE_DIR, f"{name}.pkl")
    with open(path, "rb") as f:
        data = pickle.load(f)
    if isinstance(data, dict) and "white" in data and "black" in data:
        return data["white"], data["black"]
    # Old format: single shared populations dict — use for both sides
    return data, data


def list_saved_models() -> List[str]:
    """Return list of saved model names (without extension)."""
    if not os.path.exists(SAVE_DIR):
        return []
    return sorted(f[:-4] for f in os.listdir(SAVE_DIR) if f.endswith(".pkl"))
