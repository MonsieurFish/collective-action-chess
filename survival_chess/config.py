from dataclasses import dataclass


@dataclass
class Config:
    # Network architecture
    input_size: int = 72       # Feature vector size (computed in encoding.py)
    hidden1: int = 96
    hidden2: int = 48
    voting_output_size: int = 16   # One logit per piece slot
    moving_output_size: int = 64   # One logit per board square (a1=0 .. h8=63)

    # Game parameters
    max_turns: int = 200           # Max half-moves (plies) per game
    serendipity: float = 0.4       # Softmax temperature for move selection

    # Evolution parameters
    population_size: int = 30
    elite_count: int = 3           # Top N individuals passed unmutated
    games_per_generation: int = 50
    mutation_rate: float = 0.20    # Per-weight mutation probability
    mutation_std_initial: float = 0.08   # Fix A: lower start → lower variance floor
    mutation_anneal_interval: int = 10   # Multiply mutation_std by 0.9 every N generations
    early_stop_patience: int = 30  # Stop if no improvement for N generations
    ema_decay: float = 0.3         # EMA smoothing across generations (0=no memory, 1=never update)

    # Fitness function weights
    alpha: float = 1.0   # Absolute survival weight
    gamma: float = 1.5   # Survival curvature (>1 rewards late survival more)
    beta: float = 1.5    # Win bonus weight
    kappa: float = 1.0   # Capture bonus weight (point value of pieces captured / 39)

    # Convergence / hall of fame
    hall_of_fame_size: int = 20      # Number of historical elite snapshots to keep per side
    convergence_threshold: float = 0.005  # Stop when within-pop weight variance < this
