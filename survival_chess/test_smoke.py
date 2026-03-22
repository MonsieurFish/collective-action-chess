"""
Smoke test: separate white/black populations, HoF sampling, variance-based
stopping, and opponent-avg fitness delta. Play one game, evolve, verify clean.
"""

import sys
import traceback
import numpy as np

print("=" * 60)
print("Survival Chess — Smoke Test (v2: separate white/black pops)")
print("=" * 60)

try:
    print("\n[1/7] Importing modules…", end=" ")
    import chess
    from config import Config
    from encoding import compute_features, FEATURE_SIZE
    from neural_net import make_network, forward, softmax, mutate
    from evolution import (
        init_populations,
        aggregate_fitness_for_side,
        evolve_populations,
        population_weight_variance,
        snapshot_elites,
        sample_opponent,
        piece_fitness,
    )
    from game_engine import play_game
    print("OK")

    print("[2/7] Config + feature size check…", end=" ")
    cfg = Config(games_per_generation=2, max_turns=200)
    assert cfg.input_size == FEATURE_SIZE, (
        f"input_size={cfg.input_size} != FEATURE_SIZE={FEATURE_SIZE}"
    )
    print(f"OK  (FEATURE_SIZE={FEATURE_SIZE})")

    print("[3/7] Init separate white & black populations…", end=" ")
    np.random.seed(42)
    white_pops = init_populations(cfg)
    black_pops = init_populations(cfg)
    # 1 shared voting + 6 per-type moving = 7 populations per side
    assert len(white_pops) == 7
    assert len(black_pops) == 7
    assert "voting" in white_pops
    # Confirm they are independent objects
    assert white_pops["voting"][0] is not black_pops["voting"][0]
    print("OK  (7 + 7 populations, independent)")

    print("[4/7] Play one game with separate populations…", end=" ")
    result = play_game(white_pops, black_pops, cfg, log_detail=True)
    assert len(result["piece_results"]) == 32
    assert all(np.isfinite(pr["fitness"]) for pr in result["piece_results"])
    print(
        f"OK  result={result['board_result']}, "
        f"length={result['game_length']}, "
        f"fitness=[{min(pr['fitness'] for pr in result['piece_results']):.3f}, "
        f"{max(pr['fitness'] for pr in result['piece_results']):.3f}]"
    )

    print("[5/7] Verify simplified fitness (alpha/beta/kappa only)…", end=" ")
    white_fitnesses = [pr["fitness"] for pr in result["piece_results"] if pr["color"] == chess.WHITE]
    black_fitnesses = [pr["fitness"] for pr in result["piece_results"] if pr["color"] == chess.BLACK]
    assert all(f >= 0.0 for f in white_fitnesses + black_fitnesses), "Fitness should be non-negative"
    white_sum = sum(white_fitnesses)
    black_sum = sum(black_fitnesses)
    print(f"OK  white sum={white_sum:.3f}, black sum={black_sum:.3f}")

    print("[6/7] Aggregate fitness for white only and evolve…", end=" ")
    game_results = [result]
    white_mean_fit = aggregate_fitness_for_side(game_results, white_pops, chess.WHITE)
    black_mean_fit = aggregate_fitness_for_side(game_results, black_pops, chess.BLACK)
    assert all(key in white_mean_fit for key in white_pops)
    # Initialise EMA to zeros (as the app does on first training start)
    white_ema = {key: np.zeros(cfg.population_size) for key in white_pops}
    new_white_pops, white_elite_fit, new_white_ema = evolve_populations(
        white_pops, white_mean_fit, white_ema, cfg, generation=0
    )
    assert len(new_white_pops) == 7
    for key, pop in new_white_pops.items():
        assert len(pop) == cfg.population_size
    assert len(new_white_ema) == 7
    print("OK")

    print("[7/7] HoF snapshot, sampling, and variance check…", end=" ")
    # Snapshot elites — simplified: first k elements of evolved population
    snap = snapshot_elites(new_white_pops, k=cfg.elite_count)
    assert all(len(snap[key]) == cfg.elite_count for key in snap)

    # Sample opponent (HoF with one entry)
    hof = [snap]
    opp = sample_opponent(hof, black_pops)
    assert opp is snap  # should return the single entry

    # Sample from empty HoF → fallback
    opp_fallback = sample_opponent([], black_pops)
    assert opp_fallback is black_pops

    # Variance
    var_initial = population_weight_variance(white_pops)
    var_evolved  = population_weight_variance(new_white_pops)
    print(
        f"OK  initial var={var_initial:.5f}, "
        f"after 1 gen var={var_evolved:.5f} "
        f"(threshold={cfg.convergence_threshold})"
    )

    print("\n" + "=" * 60)
    print("✅  ALL SMOKE TESTS PASSED")
    print("=" * 60)
    sys.exit(0)

except Exception:
    print("\n\n❌  SMOKE TEST FAILED:")
    traceback.print_exc()
    sys.exit(1)
