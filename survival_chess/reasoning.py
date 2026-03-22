"""
LLM-based strategy summarizer using Claude claude-sonnet-4-6.

Runs silent games with a loaded model, collects behavioral logs,
then asks Claude to describe the emergent strategy per piece type.
"""

import chess
import os
from typing import Dict, List, Optional

from evolution import PIECE_TYPE_NAMES, ALL_PIECE_TYPES

PIECE_SYMBOLS = {
    chess.PAWN: "♙",
    chess.KNIGHT: "♘",
    chess.BISHOP: "♗",
    chess.ROOK: "♖",
    chess.QUEEN: "♕",
    chess.KING: "♔",
}


def _summarize_game(result: Dict, game_idx: int) -> str:
    """Convert a game result into a compact text summary."""
    lines = [f"## Game {game_idx + 1}"]
    lines.append(f"Result: {result['board_result']}, Length: {result['game_length']} plies")

    if result["white_won"]:
        lines.append("Winner: White")
    elif result["black_won"]:
        lines.append("Winner: Black")
    else:
        lines.append("Winner: None (draw/timeout)")

    # Per-piece survival
    lines.append("\nPiece survival:")
    for pr in result["piece_results"]:
        color_str = "White" if pr["color"] == chess.WHITE else "Black"
        pt_name = PIECE_TYPE_NAMES.get(pr["original_piece_type"], "?")
        slot = pr["slot_id"]
        survived = pr["turns_survived"]
        cap = "captured" if pr["captured"] else "survived"
        lines.append(
            f"  {color_str} {pt_name} (slot {slot}): "
            f"{survived}/{result['game_length']} plies, {cap}, "
            f"fitness={pr['fitness']:.3f}"
        )

    # Turn log snippet (votes + moves per piece type)
    if result.get("turn_logs"):
        lines.append("\nVoting/move log (first 30 turns):")
        for log in result["turn_logs"][:30]:
            color_str = log["color"].capitalize()
            pt_name = PIECE_TYPE_NAMES.get(log["selected_piece_type"], "?")
            cap_str = " [CAPTURE]" if log["is_capture"] else ""
            lines.append(
                f"  Ply {log['ply']} ({color_str}): "
                f"Voted → slot {log['selected_slot']} ({pt_name}@{log['selected_square']}), "
                f"Move: {log['move_san']}{cap_str}"
            )

    return "\n".join(lines)


def build_statistical_summary(game_results: List[Dict]) -> str:
    """
    Produce a pure-statistics markdown summary — no API key required.
    Aggregates survival, capture, fitness, and voting data per piece type.
    """
    import numpy as np

    n_games = len(game_results)

    # Aggregate per (color, original_piece_type)
    stats: Dict = {}  # key -> {survivals, fitnesses, capture_values, captured_count, total}
    vote_counts: Dict = {}  # (color, original_piece_type) -> times voted to move

    for result in game_results:
        for pr in result["piece_results"]:
            key = (pr["color"], pr["original_piece_type"])
            if key not in stats:
                stats[key] = {
                    "survivals": [], "fitnesses": [], "capture_values": [], "captured": 0, "total": 0
                }
            stats[key]["survivals"].append(pr["turns_survived"])
            stats[key]["fitnesses"].append(pr["fitness"])
            stats[key]["capture_values"].append(pr.get("capture_value", 0))
            stats[key]["captured"] += int(pr["captured"])
            stats[key]["total"] += 1

        # Vote / selection frequency from turn logs
        for log in result.get("turn_logs", []):
            color = chess.WHITE if log["color"] == "white" else chess.BLACK
            pt = log["selected_piece_type"]
            vkey = (color, pt)
            vote_counts[vkey] = vote_counts.get(vkey, 0) + 1

    outcomes = [r["board_result"] for r in game_results]
    white_wins = sum(1 for r in game_results if r["white_won"])
    black_wins = sum(1 for r in game_results if r["black_won"])
    draws = n_games - white_wins - black_wins
    avg_length = float(np.mean([r["game_length"] for r in game_results]))

    lines = [
        f"## Statistical Summary ({n_games} games)",
        "",
        f"**Outcomes** — White wins: {white_wins}, Black wins: {black_wins}, Draws/timeouts: {draws}",
        f"**Average game length**: {avg_length:.1f} plies",
        "",
    ]

    for color in [chess.WHITE, chess.BLACK]:
        color_str = "White" if color == chess.WHITE else "Black"
        lines.append(f"### {color_str} pieces")
        lines.append("")
        lines.append("| Piece | Avg survival (plies) | Capture rate | Avg fitness | Avg material captured | Times selected to move |")
        lines.append("|---|---|---|---|---|---|")

        for pt in ALL_PIECE_TYPES:
            key = (color, pt)
            s = stats.get(key)
            if s is None or s["total"] == 0:
                continue
            avg_surv = float(np.mean(s["survivals"]))
            cap_rate = 100.0 * s["captured"] / s["total"]
            avg_fit  = float(np.mean(s["fitnesses"]))
            avg_capv = float(np.mean(s["capture_values"]))
            times_selected = vote_counts.get(key, 0)
            pt_name = PIECE_TYPE_NAMES[pt].capitalize()
            sym = PIECE_SYMBOLS[pt]
            lines.append(
                f"| {sym} {pt_name} | {avg_surv:.1f} | {cap_rate:.0f}% | {avg_fit:.3f} | {avg_capv:.2f} pts | {times_selected} |"
            )
        lines.append("")

    return "\n".join(lines)


def compute_analysis_data(game_results: List[Dict]) -> Dict:
    """
    Compute all structured analysis data from game results.
    Returns DataFrames and dicts ready for Streamlit display.

    Keys in returned dict:
      "n_games"        : int
      "outcomes"       : dict of outcome counts + percentages
      "per_piece"      : {"White": DataFrame, "Black": DataFrame}
      "voting_df"      : DataFrame (voter × target, values = % of votes)
      "phase_dynamics" : dict of capture phase breakdowns
    """
    import numpy as np
    import pandas as pd

    n_games = len(game_results)

    # ── Outcome stats ─────────────────────────────────────────────────────────
    white_wins = sum(1 for r in game_results if r["white_won"])
    black_wins = sum(1 for r in game_results if r["black_won"])
    draws_timeouts = n_games - white_wins - black_wins
    avg_length = float(np.mean([r["game_length"] for r in game_results]))
    outcomes = {
        "n_games": n_games,
        "white_wins": white_wins,
        "black_wins": black_wins,
        "draws_timeouts": draws_timeouts,
        "white_win_pct": round(100.0 * white_wins / max(n_games, 1), 1),
        "black_win_pct": round(100.0 * black_wins / max(n_games, 1), 1),
        "draws_pct": round(100.0 * draws_timeouts / max(n_games, 1), 1),
        "avg_length": round(avg_length, 1),
    }

    # ── Per-piece stats ───────────────────────────────────────────────────────
    total_turns = {chess.WHITE: 0, chess.BLACK: 0}
    move_counts: Dict = {}   # (color, pt) -> int
    piece_agg: Dict = {}     # (color, pt) -> {survivals, fitnesses, capture_values, captured, total}

    for result in game_results:
        for pr in result["piece_results"]:
            key = (pr["color"], pr["original_piece_type"])
            if key not in piece_agg:
                piece_agg[key] = {
                    "survivals": [], "fitnesses": [], "capture_values": [],
                    "captured": 0, "total": 0,
                }
            piece_agg[key]["survivals"].append(pr["turns_survived"])
            piece_agg[key]["fitnesses"].append(pr["fitness"])
            piece_agg[key]["capture_values"].append(pr.get("capture_value", 0))
            piece_agg[key]["captured"] += int(pr["captured"])
            piece_agg[key]["total"] += 1

        for log in result.get("turn_logs", []):
            color = chess.WHITE if log["color"] == "white" else chess.BLACK
            total_turns[color] += 1
            key = (color, log["selected_piece_type"])
            move_counts[key] = move_counts.get(key, 0) + 1

    per_piece: Dict = {}
    for color in [chess.WHITE, chess.BLACK]:
        color_str = "White" if color == chess.WHITE else "Black"
        total = max(total_turns[color], 1)
        rows = []
        for pt in ALL_PIECE_TYPES:
            key = (color, pt)
            d = piece_agg.get(key)
            if d is None or d["total"] == 0:
                continue
            rows.append({
                "Piece": f"{PIECE_SYMBOLS[pt]} {PIECE_TYPE_NAMES[pt].capitalize()}",
                "Move Freq %": round(100.0 * move_counts.get(key, 0) / total, 1),
                "Avg Survival": round(float(np.mean(d["survivals"])), 1),
                "Survival Rate %": round(100.0 * (d["total"] - d["captured"]) / d["total"], 1),
                "Avg Material Captured": round(float(np.mean(d["capture_values"])), 2),
                "Capture Rate %": round(100.0 * d["captured"] / d["total"], 1),
                "Avg Fitness": round(float(np.mean(d["fitnesses"])), 3),
            })
        per_piece[color_str] = pd.DataFrame(rows)

    # ── Voting matrix ─────────────────────────────────────────────────────────
    # Tally voter_type → target_type → count (combined across both colors)
    vote_tally = {pt: {pt2: 0 for pt2 in ALL_PIECE_TYPES} for pt in ALL_PIECE_TYPES}
    vote_totals = {pt: 0 for pt in ALL_PIECE_TYPES}

    for result in game_results:
        # Build (color, slot_id) → original_piece_type mapping
        slot_map: Dict = {}
        for pr in result["piece_results"]:
            slot_map[(pr["color"], pr["slot_id"])] = pr["original_piece_type"]

        for log in result.get("turn_logs", []):
            color = chess.WHITE if log["color"] == "white" else chess.BLACK
            for voter_pt, voted_slot in log.get("votes", {}).items():
                target_pt = slot_map.get((color, voted_slot))
                if target_pt is not None:
                    vote_tally[voter_pt][target_pt] += 1
                    vote_totals[voter_pt] += 1

    type_names = [PIECE_TYPE_NAMES[pt].capitalize() for pt in ALL_PIECE_TYPES]
    matrix = []
    for voter_pt in ALL_PIECE_TYPES:
        total = max(vote_totals[voter_pt], 1)
        matrix.append([
            round(100.0 * vote_tally[voter_pt][target_pt] / total, 1)
            for target_pt in ALL_PIECE_TYPES
        ])
    voting_df = pd.DataFrame(matrix, index=type_names, columns=type_names)
    voting_df.index.name = "Voter \\ Target"

    # ── Game phase dynamics ───────────────────────────────────────────────────
    early = mid = late = 0
    for result in game_results:
        L = max(result["game_length"], 1)
        for log in result.get("turn_logs", []):
            if log["is_capture"]:
                frac = log["ply"] / L
                if frac <= 1 / 3:
                    early += 1
                elif frac <= 2 / 3:
                    mid += 1
                else:
                    late += 1
    total_cap = early + mid + late
    phase_dynamics = {
        "total": total_cap,
        "avg_per_game": round(total_cap / max(n_games, 1), 1),
        "early_pct": round(100.0 * early / max(total_cap, 1), 1),
        "mid_pct": round(100.0 * mid / max(total_cap, 1), 1),
        "late_pct": round(100.0 * late / max(total_cap, 1), 1),
    }

    return {
        "n_games": n_games,
        "outcomes": outcomes,
        "per_piece": per_piece,
        "voting_df": voting_df,
        "phase_dynamics": phase_dynamics,
    }


def collect_game_logs(
    white_populations: Dict,
    black_populations: Dict,
    config,
    n_games: int = 20,
) -> List[Dict]:
    """Run n_games silent games with detailed logging and return results."""
    from game_engine import play_game

    results = []
    for _ in range(n_games):
        result = play_game(white_populations, black_populations, config, log_detail=True)
        results.append(result)
    return results


def build_prompt(game_results: List[Dict]) -> str:
    """Construct the prompt sent to Claude."""
    game_summaries = "\n\n".join(
        _summarize_game(r, i) for i, r in enumerate(game_results)
    )

    piece_names = ", ".join(PIECE_TYPE_NAMES[pt] for pt in ALL_PIECE_TYPES)

    prompt = f"""You are analyzing a "survival chess" AI system where chess pieces are autonomous neural network agents trained via neuroevolution. Each piece independently tries to survive as long as possible (stay uncaptured), with a secondary incentive to contribute to a team win.

Every turn, pieces VOTE for which friendly piece should move, then the selected piece independently chooses its move. There are NO human players — all behavior is emergent from the neuroevolution process.

Below are logs from {len(game_results)} games played by the current evolved model:

{game_summaries}

---

Based on these game logs, please write a strategic analysis describing the **emergent behavior** of each piece type. For each of the 6 piece types ({piece_names}), analyze:

1. **Survival strategy**: Does this piece type tend to survive long or get captured early? Any patterns?
2. **Voting behavior**: What patterns do you observe in which pieces they tend to vote for (by slot position, piece type, or game phase)?
3. **Movement style**: Do they advance aggressively, retreat defensively, cluster together, avoid conflict, or show other patterns?
4. **Surprising behaviors**: Anything unexpected or emergent that wouldn't arise from classic chess strategy?
5. **Fitness outcomes**: How do their fitness scores reflect their survival strategy?

Write one section per piece type using markdown headers. Be specific and cite evidence from the game logs. Note that these are purely emergent behaviors from neuroevolution — they may not resemble optimal chess strategy at all.
"""
    return prompt


def generate_strategy_summary(
    game_results: List[Dict],
    api_key: Optional[str] = None,
) -> str:
    """
    Send game logs to Claude and return the formatted markdown strategy summary.
    `api_key` overrides the ANTHROPIC_API_KEY environment variable.
    """
    try:
        import anthropic
    except ImportError:
        return (
            "**Error**: The `anthropic` package is not installed. "
            "Run `pip install anthropic` to enable this feature."
        )

    key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        return (
            "**Error**: No Anthropic API key provided. "
            "Set the `ANTHROPIC_API_KEY` environment variable or enter it above."
        )

    client = anthropic.Anthropic(api_key=key)
    prompt = build_prompt(game_results)

    try:
        message = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text
    except Exception as e:
        return f"**Error calling Claude API**: {e}"
