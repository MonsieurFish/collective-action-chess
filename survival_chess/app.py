"""
Survival Chess — Streamlit UI
==============================
Three pages:
  1. TRAIN  — configure hyperparameters, run neuroevolution, watch live metrics.
  2. WATCH  — load a saved model, play a game, watch the board update live.
  3. REASONING — load a model, run 5 games, ask Claude for a strategy summary.

Training changes vs. original design:
  - White and black populations are independent; only ONE side trains per generation.
  - The idle side supplies a frozen opponent drawn from its hall-of-fame.
  - Early stopping is variance-based: stop when both sides' within-population
    weight variance drops below convergence_threshold (populations have collapsed
    onto a stable solution rather than drifting aimlessly).
  - Fitness: α·(t/T_max)^γ + β·w + κ·V/39 (survival, win bonus, capture bonus).
"""

import time
import chess
import chess.svg
import numpy as np
import streamlit as st
import streamlit.components.v1 as components

from config import Config
from evolution import (
    init_populations,
    aggregate_fitness_for_side,
    evolve_populations,
    population_weight_variance,
    snapshot_elites,
    sample_opponent,
    save_model,
    load_model,
    list_saved_models,
    PIECE_TYPE_NAMES,
    ALL_PIECE_TYPES,
)
from game_engine import play_game
from reasoning import (
    collect_game_logs,
    generate_strategy_summary,
    build_statistical_summary,
    compute_analysis_data,
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Survival Chess", page_icon="♟", layout="wide")

page = st.sidebar.radio("Navigate", ["🧬 Train", "👁 Watch", "🧠 Reasoning"], index=0)

# ── Shared session state defaults ─────────────────────────────────────────────
def _default(key, val):
    if key not in st.session_state:
        st.session_state[key] = val

_default("training", False)
_default("white_populations", None)
_default("black_populations", None)
_default("white_ema_fitness", {})   # EMA fitness parallel to white_populations
_default("black_ema_fitness", {})   # EMA fitness parallel to black_populations
_default("white_hof", [])           # list of elite snapshots for white
_default("black_hof", [])           # list of elite snapshots for black
_default("training_side", chess.WHITE)   # alternates each generation
_default("generation", 0)
_default("fitness_history", [])     # list of per-generation dicts
_default("white_variance", float("inf"))
_default("black_variance", float("inf"))

# ── Board renderer ─────────────────────────────────────────────────────────────
def render_board(board: chess.Board, size: int = 400, lastmove=None):
    svg = chess.svg.board(board, size=size, lastmove=lastmove)
    components.html(svg, height=size + 20, scrolling=False)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 1 — TRAIN
# ─────────────────────────────────────────────────────────────────────────────
if page == "🧬 Train":
    st.title("🧬 Train — Survival Chess Neuroevolution")

    # ── Sidebar: hyperparameters ──────────────────────────────────────────────
    with st.sidebar:
        st.header("Fitness Function")
        alpha = st.slider("alpha (absolute survival)", 0.0, 3.0, 1.0, 0.05)
        gamma = st.slider("gamma (survival curvature)", 0.5, 3.0, 1.5, 0.05)
        beta  = st.slider("beta (win bonus)", 0.0, 3.0, 1.5, 0.05)
        kappa = st.slider("kappa (capture bonus)", 0.0, 3.0, 1.0, 0.05)

        st.header("Evolution")
        games_per_gen    = st.slider("games_per_generation", 10, 200, 50, 10)
        elite_count      = st.slider("elite_count", 1, 10, 3, 1)
        mutation_rate    = st.slider("mutation_rate", 0.01, 0.50, 0.20, 0.01)
        mutation_std     = st.slider("mutation_std (initial)", 0.01, 0.50, 0.08, 0.01)
        anneal_interval  = st.slider("anneal_interval (gens per ×0.9 std decay)", 1, 100, 10, 1)

        st.header("Convergence / HoF")
        ema_decay   = st.slider(
            "ema_decay", 0.0, 0.95, 0.3, 0.05,
            help="Smoothing across generations. 0 = no memory, 0.9 = strong memory.",
        )
        hof_size    = st.slider("hall_of_fame_size", 2, 50, 20, 1)
        conv_thresh = st.slider(
            "convergence_threshold", 0.0001, 0.05, 0.005, 0.0001,
            format="%.4f",
            help="Stop when within-population weight variance < this for both sides.",
        )

    cfg = Config(
        alpha=alpha, gamma=gamma, beta=beta, kappa=kappa,
        games_per_generation=games_per_gen,
        elite_count=elite_count,
        mutation_rate=mutation_rate,
        mutation_std_initial=mutation_std,
        mutation_anneal_interval=anneal_interval,
        ema_decay=ema_decay,
        hall_of_fame_size=hof_size,
        convergence_threshold=conv_thresh,
    )

    # ── Control buttons ───────────────────────────────────────────────────────
    col_start, col_stop, col_save = st.columns([1, 1, 2])

    with col_start:
        if st.button("▶ Start Training", disabled=st.session_state.training):
            if st.session_state.white_populations is None:
                wp = init_populations(cfg)
                bp = init_populations(cfg)
                st.session_state.white_populations = wp
                st.session_state.black_populations = bp
                # Initialise EMA arrays to zero for every population
                st.session_state.white_ema_fitness = {
                    key: np.zeros(cfg.population_size) for key in wp
                }
                st.session_state.black_ema_fitness = {
                    key: np.zeros(cfg.population_size) for key in bp
                }
                # Seed HoF so gen-1 has an opponent to draw from
                st.session_state.white_hof = [snapshot_elites(wp, k=cfg.elite_count)]
                st.session_state.black_hof = [snapshot_elites(bp, k=cfg.elite_count)]
                st.session_state.generation = 0
                st.session_state.fitness_history = []
                st.session_state.white_variance = float("inf")
                st.session_state.black_variance = float("inf")
            st.session_state.training = True
            st.session_state.training_side = chess.WHITE
            st.rerun()

    with col_stop:
        if st.button("⏹ Stop Training", disabled=not st.session_state.training):
            st.session_state.training = False
            st.rerun()

    with col_save:
        save_name = st.text_input(
            "Model name", value="model_gen_" + str(st.session_state.generation)
        )
        if st.button("💾 Save Current Model"):
            if (st.session_state.white_populations is not None
                    and st.session_state.black_populations is not None):
                path = save_model(
                    save_name,
                    st.session_state.white_populations,
                    st.session_state.black_populations,
                )
                st.success(f"Saved to `{path}`")
            else:
                st.warning("No model to save — start training first.")

    st.divider()

    # ── Metrics display ───────────────────────────────────────────────────────
    gen_counter   = st.empty()
    side_indicator = st.empty()
    chart_var     = st.empty()
    chart_fitness = st.empty()
    variance_cols = st.columns(2)

    if st.session_state.fitness_history:
        import pandas as pd

        gen_counter.metric("Generation", st.session_state.generation)

        # Variance convergence chart
        df_var = pd.DataFrame(
            {
                "White variance": [h["white_variance"] for h in st.session_state.fitness_history],
                "Black variance": [h["black_variance"] for h in st.session_state.fitness_history],
            },
            index=[h["gen"] for h in st.session_state.fitness_history],
        )
        df_var.index.name = "Generation"
        chart_var.subheader("Population convergence (↓ = converging)")
        chart_var.line_chart(df_var)

        # Per-piece-type best fitness for each side
        history = st.session_state.fitness_history
        white_rows, black_rows = {}, {}
        for pt in ALL_PIECE_TYPES:
            name = PIECE_TYPE_NAMES[pt].capitalize()
            white_rows[name] = [
                h.get(f"white_{PIECE_TYPE_NAMES[pt]}_voting", None) for h in history
            ]
            black_rows[name] = [
                h.get(f"black_{PIECE_TYPE_NAMES[pt]}_voting", None) for h in history
            ]

        # ffill: each side only writes values on its training generations,
        # leaving NaN on the other side's turns.  Forward-fill propagates the
        # last known value so lines render continuously instead of disappearing.
        df_w = pd.DataFrame(white_rows, index=[h["gen"] for h in history]).ffill()
        df_w.index.name = "Generation"
        df_b = pd.DataFrame(black_rows, index=[h["gen"] for h in history]).ffill()
        df_b.index.name = "Generation"

        fit_tab_w, fit_tab_b = st.tabs(["White elite fitness", "Black elite fitness"])
        with fit_tab_w:
            st.line_chart(df_w)
        with fit_tab_b:
            st.line_chart(df_b)

        # Current variance readouts
        with variance_cols[0]:
            wv = st.session_state.white_variance
            st.metric(
                "White variance",
                f"{wv:.5f}" if wv != float('inf') else "—",
                delta=f"threshold {conv_thresh:.4f}",
                delta_color="off",
            )
        with variance_cols[1]:
            bv = st.session_state.black_variance
            st.metric(
                "Black variance",
                f"{bv:.5f}" if bv != float('inf') else "—",
                delta=f"threshold {conv_thresh:.4f}",
                delta_color="off",
            )
    else:
        gen_counter.info("Training not started yet. Press ▶ Start Training.")

    # ── Training loop — one generation per Streamlit rerun ───────────────────
    if st.session_state.training and st.session_state.white_populations is not None:
        training_side = st.session_state.training_side
        side_name = "White" if training_side == chess.WHITE else "Black"
        gen = st.session_state.generation

        side_indicator.info(
            f"Generation {gen + 1} — training **{side_name}** "
            f"(HoF sizes: ♔ {len(st.session_state.white_hof)}  "
            f"♚ {len(st.session_state.black_hof)})"
        )

        status = st.status(f"Running gen {gen + 1} ({side_name})…", expanded=False)

        with status:
            # Pick training populations and opponent
            if training_side == chess.WHITE:
                train_pops = st.session_state.white_populations
                opp_pops   = sample_opponent(
                    st.session_state.black_hof, st.session_state.black_populations
                )
                wp, bp = train_pops, opp_pops
            else:
                train_pops = st.session_state.black_populations
                opp_pops   = sample_opponent(
                    st.session_state.white_hof, st.session_state.white_populations
                )
                wp, bp = opp_pops, train_pops

            # Play games
            game_results = [
                play_game(wp, bp, cfg, log_detail=False)
                for _ in range(cfg.games_per_generation)
            ]

            # Aggregate fitness only for the training side
            mean_fit = aggregate_fitness_for_side(game_results, train_pops, training_side)

            # Evolve — pass EMA in, receive updated EMA out
            ema_key = "white_ema_fitness" if training_side == chess.WHITE else "black_ema_fitness"
            new_train_pops, elite_fit, new_ema = evolve_populations(
                train_pops, mean_fit, st.session_state[ema_key], cfg, gen
            )

            # Update populations, EMA, and HoF
            elite_snap = snapshot_elites(new_train_pops, k=cfg.elite_count)
            if training_side == chess.WHITE:
                st.session_state.white_populations = new_train_pops
                st.session_state.white_ema_fitness = new_ema
                hof = st.session_state.white_hof
                hof.append(elite_snap)
                if len(hof) > cfg.hall_of_fame_size:
                    hof.pop(0)
                st.session_state.white_variance = population_weight_variance(new_train_pops)
            else:
                st.session_state.black_populations = new_train_pops
                st.session_state.black_ema_fitness = new_ema
                hof = st.session_state.black_hof
                hof.append(elite_snap)
                if len(hof) > cfg.hall_of_fame_size:
                    hof.pop(0)
                st.session_state.black_variance = population_weight_variance(new_train_pops)

            # Advance generation counter and flip training side
            st.session_state.generation += 1
            st.session_state.training_side = (
                chess.BLACK if training_side == chess.WHITE else chess.WHITE
            )

            # Record history
            record: dict = {
                "gen": st.session_state.generation,
                "training_side": side_name.lower(),
                "white_variance": st.session_state.white_variance,
                "black_variance": st.session_state.black_variance,
            }
            for key, fit_val in elite_fit.items():
                pt, phase = key
                record[f"{side_name.lower()}_{PIECE_TYPE_NAMES[pt]}_{phase}"] = fit_val
            st.session_state.fitness_history.append(record)

            status.update(label=f"Gen {st.session_state.generation} ({side_name}) ✓", state="complete")

        # Convergence check — requires at least 2 gens so both variances are measured
        wv = st.session_state.white_variance
        bv = st.session_state.black_variance
        if (wv < cfg.convergence_threshold and bv < cfg.convergence_threshold
                and st.session_state.generation >= 4):
            st.session_state.training = False
            st.success(
                f"Converged after {st.session_state.generation} generations. "
                f"White var={wv:.5f}, Black var={bv:.5f} "
                f"(threshold {cfg.convergence_threshold:.4f})"
            )
        elif st.session_state.training:
            time.sleep(0.05)
            st.rerun()

    # ── Fitness function reference box (always visible on Train page) ─────────
    st.divider()
    with st.expander("Fitness function reference", expanded=True):
        st.latex(
            r"f \;=\; "
            r"\underbrace{\alpha \cdot \left(\frac{t}{T_{\max}}\right)^{\!\gamma}}_{\text{absolute survival}}"
            r"\;+\; "
            r"\underbrace{\beta \cdot w}_{\text{win bonus}}"
            r"\;+\; "
            r"\underbrace{\kappa \cdot \frac{V}{39}}_{\text{capture bonus}}"
        )
        st.markdown(
            "| Variable | Meaning |\n"
            "|---|---|\n"
            r"| $t$ | Plies (half-moves) this piece survived before being captured, or the game's final ply if still alive |"
            "\n"
            r"| $T_{\max}$ | Maximum game length — fixed at **200 plies**; normalises $t$ to a consistent $[0,1]$ scale regardless of how long the actual game lasted |"
            "\n"
            r"| $w$ | **1** if this piece's side won the game, **0** for a draw, timeout, or loss |"
            "\n"
            r"| $V$ | Sum of standard point values of enemy pieces this piece captured (pawn=1, knight/bishop=3, rook=5, queen=9); divided by 39 (total material) to normalise to $[0,1]$ |"
        )
        st.divider()
        st.markdown("**EMA smoothing** (applied before selection each generation):")
        st.latex(
            r"\tilde{f}_{\,g}(i) \;=\; \lambda \cdot \tilde{f}_{\,g-1}(\mathrm{parent}_i)"
            r"\;+\; (1-\lambda) \cdot \bar{f}_{\,g}^{\,\mathrm{raw}}(i)"
        )
        st.markdown(
            "| Variable | Meaning |\n"
            "|---|---|\n"
            r"| $\tilde{f}_g(i)$ | Smoothed fitness used for selection — the quantity individuals are actually ranked by |"
            "\n"
            r"| $\bar{f}_g^{\,\mathrm{raw}}(i)$ | Mean raw fitness across all games individual $i$ participated in this generation |"
            "\n"
            r"| $\lambda$ | EMA decay (ema_decay slider); children inherit $\lambda \cdot \tilde{f}$ of their parent as a warm start |"
        )


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 2 — WATCH
# ─────────────────────────────────────────────────────────────────────────────
elif page == "👁 Watch":
    st.title("👁 Watch — Live Game Replay")

    _default("watch_game_result", None)
    _default("watch_playing", False)

    # ── Load model ────────────────────────────────────────────────────────────
    saved = list_saved_models()
    use_training = (
        st.session_state.white_populations is not None
        and st.session_state.black_populations is not None
    )
    options = (["Current training model"] if use_training else []) + saved
    source = st.radio("Model source", options, disabled=not options)

    if source == "Current training model":
        watch_wp = st.session_state.white_populations
        watch_bp = st.session_state.black_populations
    elif source:
        watch_wp, watch_bp = load_model(source)
    else:
        watch_wp = watch_bp = None

    serendipity = st.slider("serendipity (move temperature)", 0.1, 2.0, 0.8, 0.05)
    watch_cfg = Config(serendipity=serendipity)

    # ── Controls ──────────────────────────────────────────────────────────────
    col_new, col_resume, col_step = st.columns(3)

    with col_new:
        if st.button("▶ New Game", disabled=watch_wp is None):
            result = play_game(watch_wp, watch_bp, watch_cfg, log_detail=True)
            st.session_state.watch_game_result = result
            st.session_state.watch_playing = False
            # Reset the slider's own session-state key so it starts at 0
            st.session_state.ply_slider = 0
            st.rerun()

    with col_resume:
        is_playing = st.session_state.watch_playing
        resume_label = "⏸ Pause" if is_playing else "▶ Resume"
        if st.button(resume_label, disabled=st.session_state.watch_game_result is None):
            st.session_state.watch_playing = not is_playing
            st.rerun()

    with col_step:
        if st.button("⏩ Step", disabled=st.session_state.watch_game_result is None):
            # Write directly to the slider's key — the slider reads st.session_state.ply_slider
            result_now = st.session_state.watch_game_result
            max_ply = len(result_now["board_fens"]) - 1
            st.session_state.ply_slider = min(
                st.session_state.get("ply_slider", 0) + 1, max_ply
            )
            st.session_state.watch_playing = False
            st.rerun()

    result = st.session_state.watch_game_result
    if result is None:
        st.info("Select a model and press ▶ New Game.")
    else:
        fens = result["board_fens"]
        logs = result["turn_logs"]
        total_plies = len(fens)

        # If auto-advance queued a new position, apply it before the slider renders.
        # (Streamlit forbids writing to a widget's key after it is instantiated.)
        _default("ply_slider", 0)
        if "ply_pending" in st.session_state:
            st.session_state.ply_slider = st.session_state.pop("ply_pending")
        ply_idx = st.slider("Ply", 0, total_plies - 1, key="ply_slider")

        board = chess.Board(fens[ply_idx])
        lastmove = None
        if ply_idx > 0 and (ply_idx - 1) < len(logs):
            lastmove = chess.Move.from_uci(logs[ply_idx - 1]["move_uci"])

        col_board, col_info = st.columns([2, 1])
        with col_board:
            render_board(board, size=420, lastmove=lastmove)
        with col_info:
            st.subheader("Turn Info")
            if ply_idx > 0 and (ply_idx - 1) < len(logs):
                log = logs[ply_idx - 1]
                st.metric("Ply", log["ply"])
                st.write(f"**Side:** {log['color'].capitalize()}")
                pt_name = PIECE_TYPE_NAMES.get(log["selected_piece_type"], "?")
                st.write(
                    f"**Selected:** {pt_name.capitalize()} @ "
                    f"{log['selected_square']} (slot {log['selected_slot']})"
                )
                st.write(f"**Move:** {log['move_san']} ({log['move_uci']})")
                if log["is_capture"]:
                    st.write("💥 Capture!")
                st.write("**Type votes (weight → slot):**")
                votes = log.get("votes", {})
                wts = log.get("vote_weights", {})
                for pt, voted_slot in sorted(votes.items(), key=lambda x: -wts.get(x[0], 1)):
                    pt_name = PIECE_TYPE_NAMES.get(pt, "?").capitalize()
                    w = wts.get(pt, 1)
                    st.write(f"  {pt_name} (×{w}) → slot {voted_slot}")
            else:
                st.write("Start of game.")

        if ply_idx == total_plies - 1:
            st.divider()
            st.subheader("Game Over")
            st.write(f"**Result:** {result['board_result']}  |  **Length:** {result['game_length']} plies")
            if result["white_won"]:
                st.success("White wins!")
            elif result["black_won"]:
                st.success("Black wins!")
            else:
                st.info("Draw / Timeout")

            import pandas as pd
            rows = {"Color": [], "Slot": [], "Piece": [], "Survived": [], "Status": [], "Fitness": []}
            for pr in result["piece_results"]:
                rows["Color"].append("White" if pr["color"] == chess.WHITE else "Black")
                rows["Slot"].append(pr["slot_id"])
                rows["Piece"].append(PIECE_TYPE_NAMES.get(pr["original_piece_type"], "?").capitalize())
                rows["Survived"].append(pr["turns_survived"])
                rows["Status"].append("Captured" if pr["captured"] else "Alive")
                rows["Fitness"].append(f"{pr['fitness']:.3f}")
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        if st.session_state.watch_playing and ply_idx < total_plies - 1:
            time.sleep(0.5)
            # Stage the next position; applied to ply_slider before slider renders next run.
            st.session_state.ply_pending = ply_idx + 1
            st.rerun()
        elif st.session_state.watch_playing:
            st.session_state.watch_playing = False


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 3 — REASONING
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🧠 Reasoning":
    import pandas as pd

    st.title("🧠 Reasoning — Emergent Strategy Analysis")
    st.write(
        "Run games with the selected model to compute statistics on emergent strategy. "
        "Claude qualitative analysis is available if you supply an API key."
    )

    _default("reasoning_data", None)
    _default("reasoning_summary", None)

    saved = list_saved_models()
    use_training = (
        st.session_state.white_populations is not None
        and st.session_state.black_populations is not None
    )
    options = (["Current training model"] if use_training else []) + saved
    source = st.radio("Model source", options, key="reasoning_source", disabled=not options)

    if source == "Current training model":
        r_wp = st.session_state.white_populations
        r_bp = st.session_state.black_populations
    elif source:
        r_wp, r_bp = load_model(source)
    else:
        r_wp = r_bp = None

    col_ngames, col_api = st.columns([1, 2])
    with col_ngames:
        n_games = st.slider("Games to analyze", 10, 100, 20, 10)
    with col_api:
        api_key_env = __import__("os").environ.get("ANTHROPIC_API_KEY", "")
        if api_key_env:
            st.success("ANTHROPIC_API_KEY detected — Claude analysis enabled.")
            api_key_input = ""
        else:
            api_key_input = st.text_input(
                "Anthropic API Key (optional — enables Claude qualitative analysis)",
                type="password",
            )
    effective_api_key = api_key_env or api_key_input or None

    if st.button("🔍 Analyze Games", disabled=r_wp is None):
        reasoning_cfg = Config()
        with st.spinner(f"Playing {n_games} games and collecting logs…"):
            game_results = collect_game_logs(r_wp, r_bp, reasoning_cfg, n_games=n_games)
        st.session_state.reasoning_data = compute_analysis_data(game_results)
        if effective_api_key:
            with st.spinner("Sending logs to Claude for qualitative analysis…"):
                st.session_state.reasoning_summary = generate_strategy_summary(
                    game_results, api_key=effective_api_key
                )
        else:
            st.session_state.reasoning_summary = None
        st.rerun()

    # ── Helpers for table styling ─────────────────────────────────────────────
    def _highlight_row_max(s):
        max_val = s.max()
        if max_val <= 0:
            return [""] * len(s)
        return [
            "background-color: #DDEEFF; font-weight: bold" if v == max_val else ""
            for v in s
        ]

    def _style_per_piece(df: "pd.DataFrame"):
        return (
            df.style
            .highlight_max(subset=["Move Freq %"], color="#DDEEFF")
            .highlight_max(subset=["Survival Rate %"], color="#DDFFDD")
            .highlight_max(subset=["Avg Material Captured"], color="#FFE8CC")
            .format({
                "Move Freq %": "{:.1f}%",
                "Avg Survival": "{:.1f}",
                "Survival Rate %": "{:.1f}%",
                "Avg Material Captured": "{:.2f} pts",
                "Capture Rate %": "{:.1f}%",
                "Avg Fitness": "{:.3f}",
            })
        )

    # ── Display ───────────────────────────────────────────────────────────────
    data = st.session_state.reasoning_data
    if data:
        st.divider()

        # — Outcome stats —
        st.subheader("Game Outcomes")
        o = data["outcomes"]
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Games", o["n_games"])
        c2.metric("Avg Length", f"{o['avg_length']} plies")
        c3.metric("White Wins", f"{o['white_wins']}  ({o['white_win_pct']}%)")
        c4.metric("Black Wins", f"{o['black_wins']}  ({o['black_win_pct']}%)")
        c5.metric("Draws / Timeouts", f"{o['draws_timeouts']}  ({o['draws_pct']}%)")

        st.divider()

        # — Per-piece activity & survival —
        st.subheader("Per-Piece Activity & Survival")
        st.caption(
            "🔵 Highest move frequency  |  🟢 Best survival rate  |  🟠 Most material captured"
        )
        tab_w, tab_b = st.tabs(["♔ White", "♚ Black"])
        for tab, color_str in [(tab_w, "White"), (tab_b, "Black")]:
            with tab:
                df = data["per_piece"].get(color_str)
                if df is not None and not df.empty:
                    st.dataframe(
                        _style_per_piece(df),
                        use_container_width=True,
                        hide_index=True,
                    )
                else:
                    st.info("No data — run games with log_detail=True.")

        st.divider()

        # — Voting matrix —
        st.subheader("Voting Matrix")
        st.caption(
            "Row = voter piece type. Column = target piece type. "
            "Values = % of that type's votes cast for the target. "
            "🔵 Dominant preference per row. Combined across both sides."
        )
        voting_df = data["voting_df"]
        styled_voting = voting_df.style.apply(_highlight_row_max, axis=1).format("{:.1f}%")
        st.dataframe(styled_voting, use_container_width=True)

        st.divider()

        # — Game phase dynamics —
        st.subheader("Game Phase Dynamics")
        ph = data["phase_dynamics"]
        p1, p2, p3, p4, p5 = st.columns(5)
        p1.metric("Total Captures", ph["total"])
        p2.metric("Avg per Game", ph["avg_per_game"])
        p3.metric("Early (first ⅓)", f"{ph['early_pct']}%")
        p4.metric("Mid (middle ⅓)", f"{ph['mid_pct']}%")
        p5.metric("Late (last ⅓)", f"{ph['late_pct']}%")

    # — Claude qualitative analysis —
    if st.session_state.reasoning_summary:
        st.divider()
        st.subheader("Claude's Qualitative Analysis")
        st.markdown(st.session_state.reasoning_summary)
