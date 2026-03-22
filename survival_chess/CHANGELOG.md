# Survival Chess — Version History

---

## 2.0.0 — Architecture overhaul

### Conceptual
- **64-square target encoding**: the moving network now outputs one logit per board square (64 total) instead of one logit per legal move (up to 32). Illegal destination squares are masked to −∞ before softmax. Pawn promotions auto-select queen. This gives the network a fixed, position-aware output space rather than an arbitrary index into a sorted move list.
- **Single shared voting network per side**: collapsed 6 per-type voting networks into 1 shared voting network. All piece types on a side draw from the same voting population (key: `"voting"`). Per-type moving networks are unchanged. Population structure changes from 12 keys to 7 per side.
- **Friendly-piece features**: the input vector now includes the 3 nearest friendly pieces (excluding the own king, which is already encoded separately). Each friendly adds: piece-type one-hot (6) + normalised distance (1) + Δfile/7 (1) + Δrank/7 (1) = 9 dims. Zero-padded if fewer than 3 exist. Input size: 45 → 72. Network widths scaled up accordingly: hidden1 64 → 96, hidden2 32 → 48.
- **Deterministic voting representative**: replaced random representative selection with a deterministic rule. For each piece type, the representative is the attacked piece closest to the enemy king; if no piece of that type is under attack, the piece closest to the enemy king is used. This makes the voting signal more informative and reproducible.
- **Simplified fitness — 3 terms**: removed the relative-survival term (δ) and voting-credit term (ζ). The fitness function is now: `f = α·(t/T_max)^γ + β·w + κ·V/39`. Rationale: δ added noise (opponent mean survival fluctuates heavily game-to-game) and ζ required fragile post-hoc vote-quality bookkeeping that offered marginal signal.
- **Training stability tweaks**: `hall_of_fame_size` 8 → 20 (broader opponent diversity, reduces cycling), `serendipity` 0.8 → 0.4 (sharper move selection, faster exploitation), `ema_decay` 0.5 → 0.3 (less historical inertia, faster adaptation to current opponents).

### Technical
- `config.py`: `input_size` 45 → 72; `hidden1` 64 → 96; `hidden2` 32 → 48; `moving_output_size` 32 → 64; removed `delta` and `zeta`; `serendipity` 0.8 → 0.4; `ema_decay` 0.5 → 0.3; `hall_of_fame_size` 8 → 20.
- `encoding.py`: `FEATURE_SIZE` 45 → 72; `compute_features` extended with [45–71] nearest-3-friendly block.
- `evolution.py`: `init_populations` creates 7 keys (`"voting"` + 6×`(pt, "moving")`); `aggregate_fitness_for_side` updated to match; `piece_fitness` signature simplified (removed `opponent_avg_survival`, `delta`, `zeta`, `vote_quality`).
- `game_engine.py`: `_assign_networks` draws one shared voting index for all pieces; `_choose_move` rewritten for 64-square masked output with auto-queen promotion; voting loop uses deterministic representative; removed `color_type_votes`, `_is_good_piece`, vote-quality computation; `_piece_fitness` simplified to 3 terms; `Tuple` import removed.
- `app.py`: removed δ and ζ sliders; updated `Config(...)` call; fitness formula box updated; slider defaults updated to match new config defaults.
- `test_smoke.py`: step 5 updated to verify simplified fitness (non-negative, finite).

---

## 1.2.0 — Log vote weighting + voting credit signal

### Conceptual
- **Log vote weighting (#2)**: vote weight changed from `count + 1` to `log₂(count + 2)`. Starting weights: pawns ≈ 3.2×, knights/bishops/rooks = 2×, queen/king ≈ 1.6×. This reduces pawn dominance (was 9× vs king's 2×, now 3.2× vs 1.6×) and prevents a numerically dominant type from dictating all piece selection before moving networks have matured.
- **Voting credit signal — ζ (zeta, #3)**: new fitness term that directly rewards voting networks for picking "good" pieces. After each game, each type receives a `vote_quality` score `q ∈ [0,1]` = fraction of its votes cast for pieces that either survived above the side's median survival or made at least one capture. This is the first direct credit signal for voting quality; previously voting networks were evaluated entirely on piece survival, which is mostly driven by moving decisions.
- Full fitness function:

  `f = α·(t/T_max)^γ + δ·(t/t̄_opp − 1) + β·w + κ·V/39 + ζ·q`

### Technical
- `config.py`: added `zeta: float = 0.5`.
- `game_engine.py`: `import math` added; weight formula changed to `math.log2(len(pt_pieces) + 2)`; `vote_counts` and `vote_weights` changed to `float` types; `color_type_votes` dict accumulates `(color, piece_type) → [voted_slot_ids]` across all turns; post-game vote quality computation using side median and capture flags; `_piece_fitness` gains `vote_quality` argument; `vote_quality` included in `piece_results`.
- `evolution.py`: standalone `piece_fitness()` updated with `zeta` and `vote_quality` parameters.
- `app.py`: ζ slider (0–2) added; passed to `Config`; LaTeX formula and variable table updated.

---

## 1.1.0 — Type-level shared networks

### Conceptual
- **Shared networks per piece type**: all pieces of the same type now share a single voting and moving network for the duration of a game (drawn from that type's population). Previously every individual piece was assigned its own independently-drawn network.
- **Weighted type voting**: the voting phase now calls one network per living piece type rather than one per piece. Each type's vote is weighted by `(number of living pieces of that type + 1)` — so a side with 8 pawns contributes a vote of weight 9, while the lone king contributes weight 2. This gives more numerous piece types greater collective influence, reflecting their strength in numbers.
- The representative piece used to generate input features for a type's voting call is chosen **randomly** from the living pieces of that type each turn, so the same type can vote differently turn-to-turn as its membership changes.
- Moving phase is unchanged: the selected piece feeds its own specific board features into the (shared) moving network for its type.

### Technical
- `game_engine.py` — `_assign_networks`: draws one `(v_idx, m_idx)` pair per piece type; all pieces of that type receive identical `voting_weights`, `moving_weights`, and indices for that game.
- `game_engine.py` — voting loop in `play_game`: replaced per-piece loop with per-type loop. Groups living pieces by `original_piece_type`, picks a random representative, calls `_vote(rep, ...)`, accumulates weighted vote counts.
- Turn log gains `"vote_weights"` field (`piece_type → weight`); `"votes"` now maps `piece_type → voted_slot_id` (previously `voter_slot_id → voted_slot_id`).
- `app.py` — Watch page vote display updated to show type name, weight, and target slot.
- Population structure, evolution, fitness aggregation, and moving phase are all unchanged.

Versions are grouped by meaningful design milestones.
Each entry separates **conceptual changes** (rules, fitness function, training philosophy) from **technical changes** (implementation, architecture, UI, parameters).

---

## 1.0.0 — Reasoning page v2 (current)

### Conceptual
- Reasoning page now runs **20 games** instead of 5 for more statistically reliable analysis.
- Statistical analysis is available **without an API key**: per-piece-type survival rate, capture rate, average fitness, average material captured, and selection frequency are always shown.
- Claude qualitative analysis is now an optional layer on top of the statistics rather than the only output.

### Technical
- Added `build_statistical_summary()` in `reasoning.py` — generates a markdown table from raw game results, no external calls.
- Reasoning page button no longer gated on API key presence.
- `collect_game_logs()` default changed from `n_games=5` to `n_games=20`.

---

## 0.6.0 — Capture bonus term

### Conceptual
- Added **capture bonus** to the fitness function: pieces are rewarded for the standard point value of enemy pieces they capture.
- Addresses passive survival equilibrium: previously pieces had no direct incentive to take material, only to stay alive.
- New parameter **κ (kappa)**: scales the capture bonus. Capture value is normalized by 39 (total starting material) so the term is on the same scale as other fitness components.
- Full fitness function:

  `f = α·(t/T_max)^γ + δ·(t/t̄_opp − 1) + β·w + κ·V/39`

### Technical
- `config.py`: added `kappa: float = 1.0`.
- `game_engine.py`: `PieceInfo` gains `capture_value: int = 0`; accrues `MATERIAL_VALUES[captured.piece_type]` on each capture by that piece. `_piece_fitness` gains `capture_value` argument.
- `evolution.py`: standalone `piece_fitness()` updated with `kappa` and `capture_value` parameters.
- `app.py`: κ slider (0–3) added to Fitness Function sidebar section; LaTeX formula and variable table updated.

---

## 0.5.0 — UI improvements & parameter exposure

### Conceptual
- No changes to game rules or fitness function.

### Technical
- **Watch page fixes**:
  - Line chart now uses `.ffill()` so fitness lines render continuously (previously only showed hover points).
  - Step button correctly advances one ply by writing to `st.session_state.ply_slider`.
  - Resume/Pause toggle added for auto-playback.
  - Fixed `StreamlitAPIException` (widget key modified after instantiation) using a `ply_pending` staging key: the new value is written after the slider renders and consumed before it renders on the next rerun.
- **Parameter exposure**: `elite_count` (1–10) and `anneal_interval` (1–100 gens per ×0.9 std decay) added as sidebar sliders under the Evolution section.
- **Fitness formula display**: LaTeX formula and variable-explanation table always visible in a collapsible box at the bottom of the Train page.

---

## 0.4.0 — Convergence improvements

### Conceptual
- **EMA fitness smoothing (Fix C)**: selection is based on an exponential moving average of fitness across generations rather than a single generation's raw mean. This reduces noise and prevents lucky/unlucky game samples from dominating selection. Children warm-start from their parent elite's EMA discounted by one step.
- **Fixed survival normalization (Fix D)**: absolute survival fraction now normalizes against `max_turns` (fixed ceiling = 200) rather than the actual game length. A piece that survives 100 plies in a 100-ply game gets `(100/200)^γ`, not 1.0, preserving signal across games of different lengths.
- **Faster mutation annealing (Fix A)**: mutation std decays by ×0.9 every `anneal_interval` generations (default 10), replacing the original slow halving schedule. Populations converge to a tighter region over time rather than exploring indefinitely.

### Technical
- `config.py`: `mutation_std_initial` 0.15 → 0.08; `mutation_anneal_interval` default 10 (×0.9 per interval, not ÷2); `ema_decay: float = 0.5` added.
- `evolution.py`: `evolve_populations()` now takes `ema_fitness` dict in and returns `new_ema_fitness` dict out; `piece_fitness()` uses `max_turns` not `game_length`.
- `app.py`: EMA state (`white_ema_fitness`, `black_ema_fitness`) threaded through training loop.

---

## 0.3.0 — Co-evolutionary architecture

### Conceptual
- **Separate white and black populations**: white and black pieces now evolve independently rather than sharing a single population. This prevents the two sides from being co-adapted in a way that is fragile to opponent changes.
- **Alternating training**: only one side trains per generation; the other side supplies a frozen opponent. This avoids the Red Queen treadmill where both sides chase each other and no real learning accumulates.
- **Hall of Fame (HoF) opponent pool**: the frozen opponent is sampled from a rolling window of elite snapshots from past generations of the idle side. This exposes the training side to a diverse set of opponents rather than just the current best, reducing co-evolutionary cycling.
- **Variance-based early stopping**: training stops when the within-population weight variance falls below a threshold for both sides simultaneously. This detects genuine convergence (all individuals have collapsed onto a stable solution) rather than relying on a fixed generation count or a noisy fitness plateau signal.
- **Non-zero-sum delta term**: the relative survival term now compares each piece to the *opponent's* mean survival rather than its own side's. A white piece earns a bonus for outlasting the average black piece, making the fitness landscape non-zero-sum — the entire white team can gain fitness together.

### Technical
- `evolution.py`: `init_populations()` is called twice (once for white, once for black). `aggregate_fitness_for_side()` added — aggregates only for one color. `population_weight_variance()` added. `snapshot_elites()` and `sample_opponent()` added for HoF management. `save_model` / `load_model` updated to save/load `{white: ..., black: ...}` dicts.
- `game_engine.py`: `play_game()` signature changed to `play_game(white_populations, black_populations, config)`; each side draws networks only from its own populations.
- `app.py`: session state updated for `white_populations`, `black_populations`, `white_hof`, `black_hof`, `white_ema_fitness`, `black_ema_fitness`, `training_side`. Training loop alternates side each generation.
- `test_smoke.py`: updated for new `play_game`, `evolve_populations`, and `snapshot_elites` signatures (7 tests).

---

## 0.2.0 — Castling bug fix

### Conceptual
- No changes to game rules or fitness function.

### Technical
- Fixed `AssertionError: No piece at square 63` during castling.
- Root cause: `python-chess` automatically moves the rook when `board.push()` is called for a castling move, but the `sq_to_piece` lookup dict was not updated for the rook.
- Fix: detect castling before `board.push()` using `board.is_castling()` / `board.is_kingside_castling()` and manually update the rook's entry in `sq_to_piece`.

---

## 0.1.0 — Initial implementation

### Conceptual
- **Game rules**: standard chess board and move legality via `python-chess`. Each turn has two phases:
  - *Voting*: every living piece on the active side independently votes for which friendly piece should move; the piece with the most votes is selected (ties broken randomly).
  - *Moving*: the selected piece independently chooses among its legal moves.
- **Fitness function** (original):

  `f = α·(t/L)^γ + δ·(t/t̄_own − 1) + β·w`

  where `L` is the actual game length and `t̄_own` is the own-side mean survival. Both normalization choices were later revised (see 0.3.0 and 0.4.0).
- **Piece identity**: each piece retains its original type for population assignment even after pawn promotion.
- **Timeout**: if the game reaches `max_turns` (200 plies) without a decisive result, `won = False` for all pieces.

### Technical
- `config.py`: central dataclass for all hyperparameters.
- `encoding.py`: 45-dimensional feature vector per piece (position, piece type, mobility, nearest threats, king positions, material balance, turn, attack status, check, game phase).
- `neural_net.py`: NumPy-only MLP (`make_network`, `forward`, `softmax`, `mutate`, `copy_weights`). No autograd.
- `game_engine.py`: full game loop with voting and moving phases; castling, en passant, and promotion handling; piece slot indexing (0–15 per side).
- `evolution.py`: `init_populations()`, `piece_fitness()`, `aggregate_fitness()`, `evolve_populations()` (elites + mutation), `save_model()` / `load_model()`.
- `reasoning.py`: collects game logs and sends them to Claude (claude-sonnet-4-6) for emergent strategy analysis.
- `app.py`: 3-page Streamlit UI — Train (live metrics, hyperparameter sliders), Watch (board replay with step/pause/resume), Reasoning (Claude strategy summary).
- `test_smoke.py`: smoke test covering imports, feature size, game play, fitness, and evolution.
- Networks: 12 total (6 piece types × voting/moving). Voting output: 16 logits (one per piece slot). Moving output: 32 logits (first N used for N legal moves).
