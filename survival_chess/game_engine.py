"""
Chess game loop for survival chess.

Rules:
  - Standard chess board/move/win conditions via python-chess.
  - Every turn has two phases:
      VOTING:  Each living piece on the active side independently votes for
               which friendly piece should move. Piece with most votes is selected.
               Ties broken randomly.
      MOVING:  The selected piece independently chooses among its legal moves.
  - Each piece uses its assigned voting/moving neural network.
  - Game ends on checkmate, stalemate, draw-by-rule, or max_turns plies.
  - On timeout: won=False for all pieces.

Piece slot indexing (0-15 per side):
  White: slots 0-7  → squares A1..H1 (back rank)
         slots 8-15 → squares A2..H2 (pawns)
  Black: slots 0-7  → squares A8..H8 (back rank)
         slots 8-15 → squares A7..H7 (pawns)
"""

import chess
import math
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from config import Config
from encoding import compute_features, MATERIAL_VALUES, TOTAL_MATERIAL
from neural_net import forward, softmax

# Starting squares per side, index = slot_id
WHITE_STARTING_SQUARES = [
    chess.A1, chess.B1, chess.C1, chess.D1,
    chess.E1, chess.F1, chess.G1, chess.H1,
    chess.A2, chess.B2, chess.C2, chess.D2,
    chess.E2, chess.F2, chess.G2, chess.H2,
]
BLACK_STARTING_SQUARES = [
    chess.A8, chess.B8, chess.C8, chess.D8,
    chess.E8, chess.F8, chess.G8, chess.H8,
    chess.A7, chess.B7, chess.C7, chess.D7,
    chess.E7, chess.F7, chess.G7, chess.H7,
]


@dataclass
class PieceInfo:
    slot_id: int
    color: chess.Color
    initial_square: int
    piece_type: int          # Updated on promotion
    original_piece_type: int # Stays constant — determines which population

    current_square: int = 0
    captured: bool = False
    capture_turn: Optional[int] = None
    turns_survived: int = 0  # Set at game end: ply of capture, or game_length

    capture_value: int = 0       # Sum of point values of pieces captured by this piece
    voting_individual_idx: int = 0
    moving_individual_idx: int = 0
    voting_weights: Optional[Dict] = field(default=None, repr=False)
    moving_weights: Optional[Dict] = field(default=None, repr=False)

    def __post_init__(self):
        self.current_square = self.initial_square


def _init_pieces(board: chess.Board, color: chess.Color) -> List[PieceInfo]:
    """Create PieceInfo list for one side using starting squares."""
    starting = WHITE_STARTING_SQUARES if color == chess.WHITE else BLACK_STARTING_SQUARES
    pieces = []
    for slot_id, sq in enumerate(starting):
        p = board.piece_at(sq)
        if p is None or p.color != color:
            # Shouldn't happen on a fresh board, but handle gracefully
            continue
        pieces.append(PieceInfo(
            slot_id=slot_id,
            color=color,
            initial_square=sq,
            piece_type=p.piece_type,
            original_piece_type=p.piece_type,
            current_square=sq,
        ))
    return pieces


def _assign_networks(pieces: List[PieceInfo], populations: Dict) -> None:
    """
    Draw one shared voting network (same index for all pieces on this side) and
    one moving network per piece type. All pieces share the same voting weights.
    """
    # Single shared voting network for all pieces
    v_pop = populations["voting"]
    v_idx = int(np.random.randint(len(v_pop)))

    # One moving network per piece type
    type_m_idx: Dict[int, int] = {}
    for pt in set(p.original_piece_type for p in pieces):
        m_pop = populations[(pt, "moving")]
        type_m_idx[pt] = int(np.random.randint(len(m_pop)))

    for piece in pieces:
        pt = piece.original_piece_type
        m_idx = type_m_idx[pt]
        piece.voting_individual_idx = v_idx
        piece.moving_individual_idx = m_idx
        piece.voting_weights = v_pop[v_idx]
        piece.moving_weights = populations[(pt, "moving")][m_idx]


def _vote(
    piece: PieceInfo,
    side_pieces: List[PieceInfo],
    board: chess.Board,
    turn: int,
) -> int:
    """
    The piece calls its voting network and returns a slot_id to vote for.
    Captured slots are masked to -inf before softmax.
    """
    features = compute_features(board, piece.current_square, piece.color, turn)
    logits = forward(piece.voting_weights, features).astype(np.float64)

    # Mask captured pieces
    for p in side_pieces:
        if p.captured and p.slot_id < len(logits):
            logits[p.slot_id] = -np.inf

    probs = softmax(logits)
    # np.random.choice needs probabilities that sum to 1.0; renormalize defensively
    probs = probs / probs.sum()
    return int(np.random.choice(len(probs), p=probs))


def _choose_move(
    piece: PieceInfo,
    legal_moves: List[chess.Move],
    board: chess.Board,
    turn: int,
    serendipity: float,
) -> chess.Move:
    """
    The piece calls its moving network (64 outputs, one per board square a1=0..h8=63).
    Illegal destination squares are masked to -inf before softmax.
    Pawn promotions automatically choose queen.
    """
    if not legal_moves:
        raise ValueError("No legal moves to choose from")

    features = compute_features(board, piece.current_square, piece.color, turn)
    logits = forward(piece.moving_weights, features).astype(np.float64)  # shape: (64,)

    # Group moves by destination square (handles multiple promotion variants)
    dest_to_moves: Dict[int, List[chess.Move]] = {}
    for m in legal_moves:
        dest_to_moves.setdefault(m.to_square, []).append(m)

    # Mask all squares; unmask legal destinations
    masked = np.full(64, -np.inf)
    for sq in dest_to_moves:
        masked[sq] = logits[sq]

    probs = softmax(masked, temperature=serendipity)
    probs = probs / probs.sum()
    chosen_sq = int(np.random.choice(64, p=probs))

    candidates = dest_to_moves[chosen_sq]
    # Auto-promote to queen
    for m in candidates:
        if m.promotion == chess.QUEEN:
            return m
    return candidates[0]


def play_game(
    white_populations: Dict,
    black_populations: Dict,
    config: Config,
    log_detail: bool = False,
) -> Dict:
    """
    Play one game of survival chess.

    white_populations and black_populations are independent — each side draws
    networks only from its own population dict.  This allows the training loop
    to freeze one side (supplying a historical HoF snapshot) while evolving
    the other.

    Returns a dict with:
      game_length    : int   — total plies played
      white_won      : bool
      black_won      : bool
      board_result   : str   — "1-0", "0-1", "1/2-1/2", or "timeout"
      piece_results  : list  — fitness/survival info per piece
      board_fens     : list  — FEN after each ply (only when log_detail=True)
      turn_logs      : list  — vote/move info per turn (only when log_detail=True)
    """
    board = chess.Board()

    white_pieces = _init_pieces(board, chess.WHITE)
    black_pieces = _init_pieces(board, chess.BLACK)
    _assign_networks(white_pieces, white_populations)
    _assign_networks(black_pieces, black_populations)

    # square -> PieceInfo lookup (updated as pieces move)
    sq_to_piece: Dict[int, PieceInfo] = {}
    for p in white_pieces + black_pieces:
        sq_to_piece[p.current_square] = p

    board_fens: List[str] = []
    turn_logs: List[Dict] = []
    capture_turns: Dict[int, int] = {}  # piece slot_id (with color key) -> ply of capture

    ply = 0  # half-move counter

    while not board.is_game_over() and ply < config.max_turns:
        color = board.turn
        side_pieces = white_pieces if color == chess.WHITE else black_pieces
        living_pieces = [p for p in side_pieces if not p.captured]

        if not living_pieces:
            break

        # ── Phase 1: Voting (one vote per piece type, log-weighted) ───────────
        vote_counts: Dict[int, float] = {p.slot_id: 0.0 for p in living_pieces}
        raw_votes: Dict[int, int] = {}    # piece_type -> voted_slot_id
        vote_weights: Dict[int, float] = {}  # piece_type -> weight

        # Group living pieces by original piece type
        type_to_living: Dict[int, List[PieceInfo]] = {}
        for p in living_pieces:
            type_to_living.setdefault(p.original_piece_type, []).append(p)

        enemy_king_sq = board.king(not color)

        for pt, pt_pieces in type_to_living.items():
            # Deterministic representative: under-attack piece closest to enemy king;
            # if none are under attack, pick the piece closest to the enemy king.
            attacked = [p for p in pt_pieces if board.is_attacked_by(not color, p.current_square)]
            pool = attacked if attacked else pt_pieces
            if enemy_king_sq is not None:
                rep = min(pool, key=lambda p: chess.square_distance(p.current_square, enemy_king_sq))
            else:
                rep = pool[0]
            voted_slot = _vote(rep, side_pieces, board, ply)
            # log₂(count + 2): pawns ≈ 3.2×, rooks/knights/bishops ≈ 2×, king/queen ≈ 1.6×
            weight = math.log2(len(pt_pieces) + 2)
            raw_votes[pt] = voted_slot
            vote_weights[pt] = weight
            if voted_slot in vote_counts:
                vote_counts[voted_slot] += weight
            # (votes for captured/invalid slots are silently dropped)

        max_votes = max(vote_counts.values()) if vote_counts else 0
        candidates = [p for p in living_pieces if vote_counts[p.slot_id] == max_votes]
        selected = candidates[np.random.randint(len(candidates))]

        # ── Phase 2: Moving ────────────────────────────────────────────────
        legal_moves = [m for m in board.legal_moves if m.from_square == selected.current_square]

        # If selected piece has no legal moves (pinned), fall back to any piece
        if not legal_moves:
            fallback = [
                p for p in living_pieces
                if any(m.from_square == p.current_square for m in board.legal_moves)
            ]
            if not fallback:
                break  # No piece can move — game over detected by board.is_game_over()
            selected = fallback[np.random.randint(len(fallback))]
            legal_moves = [m for m in board.legal_moves if m.from_square == selected.current_square]

        move_san = board.san(legal_moves[0])  # placeholder; update after choice
        chosen_move = _choose_move(selected, legal_moves, board, ply, config.serendipity)
        move_san = board.san(chosen_move)

        # ── Record state BEFORE pushing ────────────────────────────────────
        if log_detail:
            board_fens.append(board.fen())
            turn_logs.append({
                "ply": ply,
                "color": "white" if color == chess.WHITE else "black",
                "votes": dict(raw_votes),          # piece_type -> voted_slot_id
                "vote_weights": dict(vote_weights), # piece_type -> weight (count + 1)
                "vote_counts": dict(vote_counts),   # slot_id -> total weighted votes
                "selected_slot": selected.slot_id,
                "selected_piece_type": selected.piece_type,
                "selected_square": chess.square_name(selected.current_square),
                "move_uci": chosen_move.uci(),
                "move_san": move_san,
                "is_capture": board.is_capture(chosen_move),
            })

        # ── Handle captures BEFORE pushing ────────────────────────────────
        captured_info: Optional[PieceInfo] = None

        if board.is_en_passant(chosen_move):
            # Captured pawn is one rank behind the destination
            ep_rank = 4 if color == chess.WHITE else 3
            ep_sq = chess.square(chess.square_file(chosen_move.to_square), ep_rank)
            captured_info = sq_to_piece.pop(ep_sq, None)
        elif board.is_capture(chosen_move):
            captured_info = sq_to_piece.pop(chosen_move.to_square, None)

        if captured_info is not None:
            captured_info.captured = True
            captured_info.capture_turn = ply
            selected.capture_value += MATERIAL_VALUES.get(captured_info.piece_type, 0)

        # ── Update square mapping ──────────────────────────────────────────
        from_sq = chosen_move.from_square
        to_sq = chosen_move.to_square

        # Castling: python-chess automatically moves the rook on the board,
        # but we must also update sq_to_piece for the rook before pushing.
        if board.is_castling(chosen_move):
            if board.is_kingside_castling(chosen_move):
                rook_from = chess.H1 if color == chess.WHITE else chess.H8
                rook_to   = chess.F1 if color == chess.WHITE else chess.F8
            else:
                rook_from = chess.A1 if color == chess.WHITE else chess.A8
                rook_to   = chess.D1 if color == chess.WHITE else chess.D8
            rook_info = sq_to_piece.pop(rook_from, None)
            if rook_info is not None:
                rook_info.current_square = rook_to
                sq_to_piece[rook_to] = rook_info

        if from_sq in sq_to_piece:
            mover = sq_to_piece.pop(from_sq)
            mover.current_square = to_sq
            sq_to_piece[to_sq] = mover

            # Handle promotion
            if chosen_move.promotion is not None:
                mover.piece_type = chosen_move.promotion

        board.push(chosen_move)
        ply += 1

    # ── Determine result ───────────────────────────────────────────────────
    game_length = ply
    white_won = False
    black_won = False
    board_result = "timeout"

    if board.is_checkmate():
        # The side to move is in checkmate → lost; the other side won
        if board.turn == chess.BLACK:
            white_won = True
            board_result = "1-0"
        else:
            black_won = True
            board_result = "0-1"
    elif board.is_game_over():
        board_result = board.result()  # "1/2-1/2" or similar

    # ── Compute survival and fitness ───────────────────────────────────────
    all_pieces = white_pieces + black_pieces

    # turns_survived = ply of capture, or game_length if still alive
    for p in all_pieces:
        p.turns_survived = p.capture_turn if p.captured else game_length

    white_survivals = [p.turns_survived for p in white_pieces]
    black_survivals = [p.turns_survived for p in black_pieces]

    piece_results = []
    for p in all_pieces:
        won = white_won if p.color == chess.WHITE else black_won
        fitness = _piece_fitness(p.turns_survived, won, config, p.capture_value)
        piece_results.append({
            "slot_id": p.slot_id,
            "color": p.color,
            "original_piece_type": p.original_piece_type,
            "piece_type_final": p.piece_type,
            "voting_individual_idx": p.voting_individual_idx,
            "moving_individual_idx": p.moving_individual_idx,
            "turns_survived": p.turns_survived,
            "captured": p.captured,
            "capture_turn": p.capture_turn,
            "capture_value": p.capture_value,
            "fitness": fitness,
        })

    # Append final board state to fens
    if log_detail:
        board_fens.append(board.fen())

    return {
        "game_length": game_length,
        "white_won": white_won,
        "black_won": black_won,
        "board_result": board_result,
        "piece_results": piece_results,
        "board_fens": board_fens,
        "turn_logs": turn_logs,
    }


def _piece_fitness(
    turns_survived: int,
    won: bool,
    config: Config,
    capture_value: int = 0,
) -> float:
    """Compute raw per-piece fitness: absolute survival + win bonus + capture bonus."""
    abs_survival = (turns_survived / max(config.max_turns, 1)) ** config.gamma
    win_bonus = 1.0 if won else 0.0
    capture_bonus = capture_value / TOTAL_MATERIAL
    return (
        config.alpha * abs_survival
        + config.beta * win_bonus
        + config.kappa * capture_bonus
    )
