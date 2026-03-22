"""
Board -> feature vector encoding for survival chess.

Feature layout (72 total):
  [0-1]   Own position normalized (file/7, rank/7)
  [2-7]   Own piece type one-hot (PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING)
  [8]     Own mobility (legal moves / 28)
  [9-35]  Nearest 3 enemy threats, each 9 dims:
              piece_type_one_hot (6) + dist_norm (1) + delta_file/7 (1) + delta_rank/7 (1)
  [36-37] Own king relative position (delta_file/7, delta_rank/7)
  [38-39] Enemy king relative position (delta_file/7, delta_rank/7)
  [40]    Material balance (own - enemy) / 39
  [41]    Turn number normalized (turn / 200)
  [42]    Under direct attack this turn (binary)
  [43]    King in check (binary)
  [44]    Game phase (0=opening, 0.5=midgame, 1=endgame)
  [45-71] Nearest 3 friendly pieces (excluding own king), each 9 dims:
              piece_type_one_hot (6) + dist_norm (1) + delta_file/7 (1) + delta_rank/7 (1)
              zero-padded if fewer than 3 friendly pieces exist
"""

import chess
import numpy as np

PIECE_TYPES = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]
PIECE_TYPE_TO_IDX = {pt: i for i, pt in enumerate(PIECE_TYPES)}

MATERIAL_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 0,
}
TOTAL_MATERIAL = 39  # 8*1 + 2*3 + 2*3 + 2*5 + 1*9

MAX_DIST = (7.0 ** 2 + 7.0 ** 2) ** 0.5  # ~9.9, max board distance

FEATURE_SIZE = 72


def _piece_one_hot(piece_type: int) -> np.ndarray:
    vec = np.zeros(6, dtype=np.float32)
    vec[PIECE_TYPE_TO_IDX[piece_type]] = 1.0
    return vec


def compute_features(
    board: chess.Board,
    square: int,
    color: chess.Color,
    turn: int,
) -> np.ndarray:
    """
    Compute the 72-dim feature vector for the piece at `square` of `color`.
    `turn` is the current half-move count (0-indexed).
    """
    piece = board.piece_at(square)
    assert piece is not None, f"No piece at square {square}"

    file = chess.square_file(square)
    rank = chess.square_rank(square)
    enemy_color = not color

    features = np.zeros(FEATURE_SIZE, dtype=np.float32)
    idx = 0

    # --- [0-1] Own position ---
    features[idx] = file / 7.0
    features[idx + 1] = rank / 7.0
    idx += 2

    # --- [2-7] Piece type one-hot ---
    features[idx: idx + 6] = _piece_one_hot(piece.piece_type)
    idx += 6

    # --- [8] Mobility ---
    legal_from_sq = sum(1 for m in board.legal_moves if m.from_square == square)
    features[idx] = legal_from_sq / 28.0
    idx += 1

    # --- [9-35] Nearest 3 enemy threats ---
    attacker_squares = list(board.attackers(enemy_color, square))
    threats = []
    for att_sq in attacker_squares:
        att_piece = board.piece_at(att_sq)
        if att_piece is None:
            continue
        df = chess.square_file(att_sq) - file
        dr = chess.square_rank(att_sq) - rank
        dist = (df ** 2 + dr ** 2) ** 0.5
        threats.append((dist, att_sq, att_piece, df, dr))

    threats.sort(key=lambda x: x[0])
    threats = threats[:3]

    for i in range(3):
        if i < len(threats):
            dist, att_sq, att_piece, df, dr = threats[i]
            features[idx: idx + 6] = _piece_one_hot(att_piece.piece_type)
            features[idx + 6] = dist / MAX_DIST
            features[idx + 7] = df / 7.0
            features[idx + 8] = dr / 7.0
        # else: zeros (already initialized)
        idx += 9

    # --- [36-37] Own king relative position ---
    own_king_sq = board.king(color)
    if own_king_sq is not None:
        features[idx] = (chess.square_file(own_king_sq) - file) / 7.0
        features[idx + 1] = (chess.square_rank(own_king_sq) - rank) / 7.0
    idx += 2

    # --- [38-39] Enemy king relative position ---
    enemy_king_sq = board.king(enemy_color)
    if enemy_king_sq is not None:
        features[idx] = (chess.square_file(enemy_king_sq) - file) / 7.0
        features[idx + 1] = (chess.square_rank(enemy_king_sq) - rank) / 7.0
    idx += 2

    # --- [40] Material balance ---
    own_mat = sum(
        MATERIAL_VALUES[pt] * len(board.pieces(pt, color))
        for pt in PIECE_TYPES
    )
    enemy_mat = sum(
        MATERIAL_VALUES[pt] * len(board.pieces(pt, enemy_color))
        for pt in PIECE_TYPES
    )
    features[idx] = (own_mat - enemy_mat) / TOTAL_MATERIAL
    idx += 1

    # --- [41] Turn normalized ---
    features[idx] = turn / 200.0
    idx += 1

    # --- [42] Under direct attack ---
    features[idx] = 1.0 if board.is_attacked_by(enemy_color, square) else 0.0
    idx += 1

    # --- [43] King in check ---
    features[idx] = 1.0 if board.is_check() else 0.0
    idx += 1

    # --- [44] Game phase ---
    total_pieces = sum(
        len(board.pieces(pt, c))
        for pt in [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]
        for c in [chess.WHITE, chess.BLACK]
    )
    if turn <= 20:
        features[idx] = 0.0   # opening
    elif total_pieces <= 6:
        features[idx] = 1.0   # endgame
    else:
        features[idx] = 0.5   # midgame
    idx += 1

    # --- [45-71] Nearest 3 friendly pieces (excluding own king) ---
    own_king_sq_for_friendly = board.king(color)
    friendlies = []
    for sq in board.pieces(chess.PAWN, color) | board.pieces(chess.KNIGHT, color) | \
               board.pieces(chess.BISHOP, color) | board.pieces(chess.ROOK, color) | \
               board.pieces(chess.QUEEN, color):
        if sq == square:
            continue  # exclude self
        f_piece = board.piece_at(sq)
        if f_piece is None:
            continue
        df = chess.square_file(sq) - file
        dr = chess.square_rank(sq) - rank
        dist = (df ** 2 + dr ** 2) ** 0.5
        friendlies.append((dist, sq, f_piece, df, dr))

    friendlies.sort(key=lambda x: x[0])
    friendlies = friendlies[:3]

    for i in range(3):
        if i < len(friendlies):
            dist, sq, f_piece, df, dr = friendlies[i]
            features[idx: idx + 6] = _piece_one_hot(f_piece.piece_type)
            features[idx + 6] = dist / MAX_DIST
            features[idx + 7] = df / 7.0
            features[idx + 8] = dr / 7.0
        # else: zeros (already initialized)
        idx += 9

    assert idx == FEATURE_SIZE, f"Feature size mismatch: {idx} != {FEATURE_SIZE}"
    return features
