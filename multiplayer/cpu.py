"""Simple CPU player: random votes and random moves."""

import random
import chess


def cpu_vote(living_pieces, board):
    """Vote for a random living piece that has at least one legal move."""
    movable = [
        p for p in living_pieces
        if any(m.from_square == p.current_square for m in board.legal_moves)
    ]
    pool = movable if movable else living_pieces
    return random.choice(pool).slot_id


def cpu_move(piece, board):
    """Pick a random legal move for the given piece. Auto-promotes to queen."""
    legal = [m for m in board.legal_moves if m.from_square == piece.current_square]
    queen_promos = [m for m in legal if m.promotion == chess.QUEEN]
    if queen_promos:
        return random.choice(queen_promos)
    return random.choice(legal)
