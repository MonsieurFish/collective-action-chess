"""Game state machine: board logic, voting, moving."""

import chess
import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

from cpu import cpu_vote, cpu_move
from room import PIECE_TYPE_NAMES

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

MAX_PLIES = 200


@dataclass
class PieceInfo:
    slot_id: int
    color: bool           # chess.WHITE or chess.BLACK
    piece_type: int       # Updated on promotion
    original_piece_type: int
    current_square: int
    captured: bool = False


class Game:
    def __init__(self, white_assignments: Dict[int, Optional[str]],
                 black_assignments: Dict[int, Optional[str]]):
        """
        assignments: {piece_type: player_sid_or_None} per side.
        None means CPU. King is always None.
        """
        self.board = chess.Board()
        self.white_pieces = self._init_pieces(chess.WHITE)
        self.black_pieces = self._init_pieces(chess.BLACK)

        self.sq_to_piece: Dict[int, PieceInfo] = {}
        for p in self.white_pieces + self.black_pieces:
            self.sq_to_piece[p.current_square] = p

        self.phase = "voting"       # voting | moving | game_over
        self.active_color = chess.WHITE

        # Player assignments
        self.assignments: Dict[bool, Dict[int, Optional[str]]] = {
            chess.WHITE: white_assignments,
            chess.BLACK: black_assignments,
        }
        # Reverse lookup: sid -> (color, piece_type)
        self.sid_to_assignment: Dict[str, Tuple[bool, int]] = {}
        for color in (chess.WHITE, chess.BLACK):
            for pt, sid in self.assignments[color].items():
                if sid is not None:
                    self.sid_to_assignment[sid] = (color, pt)

        # Voting state (reset each turn)
        self.votes: Dict[int, int] = {}          # piece_type -> voted slot_id
        self.human_votes_needed: Set[int] = set()  # piece_types still pending

        # Moving state
        self.selected_piece: Optional[PieceInfo] = None
        self.legal_moves_for_selected: List[chess.Move] = []

        self.move_history: List[dict] = []
        self.ply = 0
        self.eliminated_sids: Set[str] = set()

    # ── Initialization ───────────────────────────────────────────────

    def _init_pieces(self, color: bool) -> List[PieceInfo]:
        starting = WHITE_STARTING_SQUARES if color == chess.WHITE else BLACK_STARTING_SQUARES
        pieces = []
        for slot_id, sq in enumerate(starting):
            p = self.board.piece_at(sq)
            if p is None or p.color != color:
                continue
            pieces.append(PieceInfo(
                slot_id=slot_id,
                color=color,
                piece_type=p.piece_type,
                original_piece_type=p.piece_type,
                current_square=sq,
            ))
        return pieces

    # ── Helpers ──────────────────────────────────────────────────────

    def side_pieces(self, color: bool) -> List[PieceInfo]:
        return self.white_pieces if color == chess.WHITE else self.black_pieces

    def living_pieces(self, color: bool) -> List[PieceInfo]:
        return [p for p in self.side_pieces(color) if not p.captured]

    def living_types(self, color: bool) -> Set[int]:
        return {p.original_piece_type for p in self.living_pieces(color)}

    def type_count(self, color: bool, piece_type: int) -> int:
        return sum(1 for p in self.living_pieces(color)
                   if p.original_piece_type == piece_type)

    def color_str(self, color: bool) -> str:
        return "white" if color == chess.WHITE else "black"

    def piece_by_slot(self, color: bool, slot_id: int) -> Optional[PieceInfo]:
        for p in self.side_pieces(color):
            if p.slot_id == slot_id:
                return p
        return None

    def get_living_pieces_info(self, color: bool) -> List[dict]:
        """Return serializable list of living pieces for the client."""
        result = []
        for p in self.living_pieces(color):
            result.append({
                "slot_id": p.slot_id,
                "piece_type": p.original_piece_type,
                "type_name": PIECE_TYPE_NAMES.get(p.original_piece_type, "?"),
                "square": chess.square_name(p.current_square),
                "symbol": self._piece_symbol(p),
            })
        return result

    def _piece_symbol(self, p: PieceInfo) -> str:
        symbols = {
            chess.PAWN: "♙" if p.color == chess.WHITE else "♟",
            chess.KNIGHT: "♘" if p.color == chess.WHITE else "♞",
            chess.BISHOP: "♗" if p.color == chess.WHITE else "♝",
            chess.ROOK: "♖" if p.color == chess.WHITE else "♜",
            chess.QUEEN: "♕" if p.color == chess.WHITE else "♛",
            chess.KING: "♔" if p.color == chess.WHITE else "♚",
        }
        return symbols.get(p.piece_type, "?")

    # ── Voting Phase ─────────────────────────────────────────────────

    def start_voting(self) -> List[Tuple[int, int]]:
        """Begin voting. CPU types vote immediately.

        Returns list of (piece_type, slot_id) for CPU votes cast.
        """
        self.phase = "voting"
        self.votes = {}
        self.human_votes_needed = set()

        alive_types = self.living_types(self.active_color)
        assignments = self.assignments[self.active_color]
        living = self.living_pieces(self.active_color)

        cpu_votes_cast = []
        for pt in alive_types:
            sid = assignments.get(pt)
            if sid is None or sid in self.eliminated_sids:
                # CPU vote
                slot = cpu_vote(living, self.board)
                self.votes[pt] = slot
                cpu_votes_cast.append((pt, slot))
            else:
                self.human_votes_needed.add(pt)

        return cpu_votes_cast

    def submit_vote(self, piece_type: int, slot_id: int):
        """Record a human player's vote. Returns (ok, message)."""
        if self.phase != "voting":
            return False, "Not in voting phase"
        if piece_type not in self.human_votes_needed:
            return False, "Vote not expected from this piece type"

        valid_slots = {p.slot_id for p in self.living_pieces(self.active_color)}
        if slot_id not in valid_slots:
            return False, "Invalid piece selection"

        self.votes[piece_type] = slot_id
        self.human_votes_needed.discard(piece_type)
        return True, "Vote recorded"

    def all_votes_in(self) -> bool:
        return len(self.human_votes_needed) == 0

    def votes_progress(self) -> Tuple[int, int]:
        """Returns (votes_received, total_human_voters)."""
        total = len(self.human_votes_needed) + sum(
            1 for pt in self.living_types(self.active_color)
            if self.assignments[self.active_color].get(pt) is not None
            and self.assignments[self.active_color].get(pt) not in self.eliminated_sids
        )
        received = total - len(self.human_votes_needed)
        return received, total

    def tally_votes(self) -> Optional[PieceInfo]:
        """Tally weighted votes, select piece. Returns selected PieceInfo or None."""
        living = self.living_pieces(self.active_color)
        if not living:
            return None

        vote_totals: Dict[int, float] = {p.slot_id: 0.0 for p in living}

        for pt, slot_id in self.votes.items():
            count = self.type_count(self.active_color, pt)
            weight = math.log2(count + 2)
            if slot_id in vote_totals:
                vote_totals[slot_id] += weight

        # Pick highest; break ties randomly
        max_val = max(vote_totals.values()) if vote_totals else 0
        candidates = [p for p in living if vote_totals.get(p.slot_id, 0) == max_val]
        selected = random.choice(candidates)

        # Check for legal moves, fallback if pinned
        legal = [m for m in self.board.legal_moves
                 if m.from_square == selected.current_square]
        if not legal:
            # Try other pieces in order of votes
            sorted_pieces = sorted(
                living,
                key=lambda p: vote_totals.get(p.slot_id, 0),
                reverse=True,
            )
            for p in sorted_pieces:
                legal = [m for m in self.board.legal_moves
                         if m.from_square == p.current_square]
                if legal:
                    selected = p
                    break
            else:
                return None  # Nobody can move

        self.selected_piece = selected
        self.legal_moves_for_selected = [
            m for m in self.board.legal_moves
            if m.from_square == selected.current_square
        ]
        self.phase = "moving"
        return selected

    # ── Moving Phase ─────────────────────────────────────────────────

    def get_mover_sid(self) -> Optional[str]:
        """SID of the player who must move, or None for CPU."""
        if self.selected_piece is None:
            return None
        pt = self.selected_piece.original_piece_type
        sid = self.assignments[self.active_color].get(pt)
        if sid and sid in self.eliminated_sids:
            return None
        return sid

    def get_legal_moves_uci(self) -> List[dict]:
        """Legal moves for the selected piece, as dicts."""
        result = []
        for m in self.legal_moves_for_selected:
            result.append({
                "uci": m.uci(),
                "san": self.board.san(m),
            })
        return result

    def make_cpu_move(self):
        """CPU makes a random move for the selected piece."""
        move = cpu_move(self.selected_piece, self.board)
        return self.execute_move(move.uci())

    def execute_move(self, uci_str: str):
        """Execute a move. Returns (move_info, eliminated, game_result_or_None)."""
        try:
            move = chess.Move.from_uci(uci_str)
        except ValueError:
            return None, "Invalid move format"

        if move not in self.legal_moves_for_selected:
            return None, "Illegal move"

        selected = self.selected_piece
        move_san = self.board.san(move)
        is_capture = self.board.is_capture(move)

        # ── Handle captures BEFORE push ──────────────────────────
        captured_info = None
        if self.board.is_en_passant(move):
            ep_rank = 4 if self.active_color == chess.WHITE else 3
            ep_sq = chess.square(chess.square_file(move.to_square), ep_rank)
            captured_info = self.sq_to_piece.pop(ep_sq, None)
        elif is_capture:
            captured_info = self.sq_to_piece.pop(move.to_square, None)

        if captured_info is not None:
            captured_info.captured = True

        # ── Handle castling ──────────────────────────────────────
        if self.board.is_castling(move):
            if self.board.is_kingside_castling(move):
                rook_from = chess.H1 if self.active_color == chess.WHITE else chess.H8
                rook_to = chess.F1 if self.active_color == chess.WHITE else chess.F8
            else:
                rook_from = chess.A1 if self.active_color == chess.WHITE else chess.A8
                rook_to = chess.D1 if self.active_color == chess.WHITE else chess.D8
            rook_info = self.sq_to_piece.pop(rook_from, None)
            if rook_info is not None:
                rook_info.current_square = rook_to
                self.sq_to_piece[rook_to] = rook_info

        # ── Move the piece ───────────────────────────────────────
        from_sq = move.from_square
        to_sq = move.to_square
        if from_sq in self.sq_to_piece:
            mover = self.sq_to_piece.pop(from_sq)
            mover.current_square = to_sq
            if move.promotion is not None:
                mover.piece_type = move.promotion
            self.sq_to_piece[to_sq] = mover

        self.board.push(move)
        self.ply += 1
        is_check = self.board.is_check()

        move_info = {
            "uci": move.uci(),
            "san": move_san,
            "fen": self.board.fen(),
            "is_capture": is_capture,
            "captured_type": captured_info.piece_type if captured_info else None,
            "is_check": is_check,
            "move_number": (self.ply + 1) // 2,
            "active_color": self.color_str(self.board.turn),
            "moved_by_type": selected.original_piece_type,
            "moved_by_type_name": PIECE_TYPE_NAMES.get(selected.original_piece_type, "?"),
            "from_square": chess.square_name(from_sq),
            "to_square": chess.square_name(to_sq),
        }
        self.move_history.append(move_info)

        # ── Check eliminations ───────────────────────────────────
        eliminated = self._check_new_eliminations()

        # ── Check game over ──────────────────────────────────────
        if self.board.is_game_over() or self.ply >= MAX_PLIES:
            self.phase = "game_over"
            result = self._get_result()
            return move_info, eliminated, result

        # Next turn
        self.active_color = self.board.turn
        return move_info, eliminated, None

    def _check_new_eliminations(self) -> List[dict]:
        """Check for newly eliminated players. Returns list of elimination info."""
        newly_eliminated = []
        for color in (chess.WHITE, chess.BLACK):
            for pt, sid in self.assignments[color].items():
                if sid is None or sid in self.eliminated_sids:
                    continue
                pieces = [p for p in self.side_pieces(color)
                          if p.original_piece_type == pt]
                if all(p.captured for p in pieces):
                    self.eliminated_sids.add(sid)
                    newly_eliminated.append({
                        "sid": sid,
                        "piece_type": pt,
                        "type_name": PIECE_TYPE_NAMES.get(pt, "?"),
                        "color": self.color_str(color),
                    })
        return newly_eliminated

    def _get_result(self) -> dict:
        if self.board.is_checkmate():
            if self.board.turn == chess.BLACK:
                return {"result": "1-0", "reason": "checkmate", "winner": "white"}
            else:
                return {"result": "0-1", "reason": "checkmate", "winner": "black"}
        elif self.board.is_stalemate():
            return {"result": "1/2-1/2", "reason": "stalemate", "winner": None}
        elif self.ply >= MAX_PLIES:
            return {"result": "1/2-1/2", "reason": "move limit", "winner": None}
        else:
            return {"result": self.board.result(), "reason": "draw", "winner": None}
