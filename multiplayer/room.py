"""Room and lobby management."""

import random
import string
import chess
from dataclasses import dataclass, field
from typing import Dict, List, Optional

ASSIGNABLE_TYPES = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]

PIECE_TYPE_NAMES = {
    chess.PAWN: "Pawns",
    chess.KNIGHT: "Knights",
    chess.BISHOP: "Bishops",
    chess.ROOK: "Rooks",
    chess.QUEEN: "Queen",
    chess.KING: "King",
}


@dataclass
class Player:
    sid: str
    name: str
    side: Optional[str] = None       # "white", "black", or None
    piece_type: Optional[int] = None  # Assigned at game start
    is_host: bool = False
    eliminated: bool = False


@dataclass
class Room:
    code: str
    players: Dict[str, Player] = field(default_factory=dict)
    game: object = None
    state: str = "lobby"  # lobby | playing | finished

    def players_on_side(self, side: str) -> List[Player]:
        return [p for p in self.players.values() if p.side == side]

    def get_room_state(self) -> dict:
        return {
            "code": self.code,
            "state": self.state,
            "players": [
                {
                    "sid": p.sid,
                    "name": p.name,
                    "side": p.side,
                    "piece_type": p.piece_type,
                    "type_name": PIECE_TYPE_NAMES.get(p.piece_type, None),
                    "is_host": p.is_host,
                    "eliminated": p.eliminated,
                }
                for p in self.players.values()
            ],
        }

    def assign_piece_types(self):
        """Randomly assign piece types to players on each side.

        Returns {color_str: {piece_type: sid_or_None}} for both sides.
        King is always None (CPU).
        """
        assignments = {}
        for side_str, color in [("white", chess.WHITE), ("black", chess.BLACK)]:
            side_players = self.players_on_side(side_str)
            random.shuffle(side_players)
            types_shuffled = ASSIGNABLE_TYPES[:]
            random.shuffle(types_shuffled)

            mapping = {}
            for i, pt in enumerate(types_shuffled):
                if i < len(side_players):
                    mapping[pt] = side_players[i].sid
                    side_players[i].piece_type = pt
                else:
                    mapping[pt] = None  # CPU
            mapping[chess.KING] = None  # King is always CPU
            assignments[color] = mapping
        return assignments


# Global rooms registry
rooms: Dict[str, Room] = {}


def generate_room_code() -> str:
    while True:
        code = "".join(random.choices(string.ascii_uppercase + string.digits, k=4))
        if code not in rooms:
            return code


def create_room(host_sid: str, host_name: str) -> Room:
    code = generate_room_code()
    player = Player(sid=host_sid, name=host_name, is_host=True)
    room = Room(code=code, players={host_sid: player})
    rooms[code] = room
    return room


def join_room(code: str, sid: str, name: str):
    code = code.upper().strip()
    if code not in rooms:
        return None, "Room not found"
    room = rooms[code]
    if room.state != "lobby":
        return None, "Game already in progress"
    if sid in room.players:
        return room, None
    player = Player(sid=sid, name=name)
    room.players[sid] = player
    return room, None


def get_room_by_sid(sid: str) -> Optional[Room]:
    for room in rooms.values():
        if sid in room.players:
            return room
    return None


def remove_player(sid: str):
    room = get_room_by_sid(sid)
    if room is None:
        return None
    if sid in room.players:
        was_host = room.players[sid].is_host
        del room.players[sid]
        # Promote next player to host if needed
        if was_host and room.players:
            next_player = next(iter(room.players.values()))
            next_player.is_host = True
        # Clean up empty rooms
        if not room.players:
            rooms.pop(room.code, None)
            return None
    return room
