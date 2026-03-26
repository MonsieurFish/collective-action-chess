"""Flask + Socket.IO server for Collective Action Chess."""

import math
import chess
from flask import Flask, request, send_from_directory
from flask_socketio import SocketIO, emit, join_room as sio_join, leave_room as sio_leave

from room import (
    Room, create_room, join_room, get_room_by_sid, remove_player,
    PIECE_TYPE_NAMES,
)
from game import Game

app = Flask(__name__, static_folder="static", static_url_path="/static")
app.config["SECRET_KEY"] = "collective-action-chess"
socketio = SocketIO(app, cors_allowed_origins="*")


# ── Static Routes ────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory("static", "index.html")


# ── Lobby Events ─────────────────────────────────────────────────────

@socketio.on("create_room")
def handle_create_room(data):
    name = data.get("name", "").strip()
    if not name:
        emit("error", {"message": "Name is required"})
        return
    room = create_room(request.sid, name)
    sio_join(room.code)
    emit("room_created", {"room_code": room.code})
    emit("room_state", room.get_room_state(), to=room.code)


@socketio.on("join_room")
def handle_join_room(data):
    name = data.get("name", "").strip()
    code = data.get("room_code", "").strip()
    if not name:
        emit("error", {"message": "Name is required"})
        return
    room, err = join_room(code, request.sid, name)
    if err:
        emit("join_error", {"message": err})
        return
    sio_join(room.code)
    emit("room_joined", {"room_code": room.code})
    emit("room_state", room.get_room_state(), to=room.code)


@socketio.on("pick_side")
def handle_pick_side(data):
    room = get_room_by_sid(request.sid)
    if not room or room.state != "lobby":
        return
    side = data.get("side")  # "white", "black", or None
    if side not in ("white", "black", None):
        return
    room.players[request.sid].side = side
    emit("room_state", room.get_room_state(), to=room.code)


@socketio.on("start_game")
def handle_start_game(data=None):
    room = get_room_by_sid(request.sid)
    if not room:
        emit("start_error", {"message": "Room not found"})
        return
    player = room.players.get(request.sid)
    if not player or not player.is_host:
        emit("start_error", {"message": "Only the host can start the game"})
        return
    if room.state != "lobby":
        emit("start_error", {"message": "Game already started"})
        return

    # Need at least 1 player on each side
    white_players = room.players_on_side("white")
    black_players = room.players_on_side("black")
    if not white_players or not black_players:
        emit("start_error", {"message": "Need at least 1 player on each side"})
        return

    # Assign piece types randomly
    assignments = room.assign_piece_types()
    room.state = "playing"

    # Create the game
    game = Game(assignments[chess.WHITE], assignments[chess.BLACK])
    room.game = game

    # Build per-player assignment info
    player_assignments = {}
    for sid, p in room.players.items():
        if p.piece_type is not None:
            color_str = p.side
            player_assignments[sid] = {
                "side": color_str,
                "piece_type": p.piece_type,
                "type_name": PIECE_TYPE_NAMES.get(p.piece_type, "?"),
            }

    # Broadcast game start to everyone in the room
    emit("game_started", {
        "fen": game.board.fen(),
        "assignments": player_assignments,
        "active_color": "white",
    }, to=room.code)

    # Start first voting phase
    _start_voting_phase(room)


# ── Voting Events ────────────────────────────────────────────────────

@socketio.on("submit_vote")
def handle_submit_vote(data):
    room = get_room_by_sid(request.sid)
    if not room or not room.game:
        return
    game = room.game

    # Look up this player's piece type
    assignment = game.sid_to_assignment.get(request.sid)
    if not assignment:
        emit("vote_ack", {"status": "error", "message": "Not a player"})
        return
    color, piece_type = assignment

    # Must be active side
    if color != game.active_color:
        emit("vote_ack", {"status": "error", "message": "Not your side's turn"})
        return

    slot_id = data.get("slot_id")
    if slot_id is None:
        emit("vote_ack", {"status": "error", "message": "No piece selected"})
        return

    ok, msg = game.submit_vote(piece_type, slot_id)
    emit("vote_ack", {"status": "ok" if ok else "error", "message": msg})

    if ok:
        # Broadcast progress (count only, no details)
        received, total = game.votes_progress()
        emit("votes_waiting", {
            "voted_count": received,
            "total_voters": total,
        }, to=room.code)

        if game.all_votes_in():
            _tally_and_select(room)


# ── Moving Events ────────────────────────────────────────────────────

@socketio.on("submit_move")
def handle_submit_move(data):
    room = get_room_by_sid(request.sid)
    if not room or not room.game:
        return
    game = room.game

    if game.phase != "moving":
        emit("move_ack", {"status": "error", "message": "Not in move phase"})
        return

    mover_sid = game.get_mover_sid()
    if mover_sid != request.sid:
        emit("move_ack", {"status": "error", "message": "Not your turn to move"})
        return

    uci = data.get("uci", "")
    result = game.execute_move(uci)
    if result[0] is None:
        emit("move_ack", {"status": "error", "message": result[1]})
        return

    emit("move_ack", {"status": "ok", "message": "Move executed"})
    _broadcast_move_result(room, result)


# ── Game Flow Helpers ────────────────────────────────────────────────

def _start_voting_phase(room: Room):
    game = room.game
    cpu_votes = game.start_voting()

    active_str = game.color_str(game.active_color)
    living = game.get_living_pieces_info(game.active_color)

    # Tell each player in the room about the voting phase
    for sid, player in room.players.items():
        is_active = (player.side == active_str
                     and player.piece_type is not None
                     and not player.eliminated
                     and player.piece_type in game.living_types(game.active_color))
        my_weight = None
        if is_active:
            count = game.type_count(game.active_color, player.piece_type)
            my_weight = round(math.log2(count + 2), 2)

        emit("vote_phase", {
            "active_color": active_str,
            "fen": game.board.fen(),
            "your_turn": is_active,
            "living_pieces": living,
            "your_weight": my_weight,
            "your_type": PIECE_TYPE_NAMES.get(player.piece_type) if player.piece_type else None,
        }, to=sid)

    # If all votes already in (all CPU on that side), tally immediately
    if game.all_votes_in():
        _tally_and_select(room)


def _tally_and_select(room: Room):
    game = room.game
    selected = game.tally_votes()
    if selected is None:
        # No piece can move — shouldn't normally happen, game would be over
        return

    # Broadcast which piece was selected (no vote details)
    emit("piece_selected", {
        "slot_id": selected.slot_id,
        "piece_type": selected.original_piece_type,
        "type_name": PIECE_TYPE_NAMES.get(selected.original_piece_type, "?"),
        "square": chess.square_name(selected.current_square),
    }, to=room.code)

    _start_move_phase(room)


def _start_move_phase(room: Room):
    game = room.game
    mover_sid = game.get_mover_sid()
    selected = game.selected_piece

    if mover_sid is None:
        # CPU move
        result = game.make_cpu_move()
        if result[0] is None:
            return
        _broadcast_move_result(room, result)
        return

    # Human must move — find their name
    mover_name = room.players[mover_sid].name if mover_sid in room.players else "CPU"
    legal = game.get_legal_moves_uci()

    # Tell everyone about the move phase
    for sid in room.players:
        is_mover = (sid == mover_sid)
        emit("move_phase", {
            "mover_name": mover_name,
            "piece_type": selected.original_piece_type,
            "type_name": PIECE_TYPE_NAMES.get(selected.original_piece_type, "?"),
            "square": chess.square_name(selected.current_square),
            "legal_moves": legal if is_mover else [],
            "is_you": is_mover,
        }, to=sid)


def _broadcast_move_result(room: Room, result):
    game = room.game
    move_info, eliminated, game_result = result

    # Broadcast the move
    emit("move_made", move_info, to=room.code)

    # Handle eliminations
    for elim in eliminated:
        sid = elim["sid"]
        if sid in room.players:
            room.players[sid].eliminated = True
        emit("player_eliminated", elim, to=room.code)

    # Game over?
    if game_result is not None:
        emit("game_over", {
            **game_result,
            "fen": game.board.fen(),
        }, to=room.code)
        room.state = "finished"
        return

    # Next voting phase
    _start_voting_phase(room)


# ── Connection Events ────────────────────────────────────────────────

@socketio.on("connect")
def handle_connect():
    pass


@socketio.on("disconnect")
def handle_disconnect():
    room = get_room_by_sid(request.sid)
    if not room:
        return

    if room.state == "playing" and room.game:
        # Player disconnected during game — their type becomes CPU
        game = room.game
        assignment = game.sid_to_assignment.get(request.sid)
        if assignment:
            color, pt = assignment
            game.assignments[color][pt] = None
            del game.sid_to_assignment[request.sid]

            # If they had a pending vote, submit CPU vote
            if pt in game.human_votes_needed:
                from cpu import cpu_vote
                living = game.living_pieces(game.active_color)
                slot = cpu_vote(living, game.board)
                game.submit_vote(pt, slot)
                game.human_votes_needed.discard(pt)

                if game.all_votes_in():
                    _tally_and_select(room)

            # If they were supposed to move, CPU moves
            if game.phase == "moving" and game.get_mover_sid() is None:
                result = game.make_cpu_move()
                if result[0] is not None:
                    _broadcast_move_result(room, result)

    # Remove from room
    updated_room = remove_player(request.sid)
    if updated_room:
        emit("room_state", updated_room.get_room_state(), to=updated_room.code)


# ── Entry Point ──────────────────────────────────────────────────────

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5001))
    debug = os.environ.get("RENDER") is None  # Debug only locally
    print(f"Starting Collective Action Chess server on port {port}")
    socketio.run(app, host="0.0.0.0", port=port, debug=debug, allow_unsafe_werkzeug=True)
