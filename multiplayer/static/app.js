/* Collective Action Chess — Client */

const socket = io();

// ── State ────────────────────────────────────────────────────────────
const state = {
  view: "landing",
  roomCode: null,
  mySid: null,
  myName: null,
  isHost: false,
  side: null,          // "white" or "black"
  pieceType: null,     // chess piece type int (assigned by server)
  typeName: null,
  board: null,         // chessboard.js instance
  gamePhase: null,     // "voting" | "moving" | "game_over"
  activeColor: null,
  selectedVote: null,  // slot_id
  isMyVoteTurn: false,
  isMovePhase: false,
  isMyMove: false,
  legalMoves: [],      // [{uci, san}]
  selectedSquare: null,
  lastMove: null,      // {from, to}
  pendingPromotion: null, // {from, to} waiting for promotion choice
  players: [],         // from room_state / game_started
  assignments: {},     // sid -> {side, piece_type, type_name}
};


// ── View Switching ───────────────────────────────────────────────────
function showView(name) {
  document.querySelectorAll(".view").forEach(v => v.classList.remove("active"));
  const el = document.getElementById(name + "-view");
  if (el) el.classList.add("active");
  state.view = name;
}


// ── Landing ──────────────────────────────────────────────────────────
document.getElementById("create-btn").addEventListener("click", () => {
  const name = document.getElementById("player-name").value.trim();
  if (!name) return showError("landing-error", "Enter your name");
  state.myName = name;
  socket.emit("create_room", { name });
});

document.getElementById("join-btn").addEventListener("click", () => {
  const name = document.getElementById("player-name").value.trim();
  const code = document.getElementById("join-code").value.trim().toUpperCase();
  if (!name) return showError("landing-error", "Enter your name");
  if (!code) return showError("landing-error", "Enter a room code");
  state.myName = name;
  socket.emit("join_room", { name, room_code: code });
});

// Enter key support
document.getElementById("player-name").addEventListener("keydown", e => {
  if (e.key === "Enter") document.getElementById("create-btn").click();
});
document.getElementById("join-code").addEventListener("keydown", e => {
  if (e.key === "Enter") document.getElementById("join-btn").click();
});


// ── Lobby ────────────────────────────────────────────────────────────
document.querySelectorAll(".btn-side").forEach(btn => {
  btn.addEventListener("click", () => {
    const side = btn.dataset.side;
    socket.emit("pick_side", { side });
  });
});

document.getElementById("start-btn").addEventListener("click", () => {
  socket.emit("start_game");
});


// ── Socket Events: Lobby ─────────────────────────────────────────────
socket.on("connect", () => {
  state.mySid = socket.id;
});

socket.on("room_created", data => {
  state.roomCode = data.room_code;
  state.isHost = true;
  showView("lobby");
  document.getElementById("room-code").textContent = data.room_code;
});

socket.on("room_joined", data => {
  state.roomCode = data.room_code;
  showView("lobby");
  document.getElementById("room-code").textContent = data.room_code;
});

socket.on("join_error", data => {
  showError("landing-error", data.message);
});

socket.on("start_error", data => {
  showError("lobby-error", data.message);
});

socket.on("error", data => {
  showError("landing-error", data.message);
});

socket.on("room_state", data => {
  state.players = data.players;
  renderLobby(data);
});


function renderLobby(data) {
  const whitePlayers = data.players.filter(p => p.side === "white");
  const blackPlayers = data.players.filter(p => p.side === "black");
  const spectators = data.players.filter(p => !p.side);

  renderPlayerList("white-players", whitePlayers);
  renderPlayerList("black-players", blackPlayers);

  // Highlight current side buttons
  const myPlayer = data.players.find(p => p.sid === state.mySid);
  document.querySelectorAll(".btn-side").forEach(btn => {
    btn.classList.toggle("active-side", myPlayer && myPlayer.side === btn.dataset.side);
  });

  if (myPlayer) {
    state.side = myPlayer.side;
    state.isHost = myPlayer.is_host;
  }

  // Show start button for host
  const startBtn = document.getElementById("start-btn");
  startBtn.style.display = state.isHost ? "inline-block" : "none";

  // Spectators
  const specDiv = document.getElementById("spectators-lobby");
  if (spectators.length) {
    specDiv.textContent = "Unassigned: " + spectators.map(p => p.name).join(", ");
  } else {
    specDiv.textContent = "";
  }
}

function renderPlayerList(containerId, players) {
  const container = document.getElementById(containerId);
  container.innerHTML = "";
  players.forEach(p => {
    const div = document.createElement("div");
    div.className = "player-entry" + (p.sid === state.mySid ? " is-you" : "");
    let html = p.name;
    if (p.is_host) html += ' <span class="host-badge">HOST</span>';
    if (p.sid === state.mySid) html += " (you)";
    div.innerHTML = html;
    container.appendChild(div);
  });
  if (!players.length) {
    container.innerHTML = '<div class="player-entry" style="color:var(--text-muted)">No players</div>';
  }
}


// ── Socket Events: Game Start ────────────────────────────────────────
socket.on("game_started", data => {
  state.assignments = data.assignments;
  state.activeColor = data.active_color;

  // Find my assignment
  const myAssign = data.assignments[state.mySid];
  if (myAssign) {
    state.side = myAssign.side;
    state.pieceType = myAssign.piece_type;
    state.typeName = myAssign.type_name;
  }

  showView("game");
  initBoard(data.fen, state.side || "white");
  updateInfoBar();
  renderGamePlayers();
});


// ── Board ────────────────────────────────────────────────────────────
function initBoard(fen, orientation) {
  const config = {
    draggable: true,
    position: fen,
    orientation: orientation,
    pieceTheme: "https://chessboardjs.com/img/chesspieces/wikipedia/{piece}.png",
    onDragStart: onDragStart,
    onDrop: onDrop,
    onMouseoverSquare: onMouseoverSquare,
    onMouseoutSquare: onMouseoutSquare,
  };
  state.board = Chessboard("board", config);
  $(window).on("resize", () => state.board.resize());
}

function onDragStart(source, piece) {
  // Only allow dragging during move phase when it's your move
  if (!state.isMyMove) return false;
  // Only allow dragging the selected piece
  if (source !== state.selectedSquare) return false;
  return true;
}

function onDrop(source, target, piece) {
  if (!state.isMyMove) return "snapback";

  // Find matching legal move
  const matchingMoves = state.legalMoves.filter(m => {
    return m.uci.substring(0, 2) === source && m.uci.substring(2, 4) === target;
  });

  if (matchingMoves.length === 0) return "snapback";

  // Check for promotion
  if (matchingMoves.length > 1 || (matchingMoves[0].uci.length === 5)) {
    // Promotion — show modal
    state.pendingPromotion = { from: source, to: target };
    showPromotionModal();
    return "snapback"; // Snap back until promotion is chosen
  }

  // Normal move
  submitMove(matchingMoves[0].uci);
}

function onMouseoverSquare(square) {
  if (!state.isMyMove) return;
  // Highlight legal destinations when hovering the selected piece
  if (square === state.selectedSquare) {
    showLegalMoves();
  }
}

function onMouseoutSquare() {
  clearHighlights("highlight-legal");
  clearHighlights("highlight-legal-capture");
}

function submitMove(uci) {
  state.isMyMove = false;
  socket.emit("submit_move", { uci });
  clearAllHighlights();
}

function showLegalMoves() {
  clearHighlights("highlight-legal");
  clearHighlights("highlight-legal-capture");
  state.legalMoves.forEach(m => {
    const target = m.uci.substring(2, 4);
    // Check if target has a piece (capture)
    const pos = state.board.position();
    const cls = pos[target] ? "highlight-legal-capture" : "highlight-legal";
    $(`#board .square-${target}`).addClass(cls);
  });
}

function clearHighlights(cls) {
  $(`#board .${cls}`).removeClass(cls);
}

function clearAllHighlights() {
  clearHighlights("highlight-last-from");
  clearHighlights("highlight-last-to");
  clearHighlights("highlight-selected");
  clearHighlights("highlight-legal");
  clearHighlights("highlight-legal-capture");
}

function highlightLastMove(from, to) {
  clearHighlights("highlight-last-from");
  clearHighlights("highlight-last-to");
  if (from) $(`#board .square-${from}`).addClass("highlight-last-from");
  if (to) $(`#board .square-${to}`).addClass("highlight-last-to");
}

function highlightSelected(square) {
  clearHighlights("highlight-selected");
  if (square) $(`#board .square-${square}`).addClass("highlight-selected");
}


// ── Promotion Modal ──────────────────────────────────────────────────
function showPromotionModal() {
  document.getElementById("promotion-modal").style.display = "flex";
}

document.querySelectorAll(".promo-btn").forEach(btn => {
  btn.addEventListener("click", () => {
    const promo = btn.dataset.piece;
    const { from, to } = state.pendingPromotion;
    const uci = from + to + promo;
    state.pendingPromotion = null;
    document.getElementById("promotion-modal").style.display = "none";
    submitMove(uci);
  });
});


// ── Socket Events: Voting ────────────────────────────────────────────
socket.on("vote_phase", data => {
  state.gamePhase = "voting";
  state.activeColor = data.active_color;
  state.isMyVoteTurn = data.your_turn;
  state.isMyMove = false;
  state.selectedVote = null;

  // Update board position
  if (data.fen && state.board) {
    state.board.position(data.fen, false);
  }

  updateInfoBar();
  hideAllPanels();

  if (data.your_turn) {
    showVotePanel(data);
  } else {
    showWaiting(
      `${capitalize(data.active_color)}'s turn to vote`,
      "Waiting for votes..."
    );
  }
});

socket.on("vote_ack", data => {
  if (data.status === "ok") {
    document.getElementById("vote-status").textContent = "Vote submitted! Waiting for others...";
    document.getElementById("vote-submit-btn").disabled = true;
  } else {
    document.getElementById("vote-status").textContent = "Error: " + data.message;
  }
});

socket.on("votes_waiting", data => {
  const status = document.getElementById("vote-status");
  if (status && state.gamePhase === "voting") {
    status.textContent = `Votes: ${data.voted_count}/${data.total_voters}`;
  }
});

socket.on("piece_selected", data => {
  state.selectedSquare = data.square;
  // Brief highlight of selected piece
  highlightSelected(data.square);

  // Show which piece was selected in the info area
  const phaseInfo = document.getElementById("phase-info");
  phaseInfo.textContent = `Selected: ${data.type_name} on ${data.square}`;
});


function showVotePanel(data) {
  const panel = document.getElementById("vote-panel");
  panel.style.display = "block";

  // Show vote weight
  const weightInfo = document.getElementById("vote-weight-info");
  if (data.your_weight) {
    weightInfo.textContent = `Your type: ${data.your_type} (vote weight: ${data.your_weight})`;
  } else {
    weightInfo.textContent = "";
  }

  // Build vote options
  const list = document.getElementById("vote-list");
  list.innerHTML = "";
  data.living_pieces.forEach(piece => {
    const div = document.createElement("div");
    div.className = "vote-option";
    div.dataset.slotId = piece.slot_id;
    div.innerHTML = `
      <span class="piece-symbol">${piece.symbol}</span>
      <span class="piece-info">
        <span class="piece-square">${piece.square}</span>
        <span class="piece-type-label">${piece.type_name}</span>
      </span>
    `;
    div.addEventListener("click", () => {
      document.querySelectorAll(".vote-option").forEach(o => o.classList.remove("selected"));
      div.classList.add("selected");
      state.selectedVote = piece.slot_id;
      document.getElementById("vote-submit-btn").disabled = false;

      // Highlight the piece on the board
      clearHighlights("highlight-selected");
      highlightSelected(piece.square);
    });
    list.appendChild(div);
  });

  document.getElementById("vote-submit-btn").disabled = true;
  document.getElementById("vote-status").textContent = "";
}

document.getElementById("vote-submit-btn").addEventListener("click", () => {
  if (state.selectedVote === null) return;
  socket.emit("submit_vote", { slot_id: state.selectedVote });
});


// ── Socket Events: Moving ────────────────────────────────────────────
socket.on("move_phase", data => {
  state.gamePhase = "moving";
  state.isMovePhase = true;
  state.isMyMove = data.is_you;
  state.legalMoves = data.legal_moves || [];
  state.selectedSquare = data.square;

  hideAllPanels();
  highlightSelected(data.square);

  if (data.is_you) {
    // It's our turn to move
    const panel = document.getElementById("move-panel");
    panel.style.display = "block";
    document.getElementById("move-panel-title").textContent =
      `Your ${data.type_name} on ${data.square} was chosen!`;
    document.getElementById("move-panel-desc").textContent =
      "Drag the piece or click a destination square.";

    // Also allow click-to-move: show legal moves immediately
    showLegalMoves();
    setupClickToMove();
  } else {
    showWaiting(
      `${data.mover_name} is moving`,
      `${data.type_name} on ${data.square}`
    );
  }
});

function setupClickToMove() {
  // Allow clicking destination squares
  $("#board .square-55d63").off("click.move");  // chessboard.js internal class
  state.legalMoves.forEach(m => {
    const target = m.uci.substring(2, 4);
    $(`#board .square-${target}`).off("click.move").on("click.move", function () {
      // Find matching legal move(s)
      const matching = state.legalMoves.filter(mv =>
        mv.uci.substring(0, 2) === state.selectedSquare &&
        mv.uci.substring(2, 4) === target
      );
      if (matching.length > 1 || (matching[0] && matching[0].uci.length === 5)) {
        state.pendingPromotion = { from: state.selectedSquare, to: target };
        showPromotionModal();
      } else if (matching.length === 1) {
        submitMove(matching[0].uci);
      }
    });
  });
}


// ── Socket Events: Move Made ─────────────────────────────────────────
socket.on("move_made", data => {
  // Clean up click handlers
  $("#board [class*=square-]").off("click.move");

  // Update board
  if (state.board) {
    state.board.position(data.fen, true); // animate
  }

  // Highlight last move
  state.lastMove = { from: data.from_square, to: data.to_square };
  // Delay highlight slightly to let animation finish
  setTimeout(() => {
    clearAllHighlights();
    highlightLastMove(data.from_square, data.to_square);
  }, 300);

  // Add to move history
  appendMoveHistory(data);

  // Update active color
  state.activeColor = data.active_color;
  updateInfoBar();
});


// ── Socket Events: Elimination & Game Over ───────────────────────────
socket.on("player_eliminated", data => {
  const msg = `${data.type_name} (${data.color}) eliminated!`;
  // Update player list
  renderGamePlayers();
  // Flash a message
  const phaseInfo = document.getElementById("phase-info");
  phaseInfo.textContent = msg;
  phaseInfo.style.color = "var(--accent)";
  setTimeout(() => { phaseInfo.style.color = ""; }, 2000);
});

socket.on("game_over", data => {
  state.gamePhase = "game_over";

  // Update board
  if (data.fen && state.board) {
    state.board.position(data.fen, false);
  }

  // Show overlay
  const overlay = document.getElementById("game-over-overlay");
  overlay.style.display = "flex";

  const title = document.getElementById("game-over-title");
  const reason = document.getElementById("game-over-reason");

  if (data.winner) {
    title.textContent = `${capitalize(data.winner)} wins!`;
  } else {
    title.textContent = "Draw!";
  }
  reason.textContent = `${capitalize(data.reason)} — ${data.result}`;
});

document.getElementById("back-to-lobby-btn").addEventListener("click", () => {
  document.getElementById("game-over-overlay").style.display = "none";
  // For now, just reload
  window.location.reload();
});


// ── UI Helpers ───────────────────────────────────────────────────────
function hideAllPanels() {
  document.getElementById("vote-panel").style.display = "none";
  document.getElementById("move-panel").style.display = "none";
  document.getElementById("waiting-panel").style.display = "none";
}

function showWaiting(title, desc) {
  const panel = document.getElementById("waiting-panel");
  panel.style.display = "block";
  document.getElementById("waiting-title").textContent = title;
  document.getElementById("waiting-desc").textContent = desc;
}

function updateInfoBar() {
  const turnInfo = document.getElementById("turn-info");
  const phaseInfo = document.getElementById("phase-info");
  const assignInfo = document.getElementById("assignment-info");

  turnInfo.textContent = state.activeColor
    ? `${capitalize(state.activeColor)}'s turn`
    : "";
  phaseInfo.textContent = state.gamePhase
    ? capitalize(state.gamePhase)
    : "";
  assignInfo.textContent = state.typeName
    ? `You: ${state.typeName} (${capitalize(state.side || "")})`
    : "Spectator";
}

function renderGamePlayers() {
  const container = document.getElementById("game-players");
  container.innerHTML = "";
  for (const [sid, assign] of Object.entries(state.assignments)) {
    const player = state.players.find(p => p.sid === sid);
    const name = player ? player.name : "?";
    const isMe = sid === state.mySid;
    const isElim = player && player.eliminated;

    const div = document.createElement("div");
    div.className = "game-player-entry" + (isElim ? " eliminated" : "");
    div.innerHTML = `
      <span>${name}${isMe ? " (you)" : ""} — ${assign.type_name}</span>
      <span class="player-side side-${assign.side}">${assign.side}</span>
    `;
    container.appendChild(div);
  }
}

function appendMoveHistory(data) {
  const container = document.getElementById("move-history");
  const entry = document.createElement("span");
  entry.className = "move-entry";

  const isWhiteMove = data.active_color === "black"; // move was made before color flipped
  if (isWhiteMove) {
    entry.innerHTML = `<span class="move-number">${data.move_number}.</span> `;
  }

  const cls = data.is_capture ? "move-san move-capture" : "move-san";
  entry.innerHTML += `<span class="${cls}">${data.san}</span> `;

  container.appendChild(entry);
  container.scrollTop = container.scrollHeight;
}

function showError(elementId, msg) {
  const el = document.getElementById(elementId);
  if (el) {
    el.textContent = msg;
    setTimeout(() => { el.textContent = ""; }, 3000);
  }
}

function capitalize(s) {
  return s ? s.charAt(0).toUpperCase() + s.slice(1) : "";
}
