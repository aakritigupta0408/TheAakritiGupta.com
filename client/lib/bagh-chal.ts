export type PieceType = "tiger" | "goat" | null;
export type Player = "tiger" | "goat";
export type GamePhase = "placement" | "movement";

export interface Position {
  row: number;
  col: number;
}

export interface BaghChalState {
  board: PieceType[][];
  currentPlayer: Player;
  phase: GamePhase;
  goatsPlaced: number;
  goatsCaptured: number;
  selectedPosition: Position | null;
  gameOver: boolean;
  winner: Player | null;
}

export interface Move {
  from: Position;
  to: Position;
  captured?: Position;
}

// Initialize 5x5 board with tigers on corners
export const initializeBaghChal = (): BaghChalState => {
  const board: PieceType[][] = Array(5)
    .fill(null)
    .map(() => Array(5).fill(null));

  // Place tigers on corners
  board[0][0] = "tiger";
  board[0][4] = "tiger";
  board[4][0] = "tiger";
  board[4][4] = "tiger";

  return {
    board,
    currentPlayer: "goat",
    phase: "placement",
    goatsPlaced: 0,
    goatsCaptured: 0,
    selectedPosition: null,
    gameOver: false,
    winner: null,
  };
};

// Get valid positions (intersections) on the board
export const isValidPosition = (row: number, col: number): boolean => {
  return row >= 0 && row < 5 && col >= 0 && col < 5;
};

// Get adjacent positions connected by lines
export const getAdjacentPositions = (pos: Position): Position[] => {
  const { row, col } = pos;
  const adjacent: Position[] = [];

  // Horizontal and vertical connections
  const directions = [
    [-1, 0],
    [1, 0],
    [0, -1],
    [0, 1], // up, down, left, right
  ];

  directions.forEach(([dr, dc]) => {
    const newRow = row + dr;
    const newCol = col + dc;
    if (isValidPosition(newRow, newCol)) {
      adjacent.push({ row: newRow, col: newCol });
    }
  });

  // Diagonal connections (only on specific intersections)
  const diagonalConnections = [
    [0, 0],
    [0, 2],
    [0, 4],
    [1, 1],
    [1, 3],
    [2, 0],
    [2, 2],
    [2, 4],
    [3, 1],
    [3, 3],
    [4, 0],
    [4, 2],
    [4, 4],
  ];

  const isDiagonalIntersection = diagonalConnections.some(
    ([r, c]) => r === row && c === col,
  );

  if (isDiagonalIntersection) {
    const diagonals = [
      [-1, -1],
      [-1, 1],
      [1, -1],
      [1, 1], // diagonals
    ];

    diagonals.forEach(([dr, dc]) => {
      const newRow = row + dr;
      const newCol = col + dc;
      if (isValidPosition(newRow, newCol)) {
        // Check if diagonal connection exists
        const isDiagonalTarget = diagonalConnections.some(
          ([r, c]) => r === newRow && c === newCol,
        );
        if (isDiagonalTarget) {
          adjacent.push({ row: newRow, col: newCol });
        }
      }
    });
  }

  return adjacent;
};

// Check if a move is valid
export const isValidMove = (
  state: BaghChalState,
  from: Position,
  to: Position,
): boolean => {
  if (!isValidPosition(to.row, to.col)) return false;
  if (state.board[to.row][to.col] !== null) return false;

  const piece = state.board[from.row][from.col];
  if (!piece) return false;

  if (state.phase === "placement") {
    // During placement, only goats can be placed
    return false;
  }

  // During movement phase
  const adjacent = getAdjacentPositions(from);
  return adjacent.some((pos) => pos.row === to.row && pos.col === to.col);
};

// Check if a tiger can capture a goat
export const canCapture = (
  state: BaghChalState,
  from: Position,
  over: Position,
  to: Position,
): boolean => {
  if (state.board[from.row][from.col] !== "tiger") return false;
  if (state.board[over.row][over.col] !== "goat") return false;
  if (state.board[to.row][to.col] !== null) return false;

  // Check if positions are in a straight line
  const dx = to.col - from.col;
  const dy = to.row - from.row;
  const overDx = over.col - from.col;
  const overDy = over.row - from.row;

  // The "over" position should be exactly halfway
  return overDx * 2 === dx && overDy * 2 === dy;
};

// Get all valid moves for a piece
export const getValidMoves = (
  state: BaghChalState,
  from: Position,
): Position[] => {
  const moves: Position[] = [];
  const piece = state.board[from.row][from.col];

  if (!piece) return moves;

  if (state.phase === "movement") {
    const adjacent = getAdjacentPositions(from);

    adjacent.forEach((to) => {
      if (state.board[to.row][to.col] === null) {
        moves.push(to);
      }
    });

    // For tigers, check captures
    if (piece === "tiger") {
      adjacent.forEach((over) => {
        if (state.board[over.row][over.col] === "goat") {
          const captureAdjacentPositions = getAdjacentPositions(over);
          captureAdjacentPositions.forEach((to) => {
            if (canCapture(state, from, over, to)) {
              moves.push(to);
            }
          });
        }
      });
    }
  }

  return moves;
};

// Check if tigers can still move
export const canTigersMove = (state: BaghChalState): boolean => {
  for (let row = 0; row < 5; row++) {
    for (let col = 0; col < 5; col++) {
      if (state.board[row][col] === "tiger") {
        const moves = getValidMoves(state, { row, col });
        if (moves.length > 0) return true;
      }
    }
  }
  return false;
};

// Check game over conditions
export const checkGameOver = (
  state: BaghChalState,
): { gameOver: boolean; winner: Player | null } => {
  // Tigers win if they capture 5 goats
  if (state.goatsCaptured >= 5) {
    return { gameOver: true, winner: "tiger" };
  }

  // Goats win if all tigers are trapped (only in movement phase)
  if (state.phase === "movement" && state.goatsPlaced === 20) {
    if (!canTigersMove(state)) {
      return { gameOver: true, winner: "goat" };
    }
  }

  return { gameOver: false, winner: null };
};

// Make a move
export const makeMove = (
  state: BaghChalState,
  from: Position,
  to: Position,
): BaghChalState => {
  const newState = { ...state };
  newState.board = state.board.map((row) => [...row]);

  if (state.phase === "placement") {
    // Place a goat
    if (state.board[to.row][to.col] === null) {
      newState.board[to.row][to.col] = "goat";
      newState.goatsPlaced++;

      if (newState.goatsPlaced === 20) {
        newState.phase = "movement";
      }

      newState.currentPlayer =
        newState.currentPlayer === "goat" ? "tiger" : "goat";
    }
  } else {
    // Movement phase
    const piece = state.board[from.row][from.col];
    newState.board[from.row][from.col] = null;
    newState.board[to.row][to.col] = piece;

    // Check for captures (tigers only)
    if (piece === "tiger") {
      const dx = to.col - from.col;
      const dy = to.row - from.row;

      if (Math.abs(dx) === 2 || Math.abs(dy) === 2) {
        const overRow = from.row + dy / 2;
        const overCol = from.col + dx / 2;

        if (state.board[overRow][overCol] === "goat") {
          newState.board[overRow][overCol] = null;
          newState.goatsCaptured++;
        }
      }
    }

    newState.currentPlayer =
      newState.currentPlayer === "goat" ? "tiger" : "goat";
  }

  newState.selectedPosition = null;
  const gameResult = checkGameOver(newState);
  newState.gameOver = gameResult.gameOver;
  newState.winner = gameResult.winner;

  return newState;
};

// Get all possible tiger moves
const getAllPossibleTigerMoves = (state: BaghChalState): Move[] => {
  const moves: Move[] = [];

  for (let row = 0; row < 5; row++) {
    for (let col = 0; col < 5; col++) {
      if (state.board[row][col] === "tiger") {
        const validMoves = getValidMoves(state, { row, col });
        for (const move of validMoves) {
          moves.push({ from: { row, col }, to: move });
        }
      }
    }
  }

  return moves;
};

// Get all possible goat moves
const getAllPossibleGoatMoves = (state: BaghChalState): Move[] => {
  const moves: Move[] = [];

  if (state.phase === "placement") {
    // During placement, goats can be placed on any empty spot
    for (let row = 0; row < 5; row++) {
      for (let col = 0; col < 5; col++) {
        if (state.board[row][col] === null) {
          moves.push({ from: { row: 0, col: 0 }, to: { row, col } });
        }
      }
    }
  } else {
    // During movement phase, get all goat moves
    for (let row = 0; row < 5; row++) {
      for (let col = 0; col < 5; col++) {
        if (state.board[row][col] === "goat") {
          const validMoves = getValidMoves(state, { row, col });
          for (const move of validMoves) {
            moves.push({ from: { row, col }, to: move });
          }
        }
      }
    }
  }

  return moves;
};

// Check if game is over
const isGameOver = (state: BaghChalState): boolean => {
  const result = checkGameOver(state);
  return result.gameOver;
};

// Apply move and return new state
const applyMove = (state: BaghChalState, move: Move): BaghChalState => {
  return makeMove(state, move.from, move.to);
};

// AI Evaluation function
const evaluatePosition = (state: BaghChalState): number => {
  let score = 0;

  // +10 for each goat captured
  score += state.goatsCaptured * 10;

  // -2 for each tiger unable to move
  for (let row = 0; row < 5; row++) {
    for (let col = 0; col < 5; col++) {
      if (state.board[row][col] === "tiger") {
        const moves = getValidMoves(state, { row, col });
        if (moves.length === 0) {
          score -= 2;
        }
      }
    }
  }

  // Additional evaluation factors
  if (state.goatsCaptured >= 5) {
    score += 1000; // Tiger wins
  }

  if (
    state.phase === "movement" &&
    state.goatsPlaced === 20 &&
    !canTigersMove(state)
  ) {
    score -= 1000; // Goats win
  }

  return score;
};

// Improved Minimax algorithm for AI (based on provided specification)
export const minimax = (
  state: BaghChalState,
  depth: number,
  maximizing: boolean,
): { score: number; move?: Move } => {
  if (depth === 0 || isGameOver(state)) {
    return { score: evaluatePosition(state) };
  }

  if (maximizing) {
    // Tiger's turn
    let maxEval = -Infinity;
    let bestMove: Move | undefined;
    const tigerMoves = getAllPossibleTigerMoves(state);

    for (const move of tigerMoves) {
      const newState = applyMove(state, move);
      const evaluation = minimax(newState, depth - 1, false);

      if (evaluation.score > maxEval) {
        maxEval = evaluation.score;
        bestMove = move;
      }
    }

    return { score: maxEval, move: bestMove };
  } else {
    // Goat's turn
    let minEval = Infinity;
    let bestMove: Move | undefined;
    const goatMoves = getAllPossibleGoatMoves(state);

    for (const move of goatMoves) {
      const newState = applyMove(state, move);
      const evaluation = minimax(newState, depth - 1, true);

      if (evaluation.score < minEval) {
        minEval = evaluation.score;
        bestMove = move;
      }
    }

    return { score: minEval, move: bestMove };
  }
};

// Get AI move for tigers
export const getAIMove = (state: BaghChalState): Move | null => {
  if (state.currentPlayer !== "tiger" || state.gameOver) return null;

  // Use depth 3 for good performance vs quality balance
  const { move } = minimax(state, 3, true);
  return move || null;
};

// Check if a tiger is blocked (no valid moves)
const isTigerBlocked = (state: BaghChalState, tigerPos: Position): boolean => {
  const validMoves = getValidMoves(state, tigerPos);
  return validMoves.length === 0;
};

// Count tiger mobility (number of available moves)
const getTigerMobility = (state: BaghChalState): number => {
  let mobility = 0;
  for (let row = 0; row < 5; row++) {
    for (let col = 0; col < 5; col++) {
      if (state.board[row][col] === "tiger") {
        mobility += getValidMoves(state, { row, col }).length;
      }
    }
  }
  return mobility;
};

// Count blocked tigers
const getBlockedTigers = (state: BaghChalState): number => {
  let blockedCount = 0;
  for (let row = 0; row < 5; row++) {
    for (let col = 0; col < 5; col++) {
      if (state.board[row][col] === "tiger") {
        if (isTigerBlocked(state, { row, col })) {
          blockedCount++;
        }
      }
    }
  }
  return blockedCount;
};

// Evaluation function specifically for goats
const evaluateForGoats = (state: BaghChalState): number => {
  const blockedTigers = getBlockedTigers(state);
  const tigerMobility = getTigerMobility(state);

  // Goat strategy: maximize blocked tigers, minimize captures, reduce tiger mobility
  let score = 0;
  score += blockedTigers * 10; // Reward blocking tigers
  score -= state.goatsCaptured * 5; // Penalty for captured goats
  score -= tigerMobility * 2; // Penalty for tiger mobility

  // Bonus for winning condition
  if (
    blockedTigers === 4 &&
    state.phase === "movement" &&
    state.goatsPlaced === 20
  ) {
    score += 1000; // Goats win
  }

  // Major penalty for losing
  if (state.goatsCaptured >= 5) {
    score -= 1000; // Tigers win
  }

  return score;
};

// Minimax specifically optimized for goat play
const minimaxForGoat = (
  state: BaghChalState,
  depth: number,
  maximizing: boolean,
): { score: number; move?: Move } => {
  if (depth === 0 || state.goatsCaptured >= 5 || isGameOver(state)) {
    return { score: evaluateForGoats(state) };
  }

  if (maximizing) {
    // Goat's turn - maximize goat advantage
    let maxEval = -Infinity;
    let bestMove: Move | undefined;
    const goatMoves = getAllPossibleGoatMoves(state);

    for (const move of goatMoves) {
      const newState = applyMove(state, move);
      const evaluation = minimaxForGoat(newState, depth - 1, false);

      if (evaluation.score > maxEval) {
        maxEval = evaluation.score;
        bestMove = move;
      }
    }

    return { score: maxEval, move: bestMove };
  } else {
    // Tiger's turn - minimize goat advantage (maximize tiger advantage)
    let minEval = Infinity;
    let bestMove: Move | undefined;
    const tigerMoves = getAllPossibleTigerMoves(state);

    for (const move of tigerMoves) {
      const newState = applyMove(state, move);
      const evaluation = minimaxForGoat(newState, depth - 1, true);

      if (evaluation.score < minEval) {
        minEval = evaluation.score;
        bestMove = move;
      }
    }

    return { score: minEval, move: bestMove };
  }
};

// Get AI move for goats using strategic minimax
export const getGoatAIMove = (state: BaghChalState): Move | null => {
  if (state.currentPlayer !== "goat" || state.gameOver) return null;

  // Use depth 2 for strategic goat play
  const { move } = minimaxForGoat(state, 2, true);
  return move || null;
};

// Get AI move for current player
export const getAIMoveBoth = (state: BaghChalState): Move | null => {
  if (state.gameOver) return null;

  if (state.currentPlayer === "tiger") {
    return getAIMove(state);
  } else {
    return getGoatAIMove(state);
  }
};
