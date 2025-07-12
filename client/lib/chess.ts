export type PieceType =
  | "king"
  | "queen"
  | "rook"
  | "bishop"
  | "knight"
  | "pawn";
export type PieceColor = "white" | "black";

export interface ChessPiece {
  type: PieceType;
  color: PieceColor;
  id: string;
  hasMoved?: boolean;
  story?: {
    title: string;
    content: string;
    isStrength: boolean;
  };
}

export interface BoardSquare {
  piece?: ChessPiece;
  row: number;
  col: number;
}

export interface Move {
  fromRow: number;
  fromCol: number;
  toRow: number;
  toCol: number;
  capturedPiece?: ChessPiece;
}

export const isValidMove = (
  board: BoardSquare[][],
  fromRow: number,
  fromCol: number,
  toRow: number,
  toCol: number,
): boolean => {
  // Basic bounds check
  if (toRow < 0 || toRow > 7 || toCol < 0 || toCol > 7) return false;

  const piece = board[fromRow][fromCol].piece;
  if (!piece) return false;

  const targetSquare = board[toRow][toCol];

  // Can't capture own piece
  if (targetSquare.piece && targetSquare.piece.color === piece.color) {
    return false;
  }

  const rowDiff = Math.abs(toRow - fromRow);
  const colDiff = Math.abs(toCol - fromCol);
  const rowDir = toRow > fromRow ? 1 : toRow < fromRow ? -1 : 0;
  const colDir = toCol > fromCol ? 1 : toCol < fromCol ? -1 : 0;

  switch (piece.type) {
    case "pawn":
      return isValidPawnMove(
        board,
        fromRow,
        fromCol,
        toRow,
        toCol,
        piece.color,
      );

    case "rook":
      return (
        (rowDiff === 0 || colDiff === 0) &&
        isPathClear(board, fromRow, fromCol, toRow, toCol)
      );

    case "bishop":
      return (
        rowDiff === colDiff &&
        isPathClear(board, fromRow, fromCol, toRow, toCol)
      );

    case "queen":
      return (
        (rowDiff === 0 || colDiff === 0 || rowDiff === colDiff) &&
        isPathClear(board, fromRow, fromCol, toRow, toCol)
      );

    case "knight":
      return (
        (rowDiff === 2 && colDiff === 1) || (rowDiff === 1 && colDiff === 2)
      );

    case "king":
      return rowDiff <= 1 && colDiff <= 1;

    default:
      return false;
  }
};

const isValidPawnMove = (
  board: BoardSquare[][],
  fromRow: number,
  fromCol: number,
  toRow: number,
  toCol: number,
  color: PieceColor,
): boolean => {
  const direction = color === "white" ? -1 : 1;
  const startRow = color === "white" ? 6 : 1;
  const rowDiff = toRow - fromRow;
  const colDiff = Math.abs(toCol - fromCol);

  // Forward move
  if (fromCol === toCol) {
    if (board[toRow][toCol].piece) return false; // Can't move forward into piece

    if (rowDiff === direction) return true; // One square forward

    // Two squares forward from starting position
    if (fromRow === startRow && rowDiff === 2 * direction) return true;

    return false;
  }

  // Diagonal capture
  if (colDiff === 1 && rowDiff === direction) {
    return board[toRow][toCol].piece !== undefined;
  }

  return false;
};

const isPathClear = (
  board: BoardSquare[][],
  fromRow: number,
  fromCol: number,
  toRow: number,
  toCol: number,
): boolean => {
  const rowDir = toRow > fromRow ? 1 : toRow < fromRow ? -1 : 0;
  const colDir = toCol > fromCol ? 1 : toCol < fromCol ? -1 : 0;

  let currentRow = fromRow + rowDir;
  let currentCol = fromCol + colDir;

  while (currentRow !== toRow || currentCol !== toCol) {
    if (board[currentRow][currentCol].piece) return false;
    currentRow += rowDir;
    currentCol += colDir;
  }

  return true;
};

export const isInCheck = (
  board: BoardSquare[][],
  kingColor: PieceColor,
): boolean => {
  // Find the king
  let kingRow = -1;
  let kingCol = -1;

  for (let row = 0; row < 8; row++) {
    for (let col = 0; col < 8; col++) {
      const piece = board[row][col].piece;
      if (piece && piece.type === "king" && piece.color === kingColor) {
        kingRow = row;
        kingCol = col;
        break;
      }
    }
    if (kingRow !== -1) break;
  }

  if (kingRow === -1) return false; // King not found

  // Check if any enemy piece can attack the king
  for (let row = 0; row < 8; row++) {
    for (let col = 0; col < 8; col++) {
      const piece = board[row][col].piece;
      if (piece && piece.color !== kingColor) {
        if (isValidMove(board, row, col, kingRow, kingCol)) {
          return true;
        }
      }
    }
  }

  return false;
};

export const getAllValidMoves = (
  board: BoardSquare[][],
  color: PieceColor,
): Move[] => {
  const moves: Move[] = [];

  for (let fromRow = 0; fromRow < 8; fromRow++) {
    for (let fromCol = 0; fromCol < 8; fromCol++) {
      const piece = board[fromRow][fromCol].piece;
      if (piece && piece.color === color) {
        for (let toRow = 0; toRow < 8; toRow++) {
          for (let toCol = 0; toCol < 8; toCol++) {
            if (isValidMove(board, fromRow, fromCol, toRow, toCol)) {
              // Test if this move would leave the king in check
              const testBoard = JSON.parse(
                JSON.stringify(board),
              ) as BoardSquare[][];
              testBoard[toRow][toCol].piece = testBoard[fromRow][fromCol].piece;
              testBoard[fromRow][fromCol].piece = undefined;

              if (!isInCheck(testBoard, color)) {
                moves.push({
                  fromRow,
                  fromCol,
                  toRow,
                  toCol,
                  capturedPiece: board[toRow][toCol].piece,
                });
              }
            }
          }
        }
      }
    }
  }

  return moves;
};

export const isCheckmate = (
  board: BoardSquare[][],
  color: PieceColor,
): boolean => {
  return isInCheck(board, color) && getAllValidMoves(board, color).length === 0;
};

export const isStalemate = (
  board: BoardSquare[][],
  color: PieceColor,
): boolean => {
  return (
    !isInCheck(board, color) && getAllValidMoves(board, color).length === 0
  );
};

// Simple AI that makes random moves for now
export const getAIMove = (
  board: BoardSquare[][],
  color: PieceColor,
): Move | null => {
  const validMoves = getAllValidMoves(board, color);

  if (validMoves.length === 0) return null;

  // For now, prioritize captures
  const captures = validMoves.filter((move) => move.capturedPiece);
  if (captures.length > 0) {
    return captures[Math.floor(Math.random() * captures.length)];
  }

  return validMoves[Math.floor(Math.random() * validMoves.length)];
};
