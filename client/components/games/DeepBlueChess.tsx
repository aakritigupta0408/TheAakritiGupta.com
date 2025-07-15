import { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";

interface ChessPosition {
  row: number;
  col: number;
}

interface ChessPiece {
  type: "king" | "queen" | "rook" | "bishop" | "knight" | "pawn";
  color: "white" | "black";
  position: ChessPosition;
  hasMoved?: boolean;
}

const initialChessBoard = (): (ChessPiece | null)[][] => {
  const board: (ChessPiece | null)[][] = Array(8)
    .fill(null)
    .map(() => Array(8).fill(null));

  // Set up initial chess position
  // Black pieces (top)
  const blackPieces = [
    "rook",
    "knight",
    "bishop",
    "queen",
    "king",
    "bishop",
    "knight",
    "rook",
  ] as const;
  for (let col = 0; col < 8; col++) {
    board[0][col] = {
      type: blackPieces[col],
      color: "black",
      position: { row: 0, col },
    };
    board[1][col] = {
      type: "pawn",
      color: "black",
      position: { row: 1, col },
    };
  }

  // White pieces (bottom)
  const whitePieces = [
    "rook",
    "knight",
    "bishop",
    "queen",
    "king",
    "bishop",
    "knight",
    "rook",
  ] as const;
  for (let col = 0; col < 8; col++) {
    board[7][col] = {
      type: whitePieces[col],
      color: "white",
      position: { row: 7, col },
    };
    board[6][col] = {
      type: "pawn",
      color: "white",
      position: { row: 6, col },
    };
  }

  return board;
};

const pieceSymbols = {
  white: {
    king: "♔",
    queen: "♕",
    rook: "♖",
    bishop: "♗",
    knight: "♘",
    pawn: "♙",
  },
  black: {
    king: "♚",
    queen: "♛",
    rook: "♜",
    bishop: "♝",
    knight: "♞",
    pawn: "♟",
  },
};

const famousGameMoves = [
  { from: { row: 6, col: 4 }, to: { row: 4, col: 4 }, notation: "1. e4" },
  { from: { row: 1, col: 2 }, to: { row: 3, col: 2 }, notation: "1... c5" },
  { from: { row: 7, col: 6 }, to: { row: 5, col: 5 }, notation: "2. Nf3" },
  { from: { row: 1, col: 3 }, to: { row: 3, col: 3 }, notation: "2... d6" },
  { from: { row: 6, col: 3 }, to: { row: 4, col: 3 }, notation: "3. d4" },
  { from: { row: 3, col: 2 }, to: { row: 4, col: 3 }, notation: "3... cxd4" },
];

export default function DeepBlueChess() {
  const [board, setBoard] =
    useState<(ChessPiece | null)[][]>(initialChessBoard);
  const [selectedSquare, setSelectedSquare] = useState<ChessPosition | null>(
    null,
  );
  const [currentPlayer, setCurrentPlayer] = useState<"white" | "black">(
    "white",
  );
  const [gamePhase, setGamePhase] = useState<
    "setup" | "playing" | "demo" | "ended"
  >("setup");
  const [moveCount, setMoveCount] = useState(0);
  const [capturedPieces, setCapturedPieces] = useState<{
    white: ChessPiece[];
    black: ChessPiece[];
  }>({ white: [], black: [] });
  const [demoMoveIndex, setDemoMoveIndex] = useState(0);
  const [isThinking, setIsThinking] = useState(false);
  const [gameHistory, setGameHistory] = useState<string[]>([]);

  const resetGame = () => {
    setBoard(initialChessBoard());
    setSelectedSquare(null);
    setCurrentPlayer("white");
    setGamePhase("setup");
    setMoveCount(0);
    setCapturedPieces({ white: [], black: [] });
    setDemoMoveIndex(0);
    setIsThinking(false);
    setGameHistory([]);
  };

  const startDemo = () => {
    resetGame();
    setGamePhase("demo");
    setDemoMoveIndex(0);
  };

  const startGame = () => {
    resetGame();
    setGamePhase("playing");
  };

  // Demo mode - replay famous game
  useEffect(() => {
    if (gamePhase === "demo" && demoMoveIndex < famousGameMoves.length) {
      const timer = setTimeout(() => {
        const move = famousGameMoves[demoMoveIndex];
        setIsThinking(true);

        setTimeout(() => {
          setBoard((prevBoard) => {
            const newBoard = prevBoard.map((row) => [...row]);
            const piece = newBoard[move.from.row][move.from.col];

            if (piece) {
              // Capture piece if there's one at destination
              const capturedPiece = newBoard[move.to.row][move.to.col];
              if (capturedPiece) {
                setCapturedPieces((prev) => ({
                  ...prev,
                  [capturedPiece.color === "white" ? "black" : "white"]: [
                    ...prev[
                      capturedPiece.color === "white" ? "black" : "white"
                    ],
                    capturedPiece,
                  ],
                }));
              }

              // Move piece
              piece.position = move.to;
              piece.hasMoved = true;
              newBoard[move.to.row][move.to.col] = piece;
              newBoard[move.from.row][move.from.col] = null;
            }

            return newBoard;
          });

          setGameHistory((prev) => [...prev, move.notation]);
          setDemoMoveIndex((prev) => prev + 1);
          setCurrentPlayer((prev) => (prev === "white" ? "black" : "white"));
          setMoveCount((prev) => prev + 1);
          setIsThinking(false);
        }, 1000);
      }, 2000);

      return () => clearTimeout(timer);
    }
  }, [gamePhase, demoMoveIndex]);

  const isValidMove = (
    from: ChessPosition,
    to: ChessPosition,
    piece: ChessPiece,
  ): boolean => {
    // Basic validation - can be expanded
    if (to.row < 0 || to.row > 7 || to.col < 0 || to.col > 7) return false;
    if (from.row === to.row && from.col === to.col) return false;

    const targetPiece = board[to.row][to.col];
    if (targetPiece && targetPiece.color === piece.color) return false;

    // Basic pawn movement
    if (piece.type === "pawn") {
      const direction = piece.color === "white" ? -1 : 1;
      const startRow = piece.color === "white" ? 6 : 1;

      // Forward movement
      if (to.col === from.col && !targetPiece) {
        if (to.row === from.row + direction) return true;
        if (from.row === startRow && to.row === from.row + 2 * direction)
          return true;
      }

      // Diagonal capture
      if (
        Math.abs(to.col - from.col) === 1 &&
        to.row === from.row + direction &&
        targetPiece
      ) {
        return true;
      }
    }

    return true; // Simplified for demo
  };

  const handleSquareClick = (row: number, col: number) => {
    if (gamePhase !== "playing") return;

    const clickedPiece = board[row][col];

    if (selectedSquare) {
      const selectedPiece = board[selectedSquare.row][selectedSquare.col];
      if (
        selectedPiece &&
        selectedPiece.color === currentPlayer &&
        isValidMove(selectedSquare, { row, col }, selectedPiece)
      ) {
        // Make move
        setBoard((prevBoard) => {
          const newBoard = prevBoard.map((boardRow) => [...boardRow]);

          // Capture piece if present
          const capturedPiece = newBoard[row][col];
          if (capturedPiece) {
            setCapturedPieces((prev) => ({
              ...prev,
              [capturedPiece.color === "white" ? "black" : "white"]: [
                ...prev[capturedPiece.color === "white" ? "black" : "white"],
                capturedPiece,
              ],
            }));
          }

          // Move piece
          selectedPiece.position = { row, col };
          selectedPiece.hasMoved = true;
          newBoard[row][col] = selectedPiece;
          newBoard[selectedSquare.row][selectedSquare.col] = null;

          return newBoard;
        });

        setCurrentPlayer(currentPlayer === "white" ? "black" : "white");
        setMoveCount(moveCount + 1);
        setSelectedSquare(null);
      } else {
        // Select new piece or deselect
        if (clickedPiece && clickedPiece.color === currentPlayer) {
          setSelectedSquare({ row, col });
        } else {
          setSelectedSquare(null);
        }
      }
    } else {
      // Select piece
      if (clickedPiece && clickedPiece.color === currentPlayer) {
        setSelectedSquare({ row, col });
      }
    }
  };

  const getSquareColor = (row: number, col: number): string => {
    const isLight = (row + col) % 2 === 0;
    const isSelected =
      selectedSquare &&
      selectedSquare.row === row &&
      selectedSquare.col === col;

    if (isSelected) return "bg-yellow-400";
    return isLight ? "bg-amber-100" : "bg-amber-800";
  };

  return (
    <div className="w-full max-w-6xl mx-auto p-6">
      {/* Header */}
      <div className="text-center mb-8">
        <h3 className="text-3xl font-bold text-gray-800 mb-2">
          Deep Blue vs Kasparov Chess Demo
        </h3>
        <p className="text-gray-600">
          Experience the historic 1997 match that changed AI forever
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Chess Board */}
        <div className="lg:col-span-2">
          <div className="bg-amber-900 p-4 rounded-2xl shadow-2xl">
            <div className="grid grid-cols-8 gap-0 aspect-square bg-white rounded-lg overflow-hidden">
              {board.map((row, rowIndex) =>
                row.map((piece, colIndex) => (
                  <motion.button
                    key={`${rowIndex}-${colIndex}`}
                    onClick={() => handleSquareClick(rowIndex, colIndex)}
                    className={`aspect-square flex items-center justify-center text-4xl font-bold transition-all duration-200 hover:scale-105 ${getSquareColor(rowIndex, colIndex)} ${gamePhase === "playing" ? "cursor-pointer" : "cursor-default"}`}
                    whileHover={gamePhase === "playing" ? { scale: 1.05 } : {}}
                    whileTap={gamePhase === "playing" ? { scale: 0.95 } : {}}
                  >
                    {piece && (
                      <motion.span
                        initial={{ scale: 0 }}
                        animate={{ scale: 1 }}
                        className="drop-shadow-sm"
                      >
                        {pieceSymbols[piece.color][piece.type]}
                      </motion.span>
                    )}
                  </motion.button>
                )),
              )}
            </div>

            {/* Board Labels */}
            <div className="flex justify-between mt-2 px-2">
              {["a", "b", "c", "d", "e", "f", "g", "h"].map((letter) => (
                <span key={letter} className="text-amber-200 text-sm font-bold">
                  {letter}
                </span>
              ))}
            </div>
          </div>
        </div>

        {/* Game Info Panel */}
        <div className="space-y-6">
          {/* Game Status */}
          <div className="bg-white rounded-xl p-6 shadow-lg border border-gray-200">
            <h4 className="text-xl font-bold text-gray-800 mb-4">
              Game Status
            </h4>

            <div className="space-y-3">
              <div className="flex justify-between">
                <span className="text-gray-600">Current Turn:</span>
                <span className="font-bold capitalize">
                  {currentPlayer}
                  {isThinking && gamePhase === "demo" && " (AI thinking...)"}
                </span>
              </div>

              <div className="flex justify-between">
                <span className="text-gray-600">Move Count:</span>
                <span className="font-bold">{Math.ceil(moveCount / 2)}</span>
              </div>

              <div className="flex justify-between">
                <span className="text-gray-600">Game Phase:</span>
                <span className="font-bold capitalize">{gamePhase}</span>
              </div>
            </div>

            {gamePhase === "demo" && (
              <div className="mt-4 p-3 bg-blue-50 rounded-lg">
                <div className="text-sm text-blue-800">
                  🤖 Deep Blue is analyzing position...
                </div>
                <div className="text-xs text-blue-600 mt-1">
                  Processing 200 million moves per second
                </div>
              </div>
            )}
          </div>

          {/* Captured Pieces */}
          <div className="bg-white rounded-xl p-6 shadow-lg border border-gray-200">
            <h4 className="text-xl font-bold text-gray-800 mb-4">
              Captured Pieces
            </h4>

            <div className="space-y-4">
              <div>
                <div className="text-sm text-gray-600 mb-2">
                  White Captured:
                </div>
                <div className="flex flex-wrap gap-1">
                  {capturedPieces.white.map((piece, idx) => (
                    <span key={idx} className="text-2xl">
                      {pieceSymbols[piece.color][piece.type]}
                    </span>
                  ))}
                  {capturedPieces.white.length === 0 && (
                    <span className="text-gray-400 text-sm">None</span>
                  )}
                </div>
              </div>

              <div>
                <div className="text-sm text-gray-600 mb-2">
                  Black Captured:
                </div>
                <div className="flex flex-wrap gap-1">
                  {capturedPieces.black.map((piece, idx) => (
                    <span key={idx} className="text-2xl">
                      {pieceSymbols[piece.color][piece.type]}
                    </span>
                  ))}
                  {capturedPieces.black.length === 0 && (
                    <span className="text-gray-400 text-sm">None</span>
                  )}
                </div>
              </div>
            </div>
          </div>

          {/* Game Controls */}
          <div className="bg-white rounded-xl p-6 shadow-lg border border-gray-200 space-y-3">
            <h4 className="text-xl font-bold text-gray-800 mb-4">Controls</h4>

            <motion.button
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
              onClick={startDemo}
              className="w-full py-3 bg-gradient-to-r from-blue-500 to-blue-600 text-white rounded-lg font-semibold hover:from-blue-600 hover:to-blue-700 transition-all"
            >
              🎬 Watch Historic Demo
            </motion.button>

            <motion.button
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
              onClick={startGame}
              className="w-full py-3 bg-gradient-to-r from-green-500 to-green-600 text-white rounded-lg font-semibold hover:from-green-600 hover:to-green-700 transition-all"
            >
              🎮 Play Yourself
            </motion.button>

            <motion.button
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
              onClick={resetGame}
              className="w-full py-3 bg-gradient-to-r from-gray-500 to-gray-600 text-white rounded-lg font-semibold hover:from-gray-600 hover:to-gray-700 transition-all"
            >
              🔄 Reset Board
            </motion.button>
          </div>

          {/* Move History */}
          {gameHistory.length > 0 && (
            <div className="bg-white rounded-xl p-6 shadow-lg border border-gray-200">
              <h4 className="text-xl font-bold text-gray-800 mb-4">
                Move History
              </h4>
              <div className="space-y-1 max-h-40 overflow-y-auto">
                {gameHistory.map((move, idx) => (
                  <div key={idx} className="text-sm font-mono text-gray-700">
                    {move}
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Historical Context */}
      <div className="mt-8 bg-gradient-to-r from-blue-50 to-indigo-50 rounded-xl p-6 border border-blue-200">
        <h4 className="text-xl font-bold text-gray-800 mb-4">
          🏆 The Historic Match
        </h4>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div>
            <h5 className="font-bold text-gray-800 mb-2">
              Deep Blue's Victory
            </h5>
            <p className="text-gray-700 text-sm leading-relaxed">
              On May 11, 1997, IBM's Deep Blue defeated world chess champion
              Garry Kasparov in a six-game match with a score of 3.5-2.5. This
              historic victory marked the first time a computer had defeated a
              reigning world champion in a match under standard tournament
              conditions.
            </p>
          </div>
          <div>
            <h5 className="font-bold text-gray-800 mb-2">Technical Prowess</h5>
            <p className="text-gray-700 text-sm leading-relaxed">
              Deep Blue could evaluate 200 million chess positions per second
              using specialized chess chips and advanced search algorithms. The
              system combined brute-force calculation with sophisticated
              position evaluation functions developed by chess grandmasters.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
