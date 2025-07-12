import { useState, useCallback, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  ChessPiece,
  BoardSquare,
  PieceType,
  PieceColor,
  isValidMove,
  isInCheck,
  isCheckmate,
  isStalemate,
  getAIMove,
} from "@/lib/chess";

// Story content mapped to pieces
const storyContent = {
  // Aakriti's pieces (white) - Strengths when captured
  queen: {
    title: "Luxury Vision",
    content:
      "I founded Swarnawastra, a luxury fashion-tech house that blends AI-driven customization with ultra-rare materials like real gold and lab-grown diamonds.",
    isStrength: true,
  },
  king: {
    title: "Core Identity",
    content:
      "I want to democratize what was once reserved for a tiny elite — whether that's the power of advanced AI, the global stage for talented designers, or the chance for small businesses to look world-class.",
    isStrength: true,
  },
  rook1: {
    title: "Meta Engineering",
    content:
      "At Meta, I engineered sophisticated ML-driven budget pacing & ad delivery systems, optimizing billions in ad spend and improving ROI for major advertisers.",
    isStrength: true,
  },
  rook2: {
    title: "Recognition",
    content:
      "Awarded by Dr. Yann LeCun (Chief AI Scientist at Meta & Turing Award winner) for developing an engineering-efficient ML solution that balanced cost, performance, and accuracy.",
    isStrength: true,
  },
  bishop1: {
    title: "AI Innovation",
    content:
      "I founded an AI company that transforms low-quality images into professional-grade product shots, helping thousands of MSMEs compete with global e-commerce giants.",
    isStrength: true,
  },
  bishop2: {
    title: "Civic Impact",
    content:
      "Developed a face recognition system deployed in the Indian Parliament and built PPE detection systems for Tata, automating compliance in industrial settings.",
    isStrength: true,
  },
  knight1: {
    title: "Technical Foundation",
    content:
      "B.Tech in Engineering with advanced coursework in machine learning and optimization, building rigorous foundations in algorithms and system design.",
    isStrength: true,
  },
  knight2: {
    title: "Corporate Journey",
    content:
      "Enhanced search and product discovery at eBay, helping millions find products faster. At Yahoo, worked on high-volume mail infrastructure & search.",
    isStrength: true,
  },

  // Opponent pieces (black) - Weaknesses when captured
  opponentQueen: {
    title: "Perfectionism",
    content:
      "Sometimes I get so focused on engineering the perfect solution that I spend too much time optimizing when 'good enough' would suffice for the initial iteration.",
    isStrength: false,
  },
  opponentKing: {
    title: "Impatience with Inefficiency",
    content:
      "I can get frustrated when working with systems or people that move slowly, especially when I see clear paths to optimization.",
    isStrength: false,
  },
  opponentRook1: {
    title: "Technical Depth vs Breadth",
    content:
      "My deep technical background sometimes makes me dive too deep into implementation details when strategic oversight is more valuable.",
    isStrength: false,
  },
  opponentRook2: {
    title: "Scale Thinking",
    content:
      "Having worked on billion-dollar systems, I sometimes over-engineer solutions for smaller-scale problems that need simpler approaches.",
    isStrength: false,
  },
};

// Chess piece Unicode symbols
const pieceSymbols: Record<PieceColor, Record<PieceType, string>> = {
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

// Initialize chess board
const initializeBoard = (): BoardSquare[][] => {
  const board: BoardSquare[][] = [];

  // Initialize empty board
  for (let row = 0; row < 8; row++) {
    board[row] = [];
    for (let col = 0; col < 8; col++) {
      board[row][col] = { row, col };
    }
  }

  // Set up white pieces (Aakriti - bottom)
  board[7][0].piece = {
    type: "rook",
    color: "white",
    id: "rook1",
    story: storyContent.rook1,
  };
  board[7][1].piece = {
    type: "knight",
    color: "white",
    id: "knight1",
    story: storyContent.knight1,
  };
  board[7][2].piece = {
    type: "bishop",
    color: "white",
    id: "bishop1",
    story: storyContent.bishop1,
  };
  board[7][3].piece = {
    type: "queen",
    color: "white",
    id: "queen",
    story: storyContent.queen,
  };
  board[7][4].piece = {
    type: "king",
    color: "white",
    id: "king",
    story: storyContent.king,
  };
  board[7][5].piece = {
    type: "bishop",
    color: "white",
    id: "bishop2",
    story: storyContent.bishop2,
  };
  board[7][6].piece = {
    type: "knight",
    color: "white",
    id: "knight2",
    story: storyContent.knight2,
  };
  board[7][7].piece = {
    type: "rook",
    color: "white",
    id: "rook2",
    story: storyContent.rook2,
  };

  // White pawns
  for (let col = 0; col < 8; col++) {
    board[6][col].piece = { type: "pawn", color: "white", id: `pawn-w-${col}` };
  }

  // Set up black pieces (opponent - top)
  board[0][0].piece = {
    type: "rook",
    color: "black",
    id: "rook-b-1",
    story: storyContent.opponentRook1,
  };
  board[0][1].piece = { type: "knight", color: "black", id: "knight-b-1" };
  board[0][2].piece = { type: "bishop", color: "black", id: "bishop-b-1" };
  board[0][3].piece = {
    type: "queen",
    color: "black",
    id: "queen-b",
    story: storyContent.opponentQueen,
  };
  board[0][4].piece = {
    type: "king",
    color: "black",
    id: "king-b",
    story: storyContent.opponentKing,
  };
  board[0][5].piece = { type: "bishop", color: "black", id: "bishop-b-2" };
  board[0][6].piece = { type: "knight", color: "black", id: "knight-b-2" };
  board[0][7].piece = {
    type: "rook",
    color: "black",
    id: "rook-b-2",
    story: storyContent.opponentRook2,
  };

  // Black pawns
  for (let col = 0; col < 8; col++) {
    board[1][col].piece = { type: "pawn", color: "black", id: `pawn-b-${col}` };
  }

  return board;
};

export default function Index() {
  const [board, setBoard] = useState<BoardSquare[][]>(initializeBoard);
  const [selectedSquare, setSelectedSquare] = useState<{
    row: number;
    col: number;
  } | null>(null);
  const [currentPlayer, setCurrentPlayer] = useState<PieceColor>("white");
  const [revealedStory, setRevealedStory] = useState<{
    title: string;
    content: string;
    isStrength: boolean;
  } | null>(null);
  const [capturedPieces, setCapturedPieces] = useState<ChessPiece[]>([]);
  const [gameStatus, setGameStatus] = useState<
    "playing" | "check" | "checkmate" | "stalemate"
  >("playing");
  const [isThinking, setIsThinking] = useState(false);
  const [validMoves, setValidMoves] = useState<{ row: number; col: number }[]>(
    [],
  );

  const makeMove = useCallback(
    (
      fromRow: number,
      fromCol: number,
      toRow: number,
      toCol: number,
      playerColor: PieceColor,
    ) => {
      const fromSquare = board[fromRow][fromCol];
      const toSquare = board[toRow][toCol];

      if (
        !fromSquare.piece ||
        fromSquare.piece.color !== playerColor ||
        !isValidMove(board, fromRow, fromCol, toRow, toCol)
      ) {
        return false;
      }

      const newBoard = [...board.map((row) => [...row])];

      // Handle capture
      if (toSquare.piece) {
        const capturedPiece = toSquare.piece;
        setCapturedPieces((prev) => [...prev, capturedPiece]);

        // Reveal story if piece has one
        if (capturedPiece.story) {
          setRevealedStory(capturedPiece.story);
        }
      }

      // Move piece
      newBoard[toRow][toCol].piece = {
        ...fromSquare.piece,
        hasMoved: true,
      };
      newBoard[fromRow][fromCol].piece = undefined;

      setBoard(newBoard);

      // Check game status
      const nextPlayer = playerColor === "white" ? "black" : "white";
      if (isCheckmate(newBoard, nextPlayer)) {
        setGameStatus("checkmate");
      } else if (isStalemate(newBoard, nextPlayer)) {
        setGameStatus("stalemate");
      } else if (isInCheck(newBoard, nextPlayer)) {
        setGameStatus("check");
      } else {
        setGameStatus("playing");
      }

      setCurrentPlayer(nextPlayer);
      return true;
    },
    [board],
  );

  const handleSquareClick = useCallback(
    (row: number, col: number) => {
      if (currentPlayer === "black" || isThinking) return; // Only allow white (human) moves

      if (selectedSquare) {
        const moveSuccessful = makeMove(
          selectedSquare.row,
          selectedSquare.col,
          row,
          col,
          "white",
        );

        setSelectedSquare(null);
        setValidMoves([]);
      } else {
        // Select piece
        const square = board[row][col];
        if (square.piece && square.piece.color === "white") {
          setSelectedSquare({ row, col });
          // Calculate valid moves for this piece
          const moves = [];
          for (let r = 0; r < 8; r++) {
            for (let c = 0; c < 8; c++) {
              if (isValidMove(board, row, col, r, c)) {
                moves.push({ row: r, col: c });
              }
            }
          }
          setValidMoves(moves);
        }
      }
    },
    [board, selectedSquare, currentPlayer, makeMove, isThinking],
  );

  // AI move effect
  useEffect(() => {
    if (currentPlayer === "black" && gameStatus === "playing") {
      setIsThinking(true);
      const timer = setTimeout(() => {
        const aiMove = getAIMove(board, "black");
        if (aiMove) {
          makeMove(
            aiMove.fromRow,
            aiMove.fromCol,
            aiMove.toRow,
            aiMove.toCol,
            "black",
          );
        }
        setIsThinking(false);
      }, 1000); // 1 second delay for AI thinking

      return () => clearTimeout(timer);
    }
  }, [currentPlayer, board, makeMove, gameStatus]);

  const isSquareLight = (row: number, col: number) => (row + col) % 2 === 0;

  return (
    <div className="min-h-screen bg-gradient-to-br from-amber-50 via-orange-50 to-red-50 dark:from-slate-900 dark:via-slate-800 dark:to-slate-900">
      {/* Header */}
      <div className="text-center pt-8 pb-4">
        <motion.h1
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-4xl md:text-6xl font-bold bg-gradient-to-r from-amber-600 via-orange-600 to-red-600 bg-clip-text text-transparent"
        >
          TheAakritiGupta.com
        </motion.h1>
        <motion.p
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
          className="text-lg md:text-xl text-slate-600 dark:text-slate-300 mt-2"
        >
          A Chess Game of Professional Stories
        </motion.p>
        <motion.p
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.4 }}
          className="text-sm text-slate-500 dark:text-slate-400 mt-1"
        >
          {gameStatus === "checkmate" ? (
            <span className="font-bold text-red-600">
              Checkmate! {currentPlayer === "white" ? "Black" : "Aakriti"} wins!
            </span>
          ) : gameStatus === "stalemate" ? (
            <span className="font-bold text-yellow-600">Stalemate! Draw!</span>
          ) : gameStatus === "check" ? (
            <span className="font-bold text-orange-600">
              Check! Current Player:{" "}
              <span className="capitalize">
                {currentPlayer === "white"
                  ? "Aakriti (White)"
                  : "Opponent (Black)"}
              </span>
            </span>
          ) : isThinking ? (
            <span className="font-semibold text-blue-600">
              🤔 Opponent is thinking...
            </span>
          ) : (
            <span>
              Current Player:{" "}
              <span className="font-semibold capitalize">
                {currentPlayer === "white"
                  ? "Aakriti (White)"
                  : "Opponent (Black)"}
              </span>
            </span>
          )}
        </motion.p>
      </div>

      <div className="flex flex-col lg:flex-row justify-center items-start gap-8 px-4 pb-8">
        {/* Chess Board */}
        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: 0.6 }}
          className="bg-white/90 dark:bg-slate-800/90 backdrop-blur-lg p-6 rounded-2xl shadow-2xl border border-white/20 dark:border-slate-700/20"
        >
          <div className="grid grid-cols-8 gap-0 border-4 border-amber-800 dark:border-amber-600 rounded-lg overflow-hidden">
            {board.map((row, rowIndex) =>
              row.map((square, colIndex) => (
                <motion.div
                  key={`${rowIndex}-${colIndex}`}
                  className={`
                    w-16 h-16 md:w-20 md:h-20 flex items-center justify-center cursor-pointer relative text-3xl md:text-4xl select-none transition-all duration-200
                    ${
                      isSquareLight(rowIndex, colIndex)
                        ? "bg-chess-light hover:bg-chess-light/80 shadow-inner"
                        : "bg-chess-dark hover:bg-chess-dark/80 shadow-lg"
                    }
                    ${
                      selectedSquare?.row === rowIndex &&
                      selectedSquare?.col === colIndex
                        ? "ring-4 ring-chess-selected ring-inset shadow-lg"
                        : ""
                    }
                    ${
                      validMoves.some(
                        (move) =>
                          move.row === rowIndex && move.col === colIndex,
                      )
                        ? "ring-2 ring-chess-highlight ring-inset after:content-[''] after:absolute after:inset-2 after:rounded-full after:bg-chess-highlight/20"
                        : ""
                    }
                  `}
                  onClick={() => handleSquareClick(rowIndex, colIndex)}
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                  animate={{
                    scale:
                      selectedSquare?.row === rowIndex &&
                      selectedSquare?.col === colIndex
                        ? 1.1
                        : 1,
                  }}
                >
                  <AnimatePresence>
                    {square.piece && (
                      <motion.span
                        initial={{ scale: 0, rotate: 180 }}
                        animate={{
                          scale: 1,
                          rotate: 0,
                          y:
                            isThinking && square.piece.color === "black"
                              ? [0, -2, 0]
                              : 0,
                        }}
                        exit={{ scale: 0, rotate: -180 }}
                        transition={{
                          type: "spring",
                          stiffness: 200,
                          damping: 15,
                          y: {
                            repeat:
                              isThinking && square.piece.color === "black"
                                ? Infinity
                                : 0,
                            duration: 1,
                            ease: "easeInOut",
                          },
                        }}
                        className="drop-shadow-lg"
                      >
                        {pieceSymbols[square.piece.color][square.piece.type]}
                      </motion.span>
                    )}
                  </AnimatePresence>
                </motion.div>
              )),
            )}
          </div>
        </motion.div>

        {/* Story Reveal Panel */}
        <motion.div
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.8 }}
          className="w-full lg:w-96"
        >
          <div className="bg-white dark:bg-slate-800 rounded-2xl p-6 shadow-2xl h-96 overflow-y-auto">
            <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-200 mb-4">
              Professional Journey
            </h3>

            <AnimatePresence mode="wait">
              {revealedStory ? (
                <motion.div
                  key={revealedStory.title}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -20 }}
                  className={`p-4 rounded-lg ${
                    revealedStory.isStrength
                      ? "bg-story-strength/10 border border-story-strength/20"
                      : "bg-story-weakness/10 border border-story-weakness/20"
                  }`}
                >
                  <div className="flex items-center gap-2 mb-3">
                    <span
                      className={`text-2xl ${revealedStory.isStrength ? "text-story-strength" : "text-story-weakness"}`}
                    >
                      {revealedStory.isStrength ? "💪" : "🎯"}
                    </span>
                    <h4
                      className={`font-bold text-lg ${
                        revealedStory.isStrength
                          ? "text-story-strength"
                          : "text-story-weakness"
                      }`}
                    >
                      {revealedStory.isStrength
                        ? "Strength: "
                        : "Growth Area: "}
                      {revealedStory.title}
                    </h4>
                  </div>
                  <p className="text-slate-700 dark:text-slate-300 leading-relaxed">
                    {revealedStory.content}
                  </p>
                </motion.div>
              ) : (
                <motion.div
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  className="text-center text-slate-500 dark:text-slate-400 py-12"
                >
                  <div className="text-6xl mb-4">♟️</div>
                  <p className="text-lg font-medium">
                    Capture pieces to reveal
                  </p>
                  <p className="text-sm">Aakriti's professional story</p>
                  <div className="mt-6 text-xs text-slate-400 space-y-1">
                    <p>• White pieces = Aakriti's strengths</p>
                    <p>• Black pieces = Growth areas</p>
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
          </div>

          {/* Captured Pieces */}
          {capturedPieces.length > 0 && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              className="mt-4 bg-white dark:bg-slate-800 rounded-xl p-4 shadow-xl"
            >
              <h4 className="font-semibold text-slate-700 dark:text-slate-300 mb-2">
                Captured Pieces ({capturedPieces.length})
              </h4>
              <div className="flex flex-wrap gap-2">
                {capturedPieces.map((piece, index) => (
                  <motion.span
                    key={`captured-${index}`}
                    initial={{ scale: 0, rotate: 180 }}
                    animate={{ scale: 1, rotate: 0 }}
                    className="text-2xl opacity-60"
                  >
                    {pieceSymbols[piece.color][piece.type]}
                  </motion.span>
                ))}
              </div>
            </motion.div>
          )}
        </motion.div>
      </div>
    </div>
  );
}
