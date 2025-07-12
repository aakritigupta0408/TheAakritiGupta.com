import { useState } from "react";
import { motion } from "framer-motion";
import Chess from "@/components/Chess";
import BaghChal from "@/components/BaghChal";

type GameTab = "chess" | "bagh-chal";

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
  const [email, setEmail] = useState("");
  const [emailSubmitted, setEmailSubmitted] = useState(false);

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
      if (currentPlayer === "white" || isThinking) return; // Only allow black (human) moves when it's their turn

      if (selectedSquare) {
        const moveSuccessful = makeMove(
          selectedSquare.row,
          selectedSquare.col,
          row,
          col,
          "black",
        );

        setSelectedSquare(null);
        setValidMoves([]);
      } else {
        // Select piece
        const square = board[row][col];
        if (square.piece && square.piece.color === "black") {
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

  // AI move effect (Aakriti plays as white)
  useEffect(() => {
    if (currentPlayer === "white" && gameStatus === "playing") {
      setIsThinking(true);
      const timer = setTimeout(() => {
        const aiMove = getAIMove(board, "white");
        if (aiMove) {
          makeMove(
            aiMove.fromRow,
            aiMove.fromCol,
            aiMove.toRow,
            aiMove.toCol,
            "white",
          );
        }
        setIsThinking(false);
      }, 1500); // 1.5 second delay for Aakriti thinking

      return () => clearTimeout(timer);
    }
  }, [currentPlayer, board, makeMove, gameStatus]);

  const isSquareLight = (row: number, col: number) => (row + col) % 2 === 0;

  const resetGame = () => {
    setBoard(initializeBoard());
    setSelectedSquare(null);
    setCurrentPlayer("white");
    setRevealedStory(null);
    setCapturedPieces([]);
    setGameStatus("playing");
    setIsThinking(false);
    setValidMoves([]);
  };

  const handleEmailSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (email) {
      // Here you would typically send the email to your backend
      console.log("Email submitted:", email);
      setEmailSubmitted(true);
      // Reset after 3 seconds
      setTimeout(() => {
        setEmailSubmitted(false);
        setEmail("");
      }, 3000);
    }
  };

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
          Challenge the AI-Powered Professional Portfolio
        </motion.p>
        <motion.p
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3 }}
          className="text-sm text-slate-500 dark:text-slate-400 mt-2 max-w-2xl mx-auto"
        >
          Welcome! You're about to play chess against Aakriti's AI portfolio.
          Each piece represents her professional identity - capture them to
          unlock her story.
        </motion.p>
        <motion.p
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.4 }}
          className="text-sm text-slate-500 dark:text-slate-400 mt-1"
        >
          {gameStatus === "checkmate" ? (
            <span className="font-bold text-red-600">
              Checkmate! {currentPlayer === "white" ? "You" : "Aakriti"} wins!
            </span>
          ) : gameStatus === "stalemate" ? (
            <span className="font-bold text-yellow-600">Stalemate! Draw!</span>
          ) : gameStatus === "check" ? (
            <span className="font-bold text-orange-600">
              Check! Current Player:{" "}
              <span className="capitalize">
                {currentPlayer === "white" ? "Aakriti (White)" : "You (Black)"}
              </span>
            </span>
          ) : isThinking ? (
            <span className="font-semibold text-blue-600">
              🤔 Aakriti is thinking...
            </span>
          ) : (
            <span>
              Current Player:{" "}
              <span className="font-semibold capitalize">
                {currentPlayer === "white" ? "Aakriti (White)" : "You (Black)"}
              </span>
            </span>
          )}
        </motion.p>
        <motion.button
          initial={{ opacity: 0, scale: 0.8 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: 0.6 }}
          onClick={resetGame}
          className="mt-4 px-6 py-2 bg-gradient-to-r from-amber-500 to-orange-500 hover:from-amber-600 hover:to-orange-600 text-white font-semibold rounded-lg shadow-lg transition-all duration-200 transform hover:scale-105"
        >
          Reset Game
        </motion.button>
      </div>

      <div className="flex flex-col xl:flex-row justify-center items-center xl:items-start gap-8 px-4 pb-8">
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
                            isThinking && square.piece.color === "white"
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
                              isThinking && square.piece.color === "white"
                                ? Infinity
                                : 0,
                            duration: 1,
                            ease: "easeInOut",
                          },
                        }}
                        className="drop-shadow-2xl cursor-pointer select-none font-bold"
                        style={{
                          filter:
                            square.piece.color === "white"
                              ? "drop-shadow(0 0 8px rgba(255, 255, 255, 0.8)) drop-shadow(0 2px 4px rgba(0, 0, 0, 0.3))"
                              : "drop-shadow(0 0 8px rgba(0, 0, 0, 0.8)) drop-shadow(0 2px 4px rgba(255, 255, 255, 0.2))",
                          color:
                            square.piece.color === "white"
                              ? "#f8fafc"
                              : "#1e293b",
                        }}
                        title={
                          pieceDescriptions[square.piece.color][
                            square.piece.type
                          ]
                        }
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
          className="w-full lg:w-96 max-w-md"
        >
          <div className="bg-white/90 dark:bg-slate-800/90 backdrop-blur-lg rounded-2xl p-6 shadow-2xl h-96 overflow-y-auto border border-white/20 dark:border-slate-700/20">
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
                  <div className="text-6xl mb-4">���️</div>
                  <p className="text-lg font-medium">
                    Play against Aakriti's AI
                  </p>
                  <p className="text-sm">Capture pieces to unlock her story</p>
                  <div className="mt-6 text-xs text-slate-400 space-y-1">
                    <p>🤖 You're challenging Aakriti's AI portfolio</p>
                    <p>⚪ White pieces = Aakriti's strengths</p>
                    <p>⚫ Black pieces = Your perspective on her growth</p>
                    <p>🎯 Capture pieces to reveal professional insights</p>
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
              className="mt-4 bg-white/90 dark:bg-slate-800/90 backdrop-blur-lg rounded-xl p-4 shadow-xl border border-white/20 dark:border-slate-700/20"
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

      {/* Email Signup & Social Links Section */}
      <motion.div
        initial={{ opacity: 0, y: 50 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 1.0 }}
        className="mt-16 max-w-4xl mx-auto px-4 pb-16"
      >
        <div className="bg-white/90 dark:bg-slate-800/90 backdrop-blur-lg rounded-2xl p-8 shadow-2xl border border-white/20 dark:border-slate-700/20">
          <div className="text-center">
            <h2 className="text-3xl font-bold bg-gradient-to-r from-amber-600 via-yellow-500 to-orange-500 bg-clip-text text-transparent mb-4">
              Connect with Aakriti
            </h2>
            <p className="text-slate-600 dark:text-slate-300 mb-8 max-w-2xl mx-auto">
              Get her complete resume and connect with her professional journey
              in AI, engineering, and luxury tech.
            </p>
          </div>

          <div className="grid md:grid-cols-2 gap-8 items-center">
            {/* Email Signup */}
            <div>
              <h3 className="text-xl font-semibold text-slate-800 dark:text-slate-200 mb-4">
                📄 Get Aakriti's Resume
              </h3>
              <form onSubmit={handleEmailSubmit} className="space-y-4">
                <div>
                  <input
                    type="email"
                    value={email}
                    onChange={(e) => setEmail(e.target.value)}
                    placeholder="Enter your email address"
                    className="w-full px-4 py-3 border border-slate-300 dark:border-slate-600 rounded-lg bg-white dark:bg-slate-700 text-slate-900 dark:text-slate-100 focus:ring-2 focus:ring-amber-500 focus:border-transparent transition-colors"
                    required
                  />
                </div>
                <motion.button
                  type="submit"
                  disabled={emailSubmitted}
                  whileHover={{ scale: emailSubmitted ? 1 : 1.05 }}
                  whileTap={{ scale: emailSubmitted ? 1 : 0.95 }}
                  className={`w-full py-3 rounded-lg font-semibold transition-all duration-200 ${
                    emailSubmitted
                      ? "bg-green-500 text-white cursor-default"
                      : "bg-gradient-to-r from-amber-500 to-orange-500 hover:from-amber-600 hover:to-orange-600 text-white shadow-lg"
                  }`}
                >
                  {emailSubmitted ? "✅ Resume Sent!" : "Get Resume"}
                </motion.button>
              </form>
            </div>

            {/* Social Links */}
            <div>
              <h3 className="text-xl font-semibold text-slate-800 dark:text-slate-200 mb-4">
                🔗 Professional Links
              </h3>
              <div className="space-y-4">
                <motion.a
                  href="https://linkedin.com/in/aakriti-gupta"
                  target="_blank"
                  rel="noopener noreferrer"
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                  className="flex items-center gap-3 p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg border border-blue-200 dark:border-blue-800 hover:bg-blue-100 dark:hover:bg-blue-900/40 transition-colors group"
                >
                  <div className="w-8 h-8 bg-blue-600 rounded flex items-center justify-center text-white text-sm font-bold">
                    in
                  </div>
                  <div>
                    <div className="font-semibold text-slate-800 dark:text-slate-200">
                      LinkedIn Profile
                    </div>
                    <div className="text-sm text-slate-600 dark:text-slate-400">
                      Professional experience & network
                    </div>
                  </div>
                  <div className="ml-auto text-blue-600 group-hover:translate-x-1 transition-transform">
                    →
                  </div>
                </motion.a>

                <motion.a
                  href="https://github.com/aakriti-gupta"
                  target="_blank"
                  rel="noopener noreferrer"
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                  className="flex items-center gap-3 p-4 bg-slate-50 dark:bg-slate-700/50 rounded-lg border border-slate-200 dark:border-slate-600 hover:bg-slate-100 dark:hover:bg-slate-700/80 transition-colors group"
                >
                  <div className="w-8 h-8 bg-slate-800 dark:bg-slate-600 rounded flex items-center justify-center text-white text-lg">
                    ⚡
                  </div>
                  <div>
                    <div className="font-semibold text-slate-800 dark:text-slate-200">
                      GitHub Profile
                    </div>
                    <div className="text-sm text-slate-600 dark:text-slate-400">
                      Open source projects & code
                    </div>
                  </div>
                  <div className="ml-auto text-slate-600 group-hover:translate-x-1 transition-transform">
                    →
                  </div>
                </motion.a>
              </div>
            </div>
          </div>
        </div>
      </motion.div>
    </div>
  );
}
