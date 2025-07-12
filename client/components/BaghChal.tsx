import { useState, useCallback, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  BaghChalState,
  Position,
  initializeBaghChal,
  makeMove,
  getValidMoves,
  getAIMoveBoth,
} from "@/lib/bagh-chal";

const BaghChal = () => {
  const [gameState, setGameState] = useState<BaghChalState>(initializeBaghChal);
  const [aiMode, setAiMode] = useState(false);
  const [showRules, setShowRules] = useState(false);
  const [isThinking, setIsThinking] = useState(false);
  const [demoPlaying, setDemoPlaying] = useState(false);
  const [highlightedMove, setHighlightedMove] = useState<{
    from: Position;
    to: Position;
  } | null>(null);

  const resetGame = () => {
    setGameState(initializeBaghChal());
    setHighlightedMove(null);
    setIsThinking(false);
    setDemoPlaying(false);
  };

  const startDemo = () => {
    setGameState(initializeBaghChal());
    setAiMode(true);
    setDemoPlaying(true);
    setHighlightedMove(null);
    setIsThinking(false);
  };

  const stopDemo = () => {
    setDemoPlaying(false);
    setAiMode(false);
    setIsThinking(false);
    setHighlightedMove(null);
  };

  const handlePositionClick = useCallback(
    (row: number, col: number) => {
      if (gameState.gameOver || aiMode) return;

      const clickedPosition = { row, col };

      if (gameState.phase === "placement") {
        // Place goat
        if (
          gameState.board[row][col] === null &&
          gameState.currentPlayer === "goat"
        ) {
          const newState = makeMove(
            gameState,
            { row: 0, col: 0 },
            clickedPosition,
          );
          setGameState(newState);
        }
      } else {
        // Movement phase
        if (gameState.selectedPosition) {
          // Try to move
          const validMoves = getValidMoves(
            gameState,
            gameState.selectedPosition,
          );
          const isValidMove = validMoves.some(
            (move) => move.row === row && move.col === col,
          );

          if (isValidMove) {
            const newState = makeMove(
              gameState,
              gameState.selectedPosition,
              clickedPosition,
            );
            setGameState(newState);
          } else {
            // Select new position
            if (
              gameState.board[row][col] === gameState.currentPlayer ||
              gameState.board[row][col] === null
            ) {
              setGameState({
                ...gameState,
                selectedPosition:
                  gameState.board[row][col] === gameState.currentPlayer
                    ? clickedPosition
                    : null,
              });
            }
          }
        } else {
          // Select position
          if (gameState.board[row][col] === gameState.currentPlayer) {
            setGameState({
              ...gameState,
              selectedPosition: clickedPosition,
            });
          }
        }
      }
    },
    [gameState, aiMode],
  );

  // AI move effect - handles both tigers and goats
  useEffect(() => {
    if (
      (aiMode || demoPlaying) &&
      !gameState.gameOver &&
      (gameState.phase === "movement" || gameState.phase === "placement")
    ) {
      setIsThinking(true);
      const delay = gameState.currentPlayer === "tiger" ? 1200 : 800; // Faster for demo

      const timer = setTimeout(() => {
        const aiMove = getAIMoveBoth(gameState);
        if (aiMove) {
          setHighlightedMove(aiMove);
          const newState = makeMove(gameState, aiMove.from, aiMove.to);
          setGameState(newState);

          // Clear highlight after animation
          setTimeout(() => setHighlightedMove(null), 800);
        }
        setIsThinking(false);
      }, delay);

      return () => clearTimeout(timer);
    }
  }, [
    gameState.currentPlayer,
    gameState.phase,
    aiMode,
    demoPlaying,
    gameState.gameOver,
    gameState.goatsPlaced,
  ]);

  // Enhanced demo completion with winner celebration
  useEffect(() => {
    if (gameState.gameOver && demoPlaying) {
      setTimeout(() => {
        setDemoPlaying(false);
        setAiMode(false);
        // Optional: Auto-restart demo after showing winner
        // setTimeout(startDemo, 2000);
      }, 5000); // Show winner for 5 seconds before stopping
    }
  }, [gameState.gameOver, demoPlaying]);

  // Demo move counter for tracking progress
  const [moveCount, setMoveCount] = useState(0);

  useEffect(() => {
    if (demoPlaying) {
      setMoveCount(0);
    }
  }, [demoPlaying]);

  useEffect(() => {
    if (demoPlaying && !isThinking) {
      setMoveCount((prev) => prev + 1);
    }
  }, [gameState.currentPlayer, demoPlaying, isThinking]);

  const validMoves = gameState.selectedPosition
    ? getValidMoves(gameState, gameState.selectedPosition)
    : [];

  const isValidMovePosition = (row: number, col: number) =>
    validMoves.some((move) => move.row === row && move.col === col);

  return (
    <div className="w-full">
      {/* Game Header */}
      <div className="text-center mb-8">
        <h2 className="text-3xl md:text-4xl font-bold bg-gradient-to-r from-amber-600 via-yellow-500 to-orange-500 bg-clip-text text-transparent mb-4">
          Bagh-Chal: Tigers vs Goats
        </h2>
        <p className="text-slate-600 dark:text-slate-300 mb-4">
          Ancient Nepalese strategy game
        </p>

        {/* Game Controls */}
        <div className="flex flex-wrap justify-center gap-4 mb-6">
          {!demoPlaying ? (
            <motion.button
              onClick={startDemo}
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              className="px-6 py-3 bg-gradient-to-r from-green-500 to-emerald-500 text-white rounded-lg font-bold shadow-lg transition-all"
            >
              ▶️ Watch Demo Play
            </motion.button>
          ) : (
            <motion.button
              onClick={stopDemo}
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              className="px-6 py-3 bg-gradient-to-r from-red-500 to-pink-500 text-white rounded-lg font-bold shadow-lg transition-all"
            >
              ⏹️ Stop Demo
            </motion.button>
          )}
          <button
            onClick={() => setAiMode(!aiMode)}
            disabled={demoPlaying}
            className={`px-4 py-2 rounded-lg font-semibold transition-all ${
              demoPlaying
                ? "opacity-50 cursor-not-allowed"
                : aiMode
                  ? "bg-blue-500 text-white"
                  : "bg-slate-200 dark:bg-slate-700 text-slate-700 dark:text-slate-300"
            }`}
          >
            {aiMode ? "🤖 AI vs AI" : "👥 Player vs Player"}
          </button>
          <button
            onClick={resetGame}
            disabled={demoPlaying}
            className={`px-4 py-2 bg-gradient-to-r from-amber-500 to-orange-500 text-white rounded-lg font-semibold hover:scale-105 transition-transform ${
              demoPlaying ? "opacity-50 cursor-not-allowed" : ""
            }`}
          >
            Reset Game
          </button>
          <button
            onClick={() => setShowRules(!showRules)}
            className="px-4 py-2 bg-slate-200 dark:bg-slate-700 text-slate-700 dark:text-slate-300 rounded-lg font-semibold hover:scale-105 transition-transform"
          >
            {showRules ? "Hide Rules" : "Show Rules"}
          </button>
        </div>

        {/* Rules Section */}
        <AnimatePresence>
          {showRules && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: "auto" }}
              exit={{ opacity: 0, height: 0 }}
              className="bg-white/90 dark:bg-slate-800/90 backdrop-blur-lg rounded-xl p-6 mb-6 text-left border border-white/20 dark:border-slate-700/20"
            >
              <h3 className="font-bold text-lg mb-4 text-slate-800 dark:text-slate-200">
                🐅🐐 Bagh-Chal: Step by Step Rules
              </h3>
              <div className="space-y-3 text-sm text-slate-600 dark:text-slate-400 max-h-80 overflow-y-auto">
                <div className="bg-slate-50 dark:bg-slate-800 p-3 rounded-lg">
                  <h4 className="font-bold text-slate-700 dark:text-slate-300 mb-2">
                    🏁 Setup
                  </h4>
                  <p>• 4 Tigers start on corner intersections</p>
                  <p>• 20 Goats start off-board, to be placed during game</p>
                  <p>• Board has 25 intersection points connected by lines</p>
                </div>

                <div className="bg-slate-50 dark:bg-slate-800 p-3 rounded-lg">
                  <h4 className="font-bold text-slate-700 dark:text-slate-300 mb-2">
                    🎯 Objectives
                  </h4>
                  <p>
                    • <strong>Tigers win:</strong> Capture 5+ goats
                  </p>
                  <p>
                    • <strong>Goats win:</strong> Trap all tigers (no moves
                    possible)
                  </p>
                </div>

                <div className="bg-slate-50 dark:bg-slate-800 p-3 rounded-lg">
                  <h4 className="font-bold text-slate-700 dark:text-slate-300 mb-2">
                    🚶‍♂️ Game Flow
                  </h4>
                  <p>
                    <strong>Phase 1 - Placement (20 turns):</strong>
                  </p>
                  <p>• Goats place one piece per turn on empty intersections</p>
                  <p>• Tigers can move/capture immediately during this phase</p>
                  <p>• Players alternate: Goat places → Tiger moves → repeat</p>

                  <p className="mt-2">
                    <strong>Phase 2 - Movement:</strong>
                  </p>
                  <p>• After all 20 goats placed, goats can now move</p>
                  <p>• Both players move to adjacent empty intersections</p>
                  <p>• Tigers can still capture by jumping over goats</p>
                </div>

                <div className="bg-slate-50 dark:bg-slate-800 p-3 rounded-lg">
                  <h4 className="font-bold text-slate-700 dark:text-slate-300 mb-2">
                    🐅 Tiger Moves
                  </h4>
                  <p>
                    • <strong>Normal move:</strong> To adjacent empty
                    intersection
                  </p>
                  <p>
                    • <strong>Capture:</strong> Jump over adjacent goat to empty
                    space beyond
                  </p>
                  <p>
                    • Can start moving immediately (even during goat placement)
                  </p>
                </div>

                <div className="bg-slate-50 dark:bg-slate-800 p-3 rounded-lg">
                  <h4 className="font-bold text-slate-700 dark:text-slate-300 mb-2">
                    🐐 Goat Moves
                  </h4>
                  <p>
                    • <strong>Placement phase:</strong> Place on any empty
                    intersection
                  </p>
                  <p>
                    • <strong>Movement phase:</strong> Move to adjacent empty
                    intersection
                  </p>
                  <p>• Cannot jump or capture - only block and surround</p>
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      <div className="flex flex-col xl:flex-row justify-center items-center xl:items-start gap-8">
        {/* Game Board */}
        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          className="bg-white/90 dark:bg-slate-800/90 backdrop-blur-lg p-8 rounded-2xl shadow-2xl border border-white/20 dark:border-slate-700/20"
        >
          <div className="relative">
            {/* Board Grid */}
            <svg
              width="400"
              height="400"
              viewBox="0 0 400 400"
              className="absolute inset-0"
            >
              {/* Horizontal lines */}
              {[0, 1, 2, 3, 4].map((row) => (
                <line
                  key={`h-${row}`}
                  x1="50"
                  y1={50 + row * 75}
                  x2="350"
                  y2={50 + row * 75}
                  stroke="currentColor"
                  strokeWidth="2"
                  className="text-slate-400"
                />
              ))}
              {/* Vertical lines */}
              {[0, 1, 2, 3, 4].map((col) => (
                <line
                  key={`v-${col}`}
                  x1={50 + col * 75}
                  y1="50"
                  x2={50 + col * 75}
                  y2="350"
                  stroke="currentColor"
                  strokeWidth="2"
                  className="text-slate-400"
                />
              ))}
              {/* Diagonal lines */}
              {[
                // Main diagonals
                { x1: 50, y1: 50, x2: 350, y2: 350 },
                { x1: 350, y1: 50, x2: 50, y2: 350 },
                // Corner to center diagonals
                { x1: 50, y1: 200, x2: 200, y2: 50 },
                { x1: 200, y1: 50, x2: 350, y2: 200 },
                { x1: 350, y1: 200, x2: 200, y2: 350 },
                { x1: 200, y1: 350, x2: 50, y2: 200 },
                // Inner diagonals
                { x1: 125, y1: 125, x2: 275, y2: 275 },
                { x1: 275, y1: 125, x2: 125, y2: 275 },
              ].map((line, index) => (
                <line
                  key={`d-${index}`}
                  x1={line.x1}
                  y1={line.y1}
                  x2={line.x2}
                  y2={line.y2}
                  stroke="currentColor"
                  strokeWidth="2"
                  className="text-slate-400"
                />
              ))}
            </svg>

            {/* Game Pieces */}
            <div className="relative grid grid-cols-5 gap-0 w-[400px] h-[400px]">
              {gameState.board.map((row, rowIndex) =>
                row.map((piece, colIndex) => (
                  <motion.div
                    key={`${rowIndex}-${colIndex}`}
                    className={`
                      w-20 h-20 flex items-center justify-center cursor-pointer relative
                      ${
                        gameState.selectedPosition?.row === rowIndex &&
                        gameState.selectedPosition?.col === colIndex
                          ? "z-20"
                          : ""
                      }
                      ${isValidMovePosition(rowIndex, colIndex) ? "z-10" : ""}
                      ${
                        highlightedMove?.from.row === rowIndex &&
                        highlightedMove?.from.col === colIndex
                          ? "z-30"
                          : ""
                      }
                      ${
                        highlightedMove?.to.row === rowIndex &&
                        highlightedMove?.to.col === colIndex
                          ? "z-30"
                          : ""
                      }
                    `}
                    onClick={() => handlePositionClick(rowIndex, colIndex)}
                    whileHover={{ scale: 1.1 }}
                    whileTap={{ scale: 0.9 }}
                    animate={{
                      scale:
                        gameState.selectedPosition?.row === rowIndex &&
                        gameState.selectedPosition?.col === colIndex
                          ? 1.2
                          : 1,
                    }}
                  >
                    {/* Selection indicator */}
                    {gameState.selectedPosition?.row === rowIndex &&
                      gameState.selectedPosition?.col === colIndex && (
                        <motion.div
                          initial={{ scale: 0 }}
                          animate={{ scale: 1 }}
                          className="absolute inset-0 rounded-full bg-chess-selected/30 border-4 border-chess-selected"
                        />
                      )}

                    {/* Valid move indicator */}
                    {isValidMovePosition(rowIndex, colIndex) && (
                      <motion.div
                        initial={{ scale: 0 }}
                        animate={{ scale: 1 }}
                        className="absolute inset-4 rounded-full bg-chess-highlight/40 border-2 border-chess-highlight"
                      />
                    )}

                    {/* Highlight AI move */}
                    {(highlightedMove?.from.row === rowIndex &&
                      highlightedMove?.from.col === colIndex) ||
                    (highlightedMove?.to.row === rowIndex &&
                      highlightedMove?.to.col === colIndex) ? (
                      <motion.div
                        initial={{ scale: 0 }}
                        animate={{ scale: [1, 1.2, 1] }}
                        transition={{ duration: 1, repeat: 3 }}
                        className="absolute inset-0 rounded-full bg-blue-500/30 border-4 border-blue-500"
                      />
                    ) : null}

                    {/* Piece */}
                    <AnimatePresence>
                      {piece && (
                        <motion.div
                          initial={{ scale: 0, rotate: 180 }}
                          animate={{ scale: 1, rotate: 0 }}
                          exit={{ scale: 0, rotate: -180 }}
                          transition={{
                            type: "spring",
                            stiffness: 200,
                            damping: 15,
                          }}
                          className={`
                            text-4xl drop-shadow-2xl select-none font-bold z-50
                            ${
                              isThinking && piece === "tiger"
                                ? "animate-pulse"
                                : ""
                            }
                          `}
                          style={{
                            filter:
                              piece === "tiger"
                                ? "drop-shadow(0 0 8px rgba(255, 165, 0, 0.8))"
                                : "drop-shadow(0 0 8px rgba(255, 255, 255, 0.8))",
                            color: piece === "tiger" ? "#f97316" : "#f8fafc",
                          }}
                        >
                          {piece === "tiger" ? "🐅" : "🐐"}
                        </motion.div>
                      )}
                    </AnimatePresence>
                  </motion.div>
                )),
              )}
            </div>
          </div>
        </motion.div>

        {/* Game Status */}
        <motion.div
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          className="w-full xl:w-96 max-w-md"
        >
          <div className="bg-white/90 dark:bg-slate-800/90 backdrop-blur-lg rounded-2xl p-6 shadow-2xl border border-white/20 dark:border-slate-700/20">
            <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-200 mb-4">
              Game Status
            </h3>

            <div className="space-y-4">
              {/* Current Turn */}
              <div className="flex items-center justify-between p-3 bg-slate-50 dark:bg-slate-700/50 rounded-lg">
                <span className="font-medium">Current Turn:</span>
                <div className="flex items-center gap-2">
                  <span className="text-2xl">
                    {gameState.currentPlayer === "tiger" ? "🐅" : "🐐"}
                  </span>
                  <span className="font-bold capitalize">
                    {gameState.currentPlayer}
                    {isThinking && aiMode && " (AI thinking...)"}
                    {isThinking && !aiMode && " (thinking...)"}
                  </span>
                </div>
              </div>

              {/* Game Phase */}
              <div className="flex items-center justify-between p-3 bg-slate-50 dark:bg-slate-700/50 rounded-lg">
                <span className="font-medium">Phase:</span>
                <span className="font-bold capitalize">
                  {gameState.phase === "placement"
                    ? "Goat Placement"
                    : "Movement"}
                </span>
              </div>

              {/* Statistics */}
              <div className="space-y-2">
                <div className="flex items-center justify-between p-3 bg-slate-50 dark:bg-slate-700/50 rounded-lg">
                  <span className="font-medium">Goats Placed:</span>
                  <span className="font-bold">{gameState.goatsPlaced}/20</span>
                </div>
                <div className="flex items-center justify-between p-3 bg-slate-50 dark:bg-slate-700/50 rounded-lg">
                  <span className="font-medium">Goats Captured:</span>
                  <span className="font-bold text-orange-600">
                    {gameState.goatsCaptured}/5
                  </span>
                </div>
              </div>

              {/* Game Over Status */}
              {gameState.gameOver && (
                <motion.div
                  initial={{ scale: 0 }}
                  animate={{ scale: 1 }}
                  className={`p-4 rounded-lg text-center font-bold text-lg ${
                    gameState.winner === "tiger"
                      ? "bg-orange-100 dark:bg-orange-900/30 text-orange-600"
                      : "bg-green-100 dark:bg-green-900/30 text-green-600"
                  }`}
                >
                  🏆 {gameState.winner === "tiger" ? "Tigers" : "Goats"} Win!
                </motion.div>
              )}

              {/* Demo Status */}
              {demoPlaying && (
                <div className="text-sm text-center bg-gradient-to-r from-green-100 to-emerald-100 dark:from-green-900/30 dark:to-emerald-900/30 p-3 rounded-lg border border-green-200 dark:border-green-800">
                  <p className="font-bold text-green-700 dark:text-green-300 mb-1">
                    🎬 Demo: Advanced AI vs AI Battle!
                  </p>
                  <p className="text-green-600 dark:text-green-400">
                    {gameState.phase === "placement"
                      ? `Phase 1: Strategic Goat Placement (${gameState.goatsPlaced}/20)`
                      : "Phase 2: Tactical Movement & Captures"}
                  </p>
                  <p className="text-xs text-green-500 dark:text-green-400 mt-1">
                    Move #{moveCount} | Goats Captured:{" "}
                    {gameState.goatsCaptured}/5
                  </p>
                </div>
              )}

              {/* Instructions */}
              {!gameState.gameOver && !demoPlaying && (
                <div className="text-sm text-slate-600 dark:text-slate-400 bg-slate-50 dark:bg-slate-700/50 p-3 rounded-lg">
                  {gameState.phase === "placement" ? (
                    <p>📍 Click on empty intersections to place goats</p>
                  ) : (
                    <p>
                      🎯 Click a piece to select, then click destination to move
                    </p>
                  )}
                </div>
              )}

              {/* Game Phase Progress */}
              {gameState.phase === "placement" && (
                <div className="bg-slate-50 dark:bg-slate-700/50 p-3 rounded-lg">
                  <div className="flex justify-between text-sm text-slate-600 dark:text-slate-400 mb-2">
                    <span>Goat Placement Progress</span>
                    <span>{gameState.goatsPlaced}/20</span>
                  </div>
                  <div className="w-full bg-slate-200 dark:bg-slate-600 rounded-full h-2">
                    <div
                      className="bg-gradient-to-r from-amber-500 to-orange-500 h-2 rounded-full transition-all duration-300"
                      style={{
                        width: `${(gameState.goatsPlaced / 20) * 100}%`,
                      }}
                    ></div>
                  </div>
                </div>
              )}
            </div>
          </div>
        </motion.div>
      </div>
    </div>
  );
};

export default BaghChal;
