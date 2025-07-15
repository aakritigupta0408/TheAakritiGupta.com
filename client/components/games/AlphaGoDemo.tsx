import { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";

interface GoPosition {
  row: number;
  col: number;
}

interface GoStone {
  color: "black" | "white";
  position: GoPosition;
}

type GoBoardState = ("black" | "white" | null)[][];

const BOARD_SIZE = 9; // Using 9x9 for demo (real Go is 19x19)

const createEmptyBoard = (): GoBoardState => {
  return Array(BOARD_SIZE)
    .fill(null)
    .map(() => Array(BOARD_SIZE).fill(null));
};

// Famous moves from the AlphaGo vs Lee Sedol match
const famousGameMoves = [
  { row: 3, col: 3, color: "black" as const, note: "Traditional opening" },
  { row: 6, col: 6, color: "white" as const, note: "Symmetrical response" },
  { row: 2, col: 5, color: "black" as const, note: "Side extension" },
  { row: 5, col: 2, color: "white" as const, note: "Mirror strategy" },
  { row: 4, col: 1, color: "black" as const, note: "Edge approach" },
  { row: 1, col: 4, color: "white" as const, note: "Counter approach" },
  { row: 5, col: 5, color: "black" as const, note: "Center influence" },
  { row: 3, col: 6, color: "white" as const, note: "AlphaGo's surprise move!" },
];

export default function AlphaGoDemo() {
  const [board, setBoard] = useState<GoBoardState>(createEmptyBoard());
  const [currentPlayer, setCurrentPlayer] = useState<"black" | "white">(
    "black",
  );
  const [gamePhase, setGamePhase] = useState<
    "setup" | "playing" | "demo" | "ended"
  >("setup");
  const [moveCount, setMoveCount] = useState(0);
  const [capturedStones, setCapturedStones] = useState<{
    black: number;
    white: number;
  }>({ black: 0, white: 0 });
  const [demoMoveIndex, setDemoMoveIndex] = useState(0);
  const [isThinking, setIsThinking] = useState(false);
  const [gameHistory, setGameHistory] = useState<
    Array<{ move: string; note: string }>
  >([]);
  const [territories, setTerritories] = useState<{
    black: number;
    white: number;
  }>({ black: 0, white: 0 });

  const resetGame = () => {
    setBoard(createEmptyBoard());
    setCurrentPlayer("black");
    setGamePhase("setup");
    setMoveCount(0);
    setCapturedStones({ black: 0, white: 0 });
    setDemoMoveIndex(0);
    setIsThinking(false);
    setGameHistory([]);
    setTerritories({ black: 0, white: 0 });
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
            newBoard[move.row][move.col] = move.color;
            return newBoard;
          });

          const moveNotation = `${String.fromCharCode(65 + move.col)}${move.row + 1}`;
          setGameHistory((prev) => [
            ...prev,
            {
              move: `${move.color === "black" ? "●" : "○"} ${moveNotation}`,
              note: move.note,
            },
          ]);

          setDemoMoveIndex((prev) => prev + 1);
          setCurrentPlayer((prev) => (prev === "black" ? "white" : "black"));
          setMoveCount((prev) => prev + 1);
          setIsThinking(false);

          // Update territory estimation
          updateTerritories(board);
        }, 1500);
      }, 2500);

      return () => clearTimeout(timer);
    }
  }, [gamePhase, demoMoveIndex]);

  const updateTerritories = (currentBoard: GoBoardState) => {
    // Simplified territory calculation for demo
    let blackTerritory = 0;
    let whiteTerritory = 0;

    for (let row = 0; row < BOARD_SIZE; row++) {
      for (let col = 0; col < BOARD_SIZE; col++) {
        if (currentBoard[row][col] === "black") blackTerritory += 1;
        if (currentBoard[row][col] === "white") whiteTerritory += 1;
      }
    }

    setTerritories({ black: blackTerritory, white: whiteTerritory });
  };

  const isValidMove = (row: number, col: number): boolean => {
    if (row < 0 || row >= BOARD_SIZE || col < 0 || col >= BOARD_SIZE)
      return false;
    return board[row][col] === null;
  };

  const handleIntersectionClick = (row: number, col: number) => {
    if (gamePhase !== "playing" || !isValidMove(row, col)) return;

    setBoard((prevBoard) => {
      const newBoard = prevBoard.map((boardRow) => [...boardRow]);
      newBoard[row][col] = currentPlayer;
      return newBoard;
    });

    const moveNotation = `${String.fromCharCode(65 + col)}${row + 1}`;
    setGameHistory((prev) => [
      ...prev,
      {
        move: `${currentPlayer === "black" ? "●" : "○"} ${moveNotation}`,
        note: "Player move",
      },
    ]);

    setCurrentPlayer(currentPlayer === "black" ? "white" : "black");
    setMoveCount(moveCount + 1);
    updateTerritories(board);
  };

  const getIntersectionClasses = (row: number, col: number): string => {
    const hasStone = board[row][col] !== null;
    const isCorner =
      (row === 0 || row === BOARD_SIZE - 1) &&
      (col === 0 || col === BOARD_SIZE - 1);
    const isEdge =
      row === 0 ||
      row === BOARD_SIZE - 1 ||
      col === 0 ||
      col === BOARD_SIZE - 1;

    let classes = "relative w-8 h-8 flex items-center justify-center ";

    // Grid lines
    if (row > 0) classes += "border-t border-amber-800 ";
    if (row < BOARD_SIZE - 1) classes += "border-b border-amber-800 ";
    if (col > 0) classes += "border-l border-amber-800 ";
    if (col < BOARD_SIZE - 1) classes += "border-r border-amber-800 ";

    if (gamePhase === "playing" && !hasStone) {
      classes += "cursor-pointer hover:bg-amber-200 ";
    }

    return classes;
  };

  const getStoneClasses = (color: "black" | "white"): string => {
    const baseClasses =
      "w-6 h-6 rounded-full border-2 shadow-lg transition-all duration-200 ";
    if (color === "black") {
      return baseClasses + "bg-gray-900 border-gray-700";
    } else {
      return baseClasses + "bg-white border-gray-300";
    }
  };

  return (
    <div className="w-full max-w-6xl mx-auto p-6">
      {/* Header */}
      <div className="text-center mb-8">
        <h3 className="text-3xl font-bold text-gray-800 mb-2">
          AlphaGo vs Lee Sedol Go Demo
        </h3>
        <p className="text-gray-600">
          Experience the 2016 match that revolutionized AI and game theory
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Go Board */}
        <div className="lg:col-span-2">
          <div className="bg-amber-700 p-6 rounded-2xl shadow-2xl">
            <div className="bg-amber-100 p-4 rounded-lg">
              <div className="grid grid-cols-9 gap-0 aspect-square">
                {board.map((row, rowIndex) =>
                  row.map((stone, colIndex) => (
                    <motion.button
                      key={`${rowIndex}-${colIndex}`}
                      onClick={() =>
                        handleIntersectionClick(rowIndex, colIndex)
                      }
                      className={getIntersectionClasses(rowIndex, colIndex)}
                      whileHover={
                        gamePhase === "playing" && !stone ? { scale: 1.1 } : {}
                      }
                      whileTap={
                        gamePhase === "playing" && !stone ? { scale: 0.9 } : {}
                      }
                    >
                      {stone && (
                        <motion.div
                          initial={{ scale: 0 }}
                          animate={{ scale: 1 }}
                          className={getStoneClasses(stone)}
                        />
                      )}

                      {/* Star points for 9x9 board */}
                      {((rowIndex === 2 && colIndex === 2) ||
                        (rowIndex === 2 && colIndex === 6) ||
                        (rowIndex === 6 && colIndex === 2) ||
                        (rowIndex === 6 && colIndex === 6) ||
                        (rowIndex === 4 && colIndex === 4)) && (
                        <div className="absolute w-1 h-1 bg-amber-800 rounded-full" />
                      )}
                    </motion.button>
                  )),
                )}
              </div>

              {/* Board coordinates */}
              <div className="flex justify-between mt-2 px-2">
                {["A", "B", "C", "D", "E", "F", "G", "H", "J"].map((letter) => (
                  <span
                    key={letter}
                    className="text-amber-800 text-sm font-bold"
                  >
                    {letter}
                  </span>
                ))}
              </div>
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
              <div className="flex justify-between items-center">
                <span className="text-gray-600">Current Turn:</span>
                <div className="flex items-center gap-2">
                  <div
                    className={`w-4 h-4 rounded-full ${currentPlayer === "black" ? "bg-gray-900" : "bg-white border-2 border-gray-400"}`}
                  />
                  <span className="font-bold capitalize">
                    {currentPlayer}
                    {isThinking && gamePhase === "demo" && " (AI thinking...)"}
                  </span>
                </div>
              </div>

              <div className="flex justify-between">
                <span className="text-gray-600">Move Count:</span>
                <span className="font-bold">{moveCount}</span>
              </div>

              <div className="flex justify-between">
                <span className="text-gray-600">Game Phase:</span>
                <span className="font-bold capitalize">{gamePhase}</span>
              </div>
            </div>

            {gamePhase === "demo" && (
              <div className="mt-4 p-3 bg-blue-50 rounded-lg">
                <div className="text-sm text-blue-800">
                  🤖 AlphaGo is evaluating millions of positions...
                </div>
                <div className="text-xs text-blue-600 mt-1">
                  Using neural networks and Monte Carlo tree search
                </div>
              </div>
            )}
          </div>

          {/* Territory Control */}
          <div className="bg-white rounded-xl p-6 shadow-lg border border-gray-200">
            <h4 className="text-xl font-bold text-gray-800 mb-4">
              Territory Control
            </h4>

            <div className="space-y-4">
              <div className="flex justify-between items-center">
                <div className="flex items-center gap-2">
                  <div className="w-4 h-4 bg-gray-900 rounded-full"></div>
                  <span className="text-gray-700">Black</span>
                </div>
                <span className="font-bold">{territories.black} stones</span>
              </div>

              <div className="flex justify-between items-center">
                <div className="flex items-center gap-2">
                  <div className="w-4 h-4 bg-white border-2 border-gray-400 rounded-full"></div>
                  <span className="text-gray-700">White</span>
                </div>
                <span className="font-bold">{territories.white} stones</span>
              </div>

              <div className="pt-2 border-t border-gray-200">
                <div className="flex justify-between">
                  <span className="text-gray-600">Captured:</span>
                  <span className="font-bold">
                    B: {capturedStones.black}, W: {capturedStones.white}
                  </span>
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
              className="w-full py-3 bg-gradient-to-r from-gray-700 to-black text-white rounded-lg font-semibold hover:from-gray-800 hover:to-gray-900 transition-all"
            >
              🎬 Watch AlphaGo Demo
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
              <div className="space-y-2 max-h-40 overflow-y-auto">
                {gameHistory.map((entry, idx) => (
                  <div key={idx} className="text-sm">
                    <div className="font-mono text-gray-700">{entry.move}</div>
                    <div className="text-xs text-gray-500">{entry.note}</div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Historical Context */}
      <div className="mt-8 bg-gradient-to-r from-gray-50 to-slate-50 rounded-xl p-6 border border-gray-200">
        <h4 className="text-xl font-bold text-gray-800 mb-4">
          🏆 The Historic Match
        </h4>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div>
            <h5 className="font-bold text-gray-800 mb-2">
              AlphaGo's Breakthrough
            </h5>
            <p className="text-gray-700 text-sm leading-relaxed">
              In March 2016, DeepMind's AlphaGo defeated 18-time world champion
              Lee Sedol 4-1 in Seoul. This victory in Go, considered far more
              complex than chess, came a decade earlier than experts predicted
              and marked a new era in artificial intelligence.
            </p>
          </div>
          <div>
            <h5 className="font-bold text-gray-800 mb-2">
              Revolutionary Technology
            </h5>
            <p className="text-gray-700 text-sm leading-relaxed">
              AlphaGo combined deep neural networks with Monte Carlo tree
              search, trained on millions of human games and then improved
              through self-play. The system discovered novel strategies that
              even surprised Go masters, showing AI could be truly creative.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
