import { useState, useEffect, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";

interface Position {
  x: number;
  y: number;
}

interface Dot {
  id: string;
  position: Position;
  type: "normal" | "power" | "strength";
  strength?: {
    title: string;
    content: string;
    icon: string;
  };
}

interface Ghost {
  id: string;
  position: Position;
  direction: Direction;
  mode: "chase" | "flee" | "scatter";
  color: string;
}

type Direction = "up" | "down" | "left" | "right";

// Aakriti's professional strengths revealed through gameplay
const professionalStrengths = [
  {
    title: "AI Innovation",
    content:
      "Founded an AI company transforming low-quality images into professional-grade product shots, helping thousands of MSMEs compete globally.",
    icon: "🤖",
  },
  {
    title: "Meta Engineering",
    content:
      "Engineered sophisticated ML-driven budget pacing & ad delivery systems at Meta, optimizing billions in ad spend.",
    icon: "⚡",
  },
  {
    title: "Recognition by Yann LeCun",
    content:
      "Awarded by Dr. Yann LeCun (Turing Award winner) for developing an engineering-efficient ML solution balancing cost, performance, and accuracy.",
    icon: "🏆",
  },
  {
    title: "Luxury Tech Vision",
    content:
      "Founded Swarnawastra, a luxury fashion-tech house blending AI-driven customization with ultra-rare materials like real gold and lab-grown diamonds.",
    icon: "💎",
  },
  {
    title: "Civic Impact",
    content:
      "Developed face recognition system for Indian Parliament and PPE detection systems for Tata, automating compliance in industrial settings.",
    icon: "🏛️",
  },
  {
    title: "Scale Expertise",
    content:
      "Enhanced search and product discovery at eBay, helping millions find products faster. Built high-volume infrastructure at Yahoo.",
    icon: "🌐",
  },
  {
    title: "Technical Foundation",
    content:
      "B.Tech in Engineering with advanced coursework in machine learning and optimization, building rigorous foundations in algorithms and system design.",
    icon: "🎓",
  },
  {
    title: "Global Vision",
    content:
      "Democratizing advanced AI and luxury design for creators worldwide - from MSMEs to fashion designers keeping their copyrights and legacies.",
    icon: "🌍",
  },
];

// Simple maze layout (1 = wall, 0 = path, 2 = dot, 3 = power pellet, 4 = strength dot)
const maze = [
  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
  [1, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 1],
  [1, 3, 1, 1, 2, 1, 1, 1, 2, 1, 2, 1, 1, 1, 2, 1, 1, 3, 1],
  [1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1],
  [1, 2, 1, 1, 2, 1, 2, 1, 1, 1, 1, 1, 2, 1, 2, 1, 1, 2, 1],
  [1, 2, 2, 2, 2, 1, 2, 2, 2, 1, 2, 2, 2, 1, 2, 2, 2, 2, 1],
  [1, 1, 1, 1, 2, 1, 1, 1, 4, 1, 4, 1, 1, 1, 2, 1, 1, 1, 1],
  [0, 0, 0, 1, 2, 1, 2, 2, 2, 2, 2, 2, 2, 1, 2, 1, 0, 0, 0],
  [1, 1, 1, 1, 2, 1, 2, 1, 0, 0, 0, 1, 2, 1, 2, 1, 1, 1, 1],
  [2, 2, 2, 2, 2, 2, 2, 1, 0, 0, 0, 1, 2, 2, 2, 2, 2, 2, 2],
  [1, 1, 1, 1, 2, 1, 2, 1, 1, 1, 1, 1, 2, 1, 2, 1, 1, 1, 1],
  [0, 0, 0, 1, 2, 1, 2, 2, 2, 2, 2, 2, 2, 1, 2, 1, 0, 0, 0],
  [1, 1, 1, 1, 2, 1, 1, 1, 4, 1, 4, 1, 1, 1, 2, 1, 1, 1, 1],
  [1, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 1],
  [1, 2, 1, 1, 2, 1, 1, 1, 2, 1, 2, 1, 1, 1, 2, 1, 1, 2, 1],
  [1, 3, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 3, 1],
  [1, 1, 2, 1, 2, 1, 2, 1, 1, 1, 1, 1, 2, 1, 2, 1, 2, 1, 1],
  [1, 2, 2, 2, 2, 1, 2, 2, 2, 1, 2, 2, 2, 1, 2, 2, 2, 2, 1],
  [1, 2, 1, 1, 1, 1, 1, 1, 2, 1, 2, 1, 1, 1, 1, 1, 1, 2, 1],
  [1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1],
  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
];

const Pacman = () => {
  const [pacmanPos, setPacmanPos] = useState<Position>({ x: 9, y: 15 });
  const [pacmanDirection, setPacmanDirection] = useState<Direction>("right");
  const [dots, setDots] = useState<Dot[]>([]);
  const [ghosts, setGhosts] = useState<Ghost[]>([]);
  const [score, setScore] = useState(0);
  const [gameStarted, setGameStarted] = useState(false);
  const [revealedStrengths, setRevealedStrengths] = useState<any[]>([]);
  const [currentStrength, setCurrentStrength] = useState<any>(null);
  const [powerMode, setPowerMode] = useState(false);
  const [gameOver, setGameOver] = useState(false);
  const [strengthsCollected, setStrengthsCollected] = useState(0);

  // Initialize dots based on maze
  const initializeDots = useCallback(() => {
    const newDots: Dot[] = [];
    let strengthIndex = 0;

    for (let y = 0; y < maze.length; y++) {
      for (let x = 0; x < maze[y].length; x++) {
        if (maze[y][x] === 2) {
          newDots.push({
            id: `dot-${x}-${y}`,
            position: { x, y },
            type: "normal",
          });
        } else if (maze[y][x] === 3) {
          newDots.push({
            id: `power-${x}-${y}`,
            position: { x, y },
            type: "power",
          });
        } else if (
          maze[y][x] === 4 &&
          strengthIndex < professionalStrengths.length
        ) {
          newDots.push({
            id: `strength-${x}-${y}`,
            position: { x, y },
            type: "strength",
            strength: professionalStrengths[strengthIndex],
          });
          strengthIndex++;
        }
      }
    }
    setDots(newDots);
  }, []);

  // Initialize ghosts
  const initializeGhosts = useCallback(() => {
    setGhosts([
      {
        id: "ghost1",
        position: { x: 9, y: 9 },
        direction: "up",
        mode: "chase",
        color: "red",
      },
      {
        id: "ghost2",
        position: { x: 8, y: 9 },
        direction: "left",
        mode: "chase",
        color: "pink",
      },
      {
        id: "ghost3",
        position: { x: 10, y: 9 },
        direction: "right",
        mode: "chase",
        color: "cyan",
      },
      {
        id: "ghost4",
        position: { x: 9, y: 10 },
        direction: "down",
        mode: "chase",
        color: "orange",
      },
    ]);
  }, []);

  // Check if position is valid (not a wall)
  const isValidPosition = (x: number, y: number): boolean => {
    if (y < 0 || y >= maze.length || x < 0 || x >= maze[0].length) return false;
    return maze[y][x] !== 1;
  };

  // Move pacman
  const movePacman = useCallback((direction: Direction) => {
    setPacmanPos((prev) => {
      let newX = prev.x;
      let newY = prev.y;

      switch (direction) {
        case "up":
          newY--;
          break;
        case "down":
          newY++;
          break;
        case "left":
          newX--;
          break;
        case "right":
          newX++;
          break;
      }

      // Tunnel effect (wrap around)
      if (newX < 0) newX = maze[0].length - 1;
      if (newX >= maze[0].length) newX = 0;

      if (isValidPosition(newX, newY)) {
        setPacmanDirection(direction);
        return { x: newX, y: newY };
      }
      return prev;
    });
  }, []);

  // Handle keyboard input
  useEffect(() => {
    const handleKeyPress = (e: KeyboardEvent) => {
      if (!gameStarted || gameOver) return;

      switch (e.key) {
        case "ArrowUp":
          movePacman("up");
          break;
        case "ArrowDown":
          movePacman("down");
          break;
        case "ArrowLeft":
          movePacman("left");
          break;
        case "ArrowRight":
          movePacman("right");
          break;
      }
    };

    window.addEventListener("keydown", handleKeyPress);
    return () => window.removeEventListener("keydown", handleKeyPress);
  }, [gameStarted, gameOver, movePacman]);

  // Check collisions with dots
  useEffect(() => {
    const collidedDot = dots.find(
      (dot) => dot.position.x === pacmanPos.x && dot.position.y === pacmanPos.y,
    );

    if (collidedDot) {
      setDots((prev) => prev.filter((dot) => dot.id !== collidedDot.id));

      if (collidedDot.type === "normal") {
        setScore((prev) => prev + 10);
      } else if (collidedDot.type === "power") {
        setScore((prev) => prev + 50);
        setPowerMode(true);
        setTimeout(() => setPowerMode(false), 8000);
      } else if (collidedDot.type === "strength" && collidedDot.strength) {
        setScore((prev) => prev + 100);
        setCurrentStrength(collidedDot.strength);
        setRevealedStrengths((prev) => [...prev, collidedDot.strength]);
        setStrengthsCollected((prev) => prev + 1);

        // Auto-hide after 8 seconds
        setTimeout(() => setCurrentStrength(null), 8000);
      }
    }
  }, [pacmanPos, dots]);

  // Simple ghost AI
  useEffect(() => {
    if (!gameStarted || gameOver) return;

    const moveGhosts = () => {
      setGhosts((prev) =>
        prev.map((ghost) => {
          const directions: Direction[] = ["up", "down", "left", "right"];
          const validDirections = directions.filter((dir) => {
            let newX = ghost.position.x;
            let newY = ghost.position.y;

            switch (dir) {
              case "up":
                newY--;
                break;
              case "down":
                newY++;
                break;
              case "left":
                newX--;
                break;
              case "right":
                newX++;
                break;
            }

            return isValidPosition(newX, newY);
          });

          if (validDirections.length > 0) {
            const randomDirection =
              validDirections[
                Math.floor(Math.random() * validDirections.length)
              ];
            let newX = ghost.position.x;
            let newY = ghost.position.y;

            switch (randomDirection) {
              case "up":
                newY--;
                break;
              case "down":
                newY++;
                break;
              case "left":
                newX--;
                break;
              case "right":
                newX++;
                break;
            }

            return {
              ...ghost,
              position: { x: newX, y: newY },
              direction: randomDirection,
            };
          }

          return ghost;
        }),
      );
    };

    const interval = setInterval(moveGhosts, 400);
    return () => clearInterval(interval);
  }, [gameStarted, gameOver]);

  // Check collision with ghosts
  useEffect(() => {
    const collidedGhost = ghosts.find(
      (ghost) =>
        ghost.position.x === pacmanPos.x && ghost.position.y === pacmanPos.y,
    );

    if (collidedGhost && !powerMode) {
      setGameOver(true);
    }
  }, [pacmanPos, ghosts, powerMode]);

  // Check win condition
  useEffect(() => {
    const remainingDots = dots.filter(
      (dot) => dot.type === "normal" || dot.type === "strength",
    );
    if (remainingDots.length === 0 && gameStarted) {
      setGameOver(true);
    }
  }, [dots, gameStarted]);

  const startGame = () => {
    setGameStarted(true);
    setGameOver(false);
    setScore(0);
    setRevealedStrengths([]);
    setCurrentStrength(null);
    setStrengthsCollected(0);
    setPacmanPos({ x: 9, y: 15 });
    setPowerMode(false);
    initializeDots();
    initializeGhosts();
  };

  return (
    <div className="w-full">
      {/* Game Header */}
      <div className="text-center mb-8">
        <h2 className="text-3xl md:text-4xl font-bold bg-gradient-to-r from-yellow-500 via-orange-500 to-red-500 bg-clip-text text-transparent mb-4">
          🟡 Pacman: Discover Aakriti's Strengths
        </h2>
        <p className="text-slate-600 dark:text-slate-300 mb-4">
          Navigate through the maze and collect professional achievements!
        </p>

        <div className="flex flex-wrap justify-center gap-4 mb-6">
          <button
            onClick={startGame}
            className="px-6 py-3 bg-gradient-to-r from-yellow-500 to-orange-500 text-white rounded-lg font-bold shadow-lg hover:scale-105 transition-transform"
          >
            {gameStarted ? "🔄 Restart Game" : "🎮 Start Game"}
          </button>
        </div>

        {/* Score and Stats */}
        <div className="flex justify-center gap-6 text-sm">
          <div className="bg-white/90 dark:bg-slate-800/90 px-4 py-2 rounded-lg">
            <span className="font-bold">Score: {score}</span>
          </div>
          <div className="bg-white/90 dark:bg-slate-800/90 px-4 py-2 rounded-lg">
            <span className="font-bold">Strengths: {strengthsCollected}/8</span>
          </div>
          {powerMode && (
            <div className="bg-yellow-400 text-black px-4 py-2 rounded-lg animate-pulse">
              <span className="font-bold">⚡ POWER MODE!</span>
            </div>
          )}
        </div>
      </div>

      <div className="flex flex-col xl:flex-row justify-center items-center xl:items-start gap-8">
        {/* Game Board */}
        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          className="bg-black p-4 rounded-2xl shadow-2xl border-4 border-blue-600"
        >
          <div
            className="grid grid-cols-19 gap-0"
            style={{ gridTemplateColumns: "repeat(19, 1fr)" }}
          >
            {maze.map((row, y) =>
              row.map((cell, x) => {
                const isDot = dots.some(
                  (dot) => dot.position.x === x && dot.position.y === y,
                );
                const dot = dots.find(
                  (dot) => dot.position.x === x && dot.position.y === y,
                );
                const isPacman = pacmanPos.x === x && pacmanPos.y === y;
                const ghost = ghosts.find(
                  (g) => g.position.x === x && g.position.y === y,
                );

                return (
                  <div
                    key={`${x}-${y}`}
                    className={`w-6 h-6 flex items-center justify-center text-sm ${
                      cell === 1 ? "bg-blue-600" : "bg-black"
                    }`}
                  >
                    {isPacman && (
                      <motion.div
                        animate={{ rotate: [0, 45, 0] }}
                        transition={{ duration: 0.3, repeat: Infinity }}
                        className="text-yellow-400 text-lg"
                      >
                        🟡
                      </motion.div>
                    )}
                    {ghost && !isPacman && (
                      <motion.div
                        animate={{ scale: [1, 1.1, 1] }}
                        transition={{ duration: 0.5, repeat: Infinity }}
                        className={`text-lg ${
                          powerMode
                            ? "text-blue-400"
                            : ghost.color === "red"
                              ? "text-red-500"
                              : ghost.color === "pink"
                                ? "text-pink-500"
                                : ghost.color === "cyan"
                                  ? "text-cyan-500"
                                  : "text-orange-500"
                        }`}
                      >
                        👻
                      </motion.div>
                    )}
                    {isDot && !isPacman && !ghost && (
                      <div
                        className={`${
                          dot?.type === "normal"
                            ? "w-1 h-1 bg-yellow-300 rounded-full"
                            : dot?.type === "power"
                              ? "w-3 h-3 bg-yellow-400 rounded-full animate-pulse"
                              : dot?.type === "strength"
                                ? "text-xs"
                                : ""
                        }`}
                      >
                        {dot?.type === "strength" && "💎"}
                      </div>
                    )}
                  </div>
                );
              }),
            )}
          </div>

          {/* Game Instructions */}
          <div className="mt-4 text-center text-white text-sm">
            {!gameStarted ? (
              <p>
                🎮 Use arrow keys to move • Collect 💎 to reveal Aakriti's
                strengths!
              </p>
            ) : (
              <p>
                ���� Collect dots for points • 💎 Professional achievements • ⚡
                Power pellets
              </p>
            )}
          </div>
        </motion.div>

        {/* Strengths Panel */}
        <motion.div
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          className="w-full xl:w-96 max-w-md"
        >
          <div className="bg-white/90 dark:bg-slate-800/90 backdrop-blur-lg rounded-2xl p-6 shadow-2xl h-96 overflow-y-auto border border-white/20 dark:border-slate-700/20">
            <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-200 mb-4">
              Professional Strengths
            </h3>

            <AnimatePresence mode="wait">
              {currentStrength ? (
                <motion.div
                  key={currentStrength.title}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -20 }}
                  className="p-4 rounded-lg bg-gradient-to-r from-yellow-100 to-orange-100 dark:from-yellow-900/30 dark:to-orange-900/30 border border-yellow-200 dark:border-yellow-800"
                >
                  <div className="flex items-center gap-3 mb-3">
                    <span className="text-3xl">{currentStrength.icon}</span>
                    <h4 className="font-bold text-lg text-yellow-800 dark:text-yellow-200">
                      {currentStrength.title}
                    </h4>
                  </div>
                  <p className="text-slate-700 dark:text-slate-300 leading-relaxed">
                    {currentStrength.content}
                  </p>
                </motion.div>
              ) : gameOver ? (
                <motion.div
                  initial={{ opacity: 0, scale: 0.8 }}
                  animate={{ opacity: 1, scale: 1 }}
                  className="text-center py-8"
                >
                  <div className="text-6xl mb-4">
                    {strengthsCollected === 8 ? "🏆" : "👻"}
                  </div>
                  <h4 className="text-xl font-bold mb-2">
                    {strengthsCollected === 8
                      ? "All Strengths Discovered!"
                      : "Game Over!"}
                  </h4>
                  <p className="text-slate-600 dark:text-slate-400">
                    You discovered {strengthsCollected}/8 professional strengths
                  </p>
                  <p className="text-sm text-slate-500 dark:text-slate-400 mt-2">
                    Final Score: {score}
                  </p>
                </motion.div>
              ) : (
                <motion.div
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  className="text-center text-slate-500 dark:text-slate-400 py-12"
                >
                  <div className="text-6xl mb-4">🟡</div>
                  <p className="text-lg font-medium">
                    Collect 💎 diamonds to reveal
                  </p>
                  <p className="text-sm">Aakriti's professional achievements</p>
                  {gameStarted && (
                    <div className="mt-6 text-xs text-slate-400 space-y-1">
                      <p>• 🟡 Regular dots = 10 points</p>
                      <p>• ⚡ Power pellets = 50 points</p>
                      <p>• 💎 Strength diamonds = 100 points + story</p>
                    </div>
                  )}
                </motion.div>
              )}
            </AnimatePresence>
          </div>

          {/* Collected Strengths */}
          {revealedStrengths.length > 0 && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              className="mt-4 bg-white/90 dark:bg-slate-800/90 backdrop-blur-lg rounded-xl p-4 shadow-xl border border-white/20 dark:border-slate-700/20"
            >
              <h4 className="font-semibold text-slate-700 dark:text-slate-300 mb-2">
                Discovered Strengths ({revealedStrengths.length}/8)
              </h4>
              <div className="grid grid-cols-4 gap-2">
                {revealedStrengths.map((strength, index) => (
                  <motion.div
                    key={index}
                    initial={{ scale: 0, rotate: 180 }}
                    animate={{ scale: 1, rotate: 0 }}
                    className="text-2xl text-center p-2 bg-yellow-100 dark:bg-yellow-900/30 rounded-lg"
                    title={strength.title}
                  >
                    {strength.icon}
                  </motion.div>
                ))}
              </div>
            </motion.div>
          )}
        </motion.div>
      </div>
    </div>
  );
};

export default Pacman;
