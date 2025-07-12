import { useState, useEffect, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";

interface Position {
  x: number;
  y: number;
}

interface Obstacle extends Position {
  width: number;
  height: number;
  type: "building" | "cloud" | "mountain";
}

interface Collectible extends Position {
  type: "gold" | "diamond" | "award";
  story?: string;
  value: number;
}

const CANVAS_WIDTH = 800;
const CANVAS_HEIGHT = 400;
const HELICOPTER_SIZE = 40;
const GAME_SPEED = 2;

// Aakriti's achievements revealed through collectibles
const ACHIEVEMENT_COLLECTIBLES: Array<{
  type: Collectible["type"];
  story: string;
  value: number;
}> = [
  {
    type: "gold",
    story:
      "🥇 Meta Engineering Excellence: Scaled ML systems serving billions of users with minimal latency",
    value: 100,
  },
  {
    type: "diamond",
    story:
      "💎 Yann LeCun Recognition: Acknowledged by Turing Award winner at ICLR 2019 for innovative research",
    value: 150,
  },
  {
    type: "award",
    story:
      "🏆 eBay Leadership: Led critical infrastructure projects handling millions of transactions daily",
    value: 200,
  },
  {
    type: "gold",
    story:
      "🥇 Swarnawastra Vision: Founded luxury-tech company democratizing high-end fashion through AI",
    value: 100,
  },
  {
    type: "diamond",
    story:
      "💎 Academic Excellence: Top 1% AIEEE, Rank 300 IPU-CET, graduated from premier institutions",
    value: 150,
  },
  {
    type: "award",
    story:
      "🏆 Geographic Journey: Delhi → Bhubaneshwar → Bangalore → NYC → LA → Silicon Valley",
    value: 200,
  },
  {
    type: "gold",
    story:
      "🥇 Multifaceted Excellence: Equestrian, Aviator, Marksman, Motorcyclist, and Pianist",
    value: 100,
  },
  {
    type: "diamond",
    story:
      "💎 Innovation Impact: Parliament security systems, Tata PPE detection, Yahoo foundations",
    value: 150,
  },
];

export default function Helicopter() {
  const [helicopterY, setHelicopterY] = useState(CANVAS_HEIGHT / 2);
  const [obstacles, setObstacles] = useState<Obstacle[]>([]);
  const [collectibles, setCollectibles] = useState<Collectible[]>([]);
  const [gameStarted, setGameStarted] = useState(false);
  const [gameOver, setGameOver] = useState(false);
  const [isPaused, setIsPaused] = useState(false);
  const [score, setScore] = useState(0);
  const [distance, setDistance] = useState(0);
  const [discoveredStories, setDiscoveredStories] = useState<string[]>([]);
  const [currentStory, setCurrentStory] = useState<string>("");
  const [isAscending, setIsAscending] = useState(false);

  const resetGame = () => {
    setHelicopterY(CANVAS_HEIGHT / 2);
    setObstacles([]);
    setCollectibles([]);
    setGameOver(false);
    setGameStarted(true);
    setIsPaused(false);
    setScore(0);
    setDistance(0);
    setDiscoveredStories([]);
    setCurrentStory("");
    setIsAscending(false);
  };

  const startGame = () => {
    resetGame();
  };

  const generateObstacle = useCallback((): Obstacle => {
    const types: Obstacle["type"][] = ["building", "cloud", "mountain"];
    const type = types[Math.floor(Math.random() * types.length)];

    let height: number;
    switch (type) {
      case "building":
        height = 100 + Math.random() * 150;
        break;
      case "cloud":
        height = 80 + Math.random() * 100;
        break;
      case "mountain":
        height = 120 + Math.random() * 180;
        break;
    }

    return {
      x: CANVAS_WIDTH,
      y: CANVAS_HEIGHT - height,
      width: 60 + Math.random() * 40,
      height,
      type,
    };
  }, []);

  const generateCollectible = useCallback((): Collectible => {
    const achievement =
      ACHIEVEMENT_COLLECTIBLES[
        Math.floor(Math.random() * ACHIEVEMENT_COLLECTIBLES.length)
      ];

    return {
      x: CANVAS_WIDTH + Math.random() * 200,
      y: 50 + Math.random() * (CANVAS_HEIGHT - 100),
      type: achievement.type,
      story: achievement.story,
      value: achievement.value,
    };
  }, []);

  const handleKeyPress = useCallback(
    (e: KeyboardEvent) => {
      if (!gameStarted || gameOver) return;

      switch (e.key) {
        case " ":
        case "ArrowUp":
          e.preventDefault();
          setIsAscending(true);
          break;
        case "p":
        case "P":
          e.preventDefault();
          setIsPaused((prev) => !prev);
          break;
      }
    },
    [gameStarted, gameOver],
  );

  const handleKeyUp = useCallback((e: KeyboardEvent) => {
    if (e.key === " " || e.key === "ArrowUp") {
      setIsAscending(false);
    }
  }, []);

  useEffect(() => {
    window.addEventListener("keydown", handleKeyPress);
    window.addEventListener("keyup", handleKeyUp);
    return () => {
      window.removeEventListener("keydown", handleKeyPress);
      window.removeEventListener("keyup", handleKeyUp);
    };
  }, [handleKeyPress, handleKeyUp]);

  // Game physics loop
  useEffect(() => {
    if (!gameStarted || gameOver || isPaused) return;

    const gameLoop = setInterval(() => {
      // Update helicopter position
      setHelicopterY((prev) => {
        let newY = prev;
        if (isAscending) {
          newY -= 4; // Ascend when key is pressed
        } else {
          newY += 2; // Fall due to gravity
        }

        // Keep helicopter within bounds
        return Math.max(0, Math.min(CANVAS_HEIGHT - HELICOPTER_SIZE, newY));
      });

      // Update distance
      setDistance((prev) => prev + 1);

      // Move and update obstacles
      setObstacles((prev) => {
        let newObstacles = prev
          .map((obstacle) => ({ ...obstacle, x: obstacle.x - GAME_SPEED }))
          .filter((obstacle) => obstacle.x + obstacle.width > -50);

        // Add new obstacles
        if (
          newObstacles.length === 0 ||
          newObstacles[newObstacles.length - 1].x < CANVAS_WIDTH - 200
        ) {
          if (Math.random() < 0.3) {
            newObstacles.push(generateObstacle());
          }
        }

        return newObstacles;
      });

      // Move and update collectibles
      setCollectibles((prev) => {
        let newCollectibles = prev
          .map((collectible) => ({
            ...collectible,
            x: collectible.x - GAME_SPEED,
          }))
          .filter((collectible) => collectible.x > -30);

        // Add new collectibles
        if (Math.random() < 0.02) {
          newCollectibles.push(generateCollectible());
        }

        return newCollectibles;
      });

      // Check collision with ground or ceiling
      if (helicopterY <= 0 || helicopterY >= CANVAS_HEIGHT - HELICOPTER_SIZE) {
        setGameOver(true);
        return;
      }

      // Check collision with obstacles
      const helicopterX = 100;
      obstacles.forEach((obstacle) => {
        if (
          helicopterX + HELICOPTER_SIZE > obstacle.x &&
          helicopterX < obstacle.x + obstacle.width &&
          helicopterY + HELICOPTER_SIZE > obstacle.y &&
          helicopterY < obstacle.y + obstacle.height
        ) {
          setGameOver(true);
        }
      });

      // Check collision with collectibles
      collectibles.forEach((collectible, index) => {
        if (
          helicopterX + HELICOPTER_SIZE > collectible.x &&
          helicopterX < collectible.x + 30 &&
          helicopterY + HELICOPTER_SIZE > collectible.y &&
          helicopterY < collectible.y + 30
        ) {
          setScore((prev) => prev + collectible.value);
          if (collectible.story) {
            setCurrentStory(collectible.story);
            setDiscoveredStories((prev) => [...prev, collectible.story!]);
            setTimeout(() => setCurrentStory(""), 4000);
          }
          setCollectibles((prev) => prev.filter((_, i) => i !== index));
        }
      });
    }, 16); // ~60 FPS

    return () => clearInterval(gameLoop);
  }, [
    gameStarted,
    gameOver,
    isPaused,
    helicopterY,
    isAscending,
    obstacles,
    collectibles,
    generateObstacle,
    generateCollectible,
  ]);

  const getObstacleColor = (type: Obstacle["type"]) => {
    switch (type) {
      case "building":
        return "bg-gradient-to-t from-gray-800 to-gray-600 border-gray-500";
      case "cloud":
        return "bg-gradient-to-t from-white/80 to-white/60 border-white/40";
      case "mountain":
        return "bg-gradient-to-t from-stone-800 to-stone-600 border-stone-500";
    }
  };

  const getCollectibleIcon = (type: Collectible["type"]) => {
    switch (type) {
      case "gold":
        return "🥇";
      case "diamond":
        return "💎";
      case "award":
        return "🏆";
    }
  };

  return (
    <div className="max-w-7xl mx-auto">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="text-center mb-8"
      >
        <h2 className="tom-ford-heading text-4xl text-white mb-4">
          HELICOPTER
          <br />
          <span className="gold-shimmer">MASTERY</span>
        </h2>
        <p className="tom-ford-subheading text-white/60 text-lg tracking-wider max-w-3xl mx-auto">
          NAVIGATE THROUGH CHALLENGES TO DISCOVER PROFESSIONAL ACHIEVEMENTS
        </p>
      </motion.div>

      <div className="grid lg:grid-cols-3 gap-8">
        {/* Game Board */}
        <div className="lg:col-span-2">
          <motion.div
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            className="tom-ford-glass rounded-sm p-6 border border-white/10"
          >
            <div className="flex justify-between items-center mb-6">
              <div className="flex gap-8">
                <div className="text-white font-light">
                  <span className="tom-ford-subheading text-yellow-400 text-xs tracking-wider">
                    SCORE
                  </span>
                  <div className="text-2xl">{score}</div>
                </div>
                <div className="text-white font-light">
                  <span className="tom-ford-subheading text-yellow-400 text-xs tracking-wider">
                    DISTANCE
                  </span>
                  <div className="text-2xl">{Math.floor(distance / 10)}m</div>
                </div>
              </div>
              <div className="flex gap-3">
                {!gameStarted ? (
                  <motion.button
                    onClick={startGame}
                    whileHover={{ scale: 1.02 }}
                    whileTap={{ scale: 0.98 }}
                    className="tom-ford-button px-6 py-3 rounded-sm text-white font-light tracking-wider"
                  >
                    BEGIN FLIGHT
                  </motion.button>
                ) : (
                  <>
                    <motion.button
                      onClick={() => setIsPaused(!isPaused)}
                      whileHover={{ scale: 1.02 }}
                      whileTap={{ scale: 0.98 }}
                      className="border border-white/30 text-white px-4 py-2 rounded-sm font-light tracking-wider hover:border-yellow-400 hover:text-yellow-400 transition-all duration-300"
                    >
                      {isPaused ? "◆ RESUME" : "◇ PAUSE"}
                    </motion.button>
                    <motion.button
                      onClick={resetGame}
                      whileHover={{ scale: 1.02 }}
                      whileTap={{ scale: 0.98 }}
                      className="border border-white/30 text-white px-4 py-2 rounded-sm font-light tracking-wider hover:border-yellow-400 hover:text-yellow-400 transition-all duration-300"
                    >
                      ◈ RESTART
                    </motion.button>
                  </>
                )}
              </div>
            </div>

            {/* Game Canvas */}
            <div
              className="relative mx-auto bg-gradient-to-b from-blue-900/20 to-blue-800/20 border border-white/10 rounded-sm overflow-hidden"
              style={{ width: CANVAS_WIDTH, height: CANVAS_HEIGHT }}
            >
              {/* Helicopter */}
              <motion.div
                className="absolute text-3xl z-20"
                style={{
                  left: 100,
                  top: helicopterY,
                  width: HELICOPTER_SIZE,
                  height: HELICOPTER_SIZE,
                }}
                animate={{
                  rotate: isAscending ? -10 : 5,
                }}
                transition={{ duration: 0.1 }}
              >
                🚁
              </motion.div>

              {/* Obstacles */}
              {obstacles.map((obstacle, index) => (
                <motion.div
                  key={index}
                  className={`absolute rounded-sm border-2 ${getObstacleColor(obstacle.type)}`}
                  style={{
                    left: obstacle.x,
                    top: obstacle.y,
                    width: obstacle.width,
                    height: obstacle.height,
                  }}
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  transition={{ duration: 0.2 }}
                />
              ))}

              {/* Collectibles */}
              {collectibles.map((collectible, index) => (
                <motion.div
                  key={index}
                  className="absolute text-2xl z-10"
                  style={{
                    left: collectible.x,
                    top: collectible.y,
                    width: 30,
                    height: 30,
                  }}
                  animate={{
                    scale: [1, 1.2, 1],
                    rotate: [0, 10, -10, 0],
                  }}
                  transition={{
                    duration: 2,
                    repeat: Infinity,
                    ease: "easeInOut",
                  }}
                >
                  {getCollectibleIcon(collectible.type)}
                </motion.div>
              ))}

              {/* Ground and ceiling indicators */}
              <div className="absolute bottom-0 left-0 right-0 h-1 bg-gradient-to-r from-yellow-400/50 to-yellow-400/30" />
              <div className="absolute top-0 left-0 right-0 h-1 bg-gradient-to-r from-yellow-400/50 to-yellow-400/30" />

              {/* Game Over Overlay */}
              <AnimatePresence>
                {gameOver && (
                  <motion.div
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                    className="absolute inset-0 bg-black/80 backdrop-blur-sm flex items-center justify-center rounded-sm"
                  >
                    <div className="text-center text-white">
                      <div className="text-6xl mb-6 text-yellow-400">◆</div>
                      <div className="tom-ford-heading text-3xl mb-4">
                        MISSION COMPLETE
                      </div>
                      <div className="tom-ford-subheading text-lg mb-2 text-yellow-400">
                        FINAL SCORE: {score}
                      </div>
                      <div className="text-white/60 mb-8">
                        Distance: {Math.floor(distance / 10)}m
                      </div>
                      <motion.button
                        onClick={resetGame}
                        whileHover={{ scale: 1.02 }}
                        whileTap={{ scale: 0.98 }}
                        className="tom-ford-button px-8 py-4 rounded-sm text-white font-light tracking-wider"
                      >
                        NEW MISSION
                      </motion.button>
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>

              {/* Pause Overlay */}
              <AnimatePresence>
                {isPaused && !gameOver && (
                  <motion.div
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                    className="absolute inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center rounded-sm"
                  >
                    <div className="text-center text-white">
                      <div className="text-6xl mb-6 text-yellow-400">◇</div>
                      <div className="tom-ford-heading text-2xl mb-4">
                        FLIGHT PAUSED
                      </div>
                      <div className="tom-ford-subheading text-sm text-white/60">
                        PRESS P TO CONTINUE
                      </div>
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>
            </div>

            {/* Controls Info */}
            <div className="mt-6 text-center text-white/60 font-light">
              <div className="tom-ford-subheading text-xs tracking-wider">
                HOLD SPACE OR ↑ TO ASCEND • P TO PAUSE
              </div>
            </div>
          </motion.div>
        </div>

        {/* Achievement Panel */}
        <div className="lg:col-span-1">
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            className="tom-ford-glass rounded-sm p-6 border border-white/10 h-fit"
          >
            <h3 className="tom-ford-heading text-2xl text-white mb-6 text-center">
              PROFESSIONAL
              <br />
              <span className="gold-shimmer">ACHIEVEMENTS</span>
            </h3>

            {/* Current Story */}
            <AnimatePresence>
              {currentStory && (
                <motion.div
                  initial={{ opacity: 0, y: 20, scale: 0.9 }}
                  animate={{ opacity: 1, y: 0, scale: 1 }}
                  exit={{ opacity: 0, y: -20, scale: 0.9 }}
                  className="mb-6 p-4 bg-gradient-to-r from-yellow-400/10 to-yellow-400/5 border border-yellow-400/30 rounded-sm"
                >
                  <div className="tom-ford-subheading text-yellow-400 text-xs tracking-wider mb-2">
                    NEW ACHIEVEMENT UNLOCKED
                  </div>
                  <div className="text-white text-sm font-light leading-relaxed">
                    {currentStory}
                  </div>
                </motion.div>
              )}
            </AnimatePresence>

            {/* Discovered Stories List */}
            <div className="space-y-3 max-h-96 overflow-y-auto">
              {discoveredStories.length === 0 ? (
                <div className="text-center text-white/60 py-8">
                  <div className="text-6xl mb-4 text-yellow-400">◈</div>
                  <div className="tom-ford-subheading text-lg mb-2 text-white">
                    ACHIEVEMENTS AWAIT
                  </div>
                  <div className="text-sm font-light">
                    Collect gold, diamonds, and awards to discover professional
                    milestones
                  </div>
                </div>
              ) : (
                discoveredStories.map((story, index) => (
                  <motion.div
                    key={index}
                    initial={{ opacity: 0, x: 20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: index * 0.1 }}
                    className="p-3 bg-white/5 backdrop-blur-sm border border-white/10 rounded-sm"
                  >
                    <div className="text-white/80 text-sm font-light leading-relaxed">
                      {story}
                    </div>
                  </motion.div>
                ))
              )}
            </div>

            {/* Progress */}
            <div className="mt-6 pt-4 border-t border-white/10">
              <div className="text-center text-white/60">
                <div className="tom-ford-subheading text-xs tracking-wider">
                  ACHIEVEMENTS DISCOVERED: {discoveredStories.length}/
                  {ACHIEVEMENT_COLLECTIBLES.length}
                </div>
                <div className="w-full bg-white/10 rounded-full h-1 mt-3">
                  <motion.div
                    className="bg-gradient-to-r from-yellow-400 to-yellow-600 h-1 rounded-full"
                    initial={{ width: 0 }}
                    animate={{
                      width: `${(discoveredStories.length / ACHIEVEMENT_COLLECTIBLES.length) * 100}%`,
                    }}
                    transition={{ duration: 0.5 }}
                  />
                </div>
              </div>
            </div>
          </motion.div>
        </div>
      </div>
    </div>
  );
}
