import { useState, useEffect, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";

interface Position {
  x: number;
  y: number;
}

interface SnakeSegment extends Position {}

interface Food extends Position {
  type: "apple" | "star" | "flower";
  story?: string;
}

type Direction = "UP" | "DOWN" | "LEFT" | "RIGHT";

const GRID_SIZE = 20;
const CANVAS_SIZE = 400;

// Aakriti's professional milestones revealed through special foods
const STORY_FOODS: Array<{ type: Food["type"]; story: string; emoji: string }> =
  [
    {
      type: "apple",
      story:
        "🍎 Meta Engineering Excellence: Built ML-driven ad systems serving billions of users worldwide",
      emoji: "🍎",
    },
    {
      type: "star",
      story:
        "⭐ Yann LeCun Recognition: Acknowledged by Turing Award winner for innovative AI contributions",
      emoji: "⭐",
    },
    {
      type: "flower",
      story:
        "🌸 Swarnawastra Vision: Luxury fashion-tech founder democratizing designer access through AI",
      emoji: "🌸",
    },
    {
      type: "apple",
      story:
        "🍎 eBay Infrastructure: Scaled e-commerce systems handling millions of daily transactions",
      emoji: "🍎",
    },
    {
      type: "star",
      story:
        "⭐ Parliament Innovation: Developed face recognition systems for government security",
      emoji: "⭐",
    },
    {
      type: "flower",
      story:
        "🌸 Tata Impact: Created PPE detection systems enhancing workplace safety",
      emoji: "🌸",
    },
    {
      type: "apple",
      story:
        "🍎 Yahoo Foundation: Early engineering experience in large-scale web technologies",
      emoji: "🍎",
    },
    {
      type: "star",
      story:
        "⭐ AI Innovation: Founded AI company transforming product image enhancement",
      emoji: "⭐",
    },
  ];

export default function Snake() {
  const [snake, setSnake] = useState<SnakeSegment[]>([
    { x: 10, y: 10 },
    { x: 9, y: 10 },
    { x: 8, y: 10 },
  ]);
  const [food, setFood] = useState<Food>({ x: 15, y: 15, type: "apple" });
  const [direction, setDirection] = useState<Direction>("RIGHT");
  const [gameOver, setGameOver] = useState(false);
  const [gameStarted, setGameStarted] = useState(false);
  const [score, setScore] = useState(0);
  const [discoveredStories, setDiscoveredStories] = useState<string[]>([]);
  const [currentStory, setCurrentStory] = useState<string>("");
  const [isPaused, setIsPaused] = useState(false);

  const generateFood = useCallback((): Food => {
    const storyFood =
      STORY_FOODS[Math.floor(Math.random() * STORY_FOODS.length)];
    let newFood: Food;

    do {
      newFood = {
        x: Math.floor(Math.random() * GRID_SIZE),
        y: Math.floor(Math.random() * GRID_SIZE),
        type: storyFood.type,
        story: storyFood.story,
      };
    } while (
      snake.some(
        (segment) => segment.x === newFood.x && segment.y === newFood.y,
      )
    );

    return newFood;
  }, [snake]);

  const resetGame = () => {
    setSnake([
      { x: 10, y: 10 },
      { x: 9, y: 10 },
      { x: 8, y: 10 },
    ]);
    setDirection("RIGHT");
    setGameOver(false);
    setGameStarted(true);
    setScore(0);
    setDiscoveredStories([]);
    setCurrentStory("");
    setIsPaused(false);
    setFood(generateFood());
  };

  const startGame = () => {
    resetGame();
  };

  const handleKeyPress = useCallback(
    (e: KeyboardEvent) => {
      if (!gameStarted || gameOver) return;

      switch (e.key) {
        case "ArrowUp":
          e.preventDefault();
          setDirection((prev) => (prev !== "DOWN" ? "UP" : prev));
          break;
        case "ArrowDown":
          e.preventDefault();
          setDirection((prev) => (prev !== "UP" ? "DOWN" : prev));
          break;
        case "ArrowLeft":
          e.preventDefault();
          setDirection((prev) => (prev !== "RIGHT" ? "LEFT" : prev));
          break;
        case "ArrowRight":
          e.preventDefault();
          setDirection((prev) => (prev !== "LEFT" ? "RIGHT" : prev));
          break;
        case " ":
          e.preventDefault();
          setIsPaused((prev) => !prev);
          break;
      }
    },
    [gameStarted, gameOver],
  );

  useEffect(() => {
    window.addEventListener("keydown", handleKeyPress);
    return () => window.removeEventListener("keydown", handleKeyPress);
  }, [handleKeyPress]);

  useEffect(() => {
    if (!gameStarted || gameOver || isPaused) return;

    const gameLoop = setInterval(() => {
      setSnake((currentSnake) => {
        const newSnake = [...currentSnake];
        const head = { ...newSnake[0] };

        // Move head based on direction
        switch (direction) {
          case "UP":
            head.y -= 1;
            break;
          case "DOWN":
            head.y += 1;
            break;
          case "LEFT":
            head.x -= 1;
            break;
          case "RIGHT":
            head.x += 1;
            break;
        }

        // Check wall collision
        if (
          head.x < 0 ||
          head.x >= GRID_SIZE ||
          head.y < 0 ||
          head.y >= GRID_SIZE
        ) {
          setGameOver(true);
          return currentSnake;
        }

        // Check self collision
        if (
          currentSnake.some(
            (segment) => segment.x === head.x && segment.y === head.y,
          )
        ) {
          setGameOver(true);
          return currentSnake;
        }

        newSnake.unshift(head);

        // Check food collision
        if (head.x === food.x && head.y === food.y) {
          setScore((prev) => prev + 10);
          if (food.story) {
            setCurrentStory(food.story);
            setDiscoveredStories((prev) => [...prev, food.story!]);
            setTimeout(() => setCurrentStory(""), 3000);
          }
          setFood(generateFood());
        } else {
          newSnake.pop();
        }

        return newSnake;
      });
    }, 150);

    return () => clearInterval(gameLoop);
  }, [direction, food, gameStarted, gameOver, isPaused, generateFood]);

  const getFoodEmoji = (type: Food["type"]) => {
    switch (type) {
      case "apple":
        return "🍎";
      case "star":
        return "⭐";
      case "flower":
        return "🌸";
      default:
        return "🍎";
    }
  };

  return (
    <div className="max-w-6xl mx-auto">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="text-center mb-8"
      >
        <h2 className="text-4xl font-bold ghibli-text-gradient mb-4 ghibli-float">
          🐍 Magical Snake Adventure 🐍
        </h2>
        <p className="text-purple-700 dark:text-purple-300 text-lg max-w-3xl mx-auto">
          ✨ Guide the magical snake to collect enchanted fruits and discover
          Aakriti's professional journey! ✨
        </p>
      </motion.div>

      <div className="grid lg:grid-cols-3 gap-8">
        {/* Game Board */}
        <div className="lg:col-span-2">
          <motion.div
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            className="ghibli-glass rounded-3xl p-6 shadow-2xl border-2 border-white/30"
          >
            <div className="flex justify-between items-center mb-6">
              <div className="text-xl font-bold text-purple-800 dark:text-purple-200">
                Score: {score} 🌟
              </div>
              <div className="flex gap-3">
                {!gameStarted ? (
                  <motion.button
                    onClick={startGame}
                    whileHover={{ scale: 1.05 }}
                    whileTap={{ scale: 0.95 }}
                    className="ghibli-button px-6 py-3 rounded-2xl text-white font-bold"
                  >
                    🌟 Start Adventure
                  </motion.button>
                ) : (
                  <>
                    <motion.button
                      onClick={() => setIsPaused(!isPaused)}
                      whileHover={{ scale: 1.05 }}
                      whileTap={{ scale: 0.95 }}
                      className="ghibli-button px-4 py-2 rounded-xl text-white font-semibold"
                    >
                      {isPaused ? "▶️ Resume" : "���️ Pause"}
                    </motion.button>
                    <motion.button
                      onClick={resetGame}
                      whileHover={{ scale: 1.05 }}
                      whileTap={{ scale: 0.95 }}
                      className="ghibli-button px-4 py-2 rounded-xl text-white font-semibold"
                    >
                      🔄 Restart
                    </motion.button>
                  </>
                )}
              </div>
            </div>

            {/* Game Canvas */}
            <div
              className="relative mx-auto"
              style={{ width: CANVAS_SIZE, height: CANVAS_SIZE }}
            >
              <div
                className="relative border-4 border-purple-300 dark:border-purple-600 rounded-2xl bg-gradient-to-br from-green-100 to-green-200 dark:from-green-900/30 dark:to-green-800/30 overflow-hidden"
                style={{ width: CANVAS_SIZE, height: CANVAS_SIZE }}
              >
                {/* Snake segments */}
                {snake.map((segment, index) => (
                  <motion.div
                    key={index}
                    className={`absolute ${
                      index === 0
                        ? "bg-gradient-to-r from-purple-500 to-pink-500 border-2 border-purple-300"
                        : "bg-gradient-to-r from-purple-400 to-pink-400 border border-purple-200"
                    } rounded-lg shadow-lg`}
                    style={{
                      left: segment.x * (CANVAS_SIZE / GRID_SIZE),
                      top: segment.y * (CANVAS_SIZE / GRID_SIZE),
                      width: CANVAS_SIZE / GRID_SIZE - 2,
                      height: CANVAS_SIZE / GRID_SIZE - 2,
                    }}
                    initial={{ scale: 0 }}
                    animate={{ scale: 1 }}
                    transition={{ duration: 0.1 }}
                  >
                    {index === 0 && (
                      <div className="flex items-center justify-center h-full text-xs">
                        {direction === "UP" && "⬆️"}
                        {direction === "DOWN" && "⬇️"}
                        {direction === "LEFT" && "⬅️"}
                        {direction === "RIGHT" && "➡️"}
                      </div>
                    )}
                  </motion.div>
                ))}

                {/* Food */}
                <motion.div
                  className="absolute flex items-center justify-center text-lg"
                  style={{
                    left: food.x * (CANVAS_SIZE / GRID_SIZE),
                    top: food.y * (CANVAS_SIZE / GRID_SIZE),
                    width: CANVAS_SIZE / GRID_SIZE,
                    height: CANVAS_SIZE / GRID_SIZE,
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
                  {getFoodEmoji(food.type)}
                </motion.div>

                {/* Game Over Overlay */}
                <AnimatePresence>
                  {gameOver && (
                    <motion.div
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      exit={{ opacity: 0 }}
                      className="absolute inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center rounded-2xl"
                    >
                      <div className="text-center text-white">
                        <div className="text-4xl mb-4">💫</div>
                        <div className="text-2xl font-bold mb-2">
                          Adventure Complete!
                        </div>
                        <div className="text-lg mb-4">
                          Final Score: {score} 🌟
                        </div>
                        <motion.button
                          onClick={resetGame}
                          whileHover={{ scale: 1.05 }}
                          whileTap={{ scale: 0.95 }}
                          className="ghibli-button px-6 py-3 rounded-2xl text-white font-bold"
                        >
                          🌟 New Adventure
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
                      className="absolute inset-0 bg-white/30 backdrop-blur-sm flex items-center justify-center rounded-2xl"
                    >
                      <div className="text-center text-purple-800">
                        <div className="text-4xl mb-4">⏸️</div>
                        <div className="text-2xl font-bold">Paused</div>
                        <div className="text-sm mt-2">
                          Press SPACE to continue
                        </div>
                      </div>
                    </motion.div>
                  )}
                </AnimatePresence>
              </div>
            </div>

            {/* Controls Info */}
            <div className="mt-6 text-center text-purple-700 dark:text-purple-300">
              <div className="text-sm">
                🎮 Use arrow keys to move • SPACE to pause 🎮
              </div>
            </div>
          </motion.div>
        </div>

        {/* Story Panel */}
        <div className="lg:col-span-1">
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            className="ghibli-glass rounded-3xl p-6 shadow-2xl border-2 border-white/30 h-fit"
          >
            <h3 className="text-2xl font-bold ghibli-text-gradient mb-6 text-center">
              🌟 Discovered Stories 🌟
            </h3>

            {/* Current Story */}
            <AnimatePresence>
              {currentStory && (
                <motion.div
                  initial={{ opacity: 0, y: 20, scale: 0.9 }}
                  animate={{ opacity: 1, y: 0, scale: 1 }}
                  exit={{ opacity: 0, y: -20, scale: 0.9 }}
                  className="mb-6 p-4 bg-gradient-to-r from-yellow-100 to-pink-100 dark:from-yellow-900/30 dark:to-pink-900/30 rounded-2xl border-2 border-yellow-200 dark:border-yellow-700"
                >
                  <div className="text-sm font-bold text-purple-800 dark:text-purple-200 mb-2">
                    ✨ New Discovery! ✨
                  </div>
                  <div className="text-purple-700 dark:text-purple-300 text-sm">
                    {currentStory}
                  </div>
                </motion.div>
              )}
            </AnimatePresence>

            {/* Discovered Stories List */}
            <div className="space-y-3 max-h-96 overflow-y-auto">
              {discoveredStories.length === 0 ? (
                <div className="text-center text-purple-600 dark:text-purple-400 py-8">
                  <div className="text-4xl mb-4">🔮</div>
                  <div className="text-lg font-semibold mb-2">
                    Magical Stories Await!
                  </div>
                  <div className="text-sm">
                    Collect enchanted fruits to discover Aakriti's professional
                    journey
                  </div>
                </div>
              ) : (
                discoveredStories.map((story, index) => (
                  <motion.div
                    key={index}
                    initial={{ opacity: 0, x: 20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: index * 0.1 }}
                    className="p-3 bg-white/50 dark:bg-purple-900/30 rounded-xl border border-purple-200 dark:border-purple-600"
                  >
                    <div className="text-purple-700 dark:text-purple-300 text-sm">
                      {story}
                    </div>
                  </motion.div>
                ))
              )}
            </div>

            {/* Progress */}
            <div className="mt-6 pt-4 border-t border-purple-200 dark:border-purple-600">
              <div className="text-center text-purple-700 dark:text-purple-300">
                <div className="text-sm font-semibold">
                  Stories Discovered: {discoveredStories.length}/
                  {STORY_FOODS.length}
                </div>
                <div className="w-full bg-purple-200 dark:bg-purple-700 rounded-full h-2 mt-2">
                  <motion.div
                    className="bg-gradient-to-r from-purple-500 to-pink-500 h-2 rounded-full"
                    initial={{ width: 0 }}
                    animate={{
                      width: `${(discoveredStories.length / STORY_FOODS.length) * 100}%`,
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
