import { useState, useEffect, useRef, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";

interface MarioState {
  x: number;
  y: number;
  isMoving: boolean;
  isJumping: boolean;
  facingDirection: "left" | "right";
  emotion: "happy" | "confused" | "excited" | "surprised";
}

interface GameStats {
  iteration: number;
  currentX: number;
  currentY: number;
  gradient: number;
  learningRate: number;
  coinsCollected: number;
}

// Loss function: y = (x-3)^2 + sin(2x) + 2
const lossFunction = (x: number): number => {
  return Math.pow(x - 3, 2) + Math.sin(2 * x) + 2;
};

// Derivative of loss function
const gradientFunction = (x: number): number => {
  return 2 * (x - 3) + 2 * Math.cos(2 * x);
};

export default function MarioGradientDescent() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animationRef = useRef<number>();

  const [learningRate, setLearningRate] = useState(0.1);
  const [isRunning, setIsRunning] = useState(false);
  const [mario, setMario] = useState<MarioState>({
    x: 0,
    y: lossFunction(0),
    isMoving: false,
    isJumping: false,
    facingDirection: "right",
    emotion: "happy",
  });

  const [stats, setStats] = useState<GameStats>({
    iteration: 0,
    currentX: 0,
    currentY: lossFunction(0),
    gradient: gradientFunction(0),
    learningRate: 0.1,
    coinsCollected: 0,
  });

  const [coins, setCoins] = useState<
    Array<{ x: number; y: number; collected: boolean; id: number }>
  >([]);
  const [marioMessage, setMarioMessage] = useState<string>("");
  const [showMessage, setShowMessage] = useState(false);

  // Canvas dimensions and scaling
  const canvasWidth = 800;
  const canvasHeight = 400;
  const xMin = -2;
  const xMax = 8;
  const yMin = 0;
  const yMax = 8;

  // Convert world coordinates to canvas coordinates
  const worldToCanvas = useCallback((worldX: number, worldY: number) => {
    const canvasX = ((worldX - xMin) / (xMax - xMin)) * canvasWidth;
    const canvasY =
      canvasHeight - ((worldY - yMin) / (yMax - yMin)) * canvasHeight;
    return { x: canvasX, y: canvasY };
  }, []);

  // Convert canvas coordinates to world coordinates
  const canvasToWorld = useCallback((canvasX: number, canvasY: number) => {
    const worldX = (canvasX / canvasWidth) * (xMax - xMin) + xMin;
    const worldY =
      ((canvasHeight - canvasY) / canvasHeight) * (yMax - yMin) + yMin;
    return { x: worldX, y: worldY };
  }, []);

  // Draw the loss function curve
  const drawLossFunction = useCallback(
    (ctx: CanvasRenderingContext2D) => {
      ctx.strokeStyle = "#3b82f6";
      ctx.lineWidth = 3;
      ctx.beginPath();

      for (let x = xMin; x <= xMax; x += 0.05) {
        const y = lossFunction(x);
        const canvasPoint = worldToCanvas(x, y);

        if (x === xMin) {
          ctx.moveTo(canvasPoint.x, canvasPoint.y);
        } else {
          ctx.lineTo(canvasPoint.x, canvasPoint.y);
        }
      }
      ctx.stroke();

      // Draw grid
      ctx.strokeStyle = "#e5e7eb";
      ctx.lineWidth = 1;

      // Vertical grid lines
      for (let x = Math.ceil(xMin); x <= Math.floor(xMax); x++) {
        const canvasPoint = worldToCanvas(x, yMin);
        ctx.beginPath();
        ctx.moveTo(canvasPoint.x, canvasHeight);
        ctx.lineTo(canvasPoint.x, 0);
        ctx.stroke();
      }

      // Horizontal grid lines
      for (let y = Math.ceil(yMin); y <= Math.floor(yMax); y++) {
        const canvasPoint = worldToCanvas(xMin, y);
        ctx.beginPath();
        ctx.moveTo(0, canvasPoint.y);
        ctx.lineTo(canvasWidth, canvasPoint.y);
        ctx.stroke();
      }
    },
    [worldToCanvas],
  );

  // Draw Mario character
  const drawMario = useCallback(
    (ctx: CanvasRenderingContext2D, marioState: MarioState) => {
      const canvasPoint = worldToCanvas(marioState.x, marioState.y);
      const size = 20;

      // Mario's body (red)
      ctx.fillStyle = "#dc2626";
      ctx.fillRect(
        canvasPoint.x - size / 2,
        canvasPoint.y - size,
        size,
        size * 0.7,
      );

      // Mario's hat (red)
      ctx.fillStyle = "#dc2626";
      ctx.fillRect(
        canvasPoint.x - size / 2 - 2,
        canvasPoint.y - size - 5,
        size + 4,
        8,
      );

      // Mario's face (peach)
      ctx.fillStyle = "#fbbf24";
      ctx.fillRect(
        canvasPoint.x - size / 3,
        canvasPoint.y - size + 5,
        size * 0.6,
        size * 0.4,
      );

      // Eyes
      ctx.fillStyle = "#000000";
      ctx.fillRect(canvasPoint.x - 6, canvasPoint.y - size + 8, 3, 3);
      ctx.fillRect(canvasPoint.x + 3, canvasPoint.y - size + 8, 3, 3);

      // Mustache
      ctx.fillStyle = "#8b4513";
      ctx.fillRect(canvasPoint.x - 8, canvasPoint.y - size + 12, 16, 4);

      // Legs (blue)
      ctx.fillStyle = "#2563eb";
      ctx.fillRect(
        canvasPoint.x - 8,
        canvasPoint.y - size * 0.3,
        6,
        size * 0.3,
      );
      ctx.fillRect(
        canvasPoint.x + 2,
        canvasPoint.y - size * 0.3,
        6,
        size * 0.3,
      );

      // Gradient arrow (show direction Mario will move)
      if (isRunning) {
        const gradient = gradientFunction(marioState.x);
        const arrowLength = Math.min(Math.abs(gradient) * 30, 50);
        const arrowDirection = gradient > 0 ? -1 : 1;

        ctx.strokeStyle = "#ef4444";
        ctx.lineWidth = 3;
        ctx.beginPath();
        ctx.moveTo(canvasPoint.x, canvasPoint.y - size - 10);
        ctx.lineTo(
          canvasPoint.x + arrowDirection * arrowLength,
          canvasPoint.y - size - 10,
        );

        // Arrow head
        ctx.lineTo(
          canvasPoint.x + arrowDirection * (arrowLength - 8),
          canvasPoint.y - size - 15,
        );
        ctx.moveTo(
          canvasPoint.x + arrowDirection * arrowLength,
          canvasPoint.y - size - 10,
        );
        ctx.lineTo(
          canvasPoint.x + arrowDirection * (arrowLength - 8),
          canvasPoint.y - size - 5,
        );
        ctx.stroke();
      }
    },
    [worldToCanvas, isRunning],
  );

  // Draw coins
  const drawCoins = useCallback(
    (ctx: CanvasRenderingContext2D, coinArray: typeof coins) => {
      coinArray.forEach((coin) => {
        if (!coin.collected) {
          const canvasPoint = worldToCanvas(coin.x, coin.y);

          // Coin (golden circle)
          ctx.fillStyle = "#fbbf24";
          ctx.beginPath();
          ctx.arc(canvasPoint.x, canvasPoint.y, 8, 0, 2 * Math.PI);
          ctx.fill();

          // Coin shine
          ctx.fillStyle = "#ffffff";
          ctx.beginPath();
          ctx.arc(canvasPoint.x - 2, canvasPoint.y - 2, 3, 0, 2 * Math.PI);
          ctx.fill();
        }
      });
    },
    [worldToCanvas],
  );

  // Animation loop
  const animate = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    // Clear canvas
    ctx.clearRect(0, 0, canvasWidth, canvasHeight);

    // Draw background
    const gradient = ctx.createLinearGradient(0, 0, 0, canvasHeight);
    gradient.addColorStop(0, "#dbeafe");
    gradient.addColorStop(1, "#bfdbfe");
    ctx.fillStyle = gradient;
    ctx.fillRect(0, 0, canvasWidth, canvasHeight);

    // Draw loss function
    drawLossFunction(ctx);

    // Draw coins
    drawCoins(ctx, coins);

    // Draw Mario
    drawMario(ctx, mario);

    animationRef.current = requestAnimationFrame(animate);
  }, [mario, coins, drawLossFunction, drawCoins, drawMario]);

  // Start animation
  useEffect(() => {
    animate();
    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, [animate]);

  // Show Mario message
  const showMarioMessage = (
    message: string,
    emotion: MarioState["emotion"] = "happy",
  ) => {
    setMarioMessage(message);
    setShowMessage(true);
    setMario((prev) => ({ ...prev, emotion }));
    setTimeout(() => setShowMessage(false), 3000);
  };

  // Gradient descent step
  const performGradientStep = useCallback(() => {
    setMario((prevMario) => {
      const currentGradient = gradientFunction(prevMario.x);
      const step = learningRate * currentGradient;
      const newX = prevMario.x - step;
      const newY = lossFunction(newX);

      // Check bounds
      if (newX < xMin || newX > xMax) {
        showMarioMessage("Whoa! I went out of bounds!", "surprised");
        setIsRunning(false);
        return prevMario;
      }

      // Check if learning rate is too large (oscillating)
      if (Math.abs(step) > 1) {
        showMarioMessage(
          "Whoa! Too fast! Lower the learning rate!",
          "confused",
        );
        setIsRunning(false);
        return { ...prevMario, emotion: "confused" };
      }

      // Check if converged (gradient close to 0)
      if (Math.abs(currentGradient) < 0.01) {
        showMarioMessage("Wahoo! I found the minimum!", "excited");
        setIsRunning(false);
        return { ...prevMario, emotion: "excited" };
      }

      // Check if improvement (collect coin)
      if (newY < prevMario.y - 0.1) {
        const newCoin = {
          x: newX,
          y: newY,
          collected: false,
          id: Date.now(),
        };
        setCoins((prev) => [...prev, newCoin]);
        setStats((prev) => ({
          ...prev,
          coinsCollected: prev.coinsCollected + 1,
        }));
        showMarioMessage("Coin! Getting closer!", "excited");
      }

      // Update stats
      setStats((prev) => ({
        ...prev,
        iteration: prev.iteration + 1,
        currentX: newX,
        currentY: newY,
        gradient: currentGradient,
        learningRate,
      }));

      return {
        ...prevMario,
        x: newX,
        y: newY,
        isMoving: true,
        facingDirection: step > 0 ? "left" : "right",
        emotion: "happy",
      };
    });
  }, [learningRate]);

  // Gradient descent animation
  useEffect(() => {
    if (!isRunning) return;

    const interval = setInterval(() => {
      performGradientStep();
    }, 500); // 500ms between steps

    return () => clearInterval(interval);
  }, [isRunning, performGradientStep]);

  // Reset game
  const resetGame = () => {
    const startX = Math.random() * (xMax - xMin) + xMin;
    setMario({
      x: startX,
      y: lossFunction(startX),
      isMoving: false,
      isJumping: false,
      facingDirection: "right",
      emotion: "happy",
    });
    setStats({
      iteration: 0,
      currentX: startX,
      currentY: lossFunction(startX),
      gradient: gradientFunction(startX),
      learningRate,
      coinsCollected: 0,
    });
    setCoins([]);
    setIsRunning(false);
    setShowMessage(false);
  };

  // Handle canvas click to place Mario
  const handleCanvasClick = (event: React.MouseEvent<HTMLCanvasElement>) => {
    if (isRunning) return;

    const canvas = canvasRef.current;
    if (!canvas) return;

    const rect = canvas.getBoundingClientRect();
    const canvasX = event.clientX - rect.left;
    const canvasY = event.clientY - rect.top;

    const worldCoords = canvasToWorld(canvasX, canvasY);
    const newY = lossFunction(worldCoords.x);

    setMario((prev) => ({
      ...prev,
      x: worldCoords.x,
      y: newY,
    }));

    setStats((prev) => ({
      ...prev,
      currentX: worldCoords.x,
      currentY: newY,
      gradient: gradientFunction(worldCoords.x),
    }));
  };

  return (
    <div className="max-w-7xl mx-auto">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="text-center mb-8"
      >
        <h2 className="text-4xl font-bold text-slate-900 dark:text-slate-100 mb-4">
          🍄 Mario's Gradient Descent Adventure 🍄
        </h2>
        <p className="text-lg text-slate-600 dark:text-slate-300 max-w-4xl mx-auto">
          Help Mario find the lowest point on the loss function! Adjust the
          learning rate and watch Mario use gradient descent to minimize the
          loss.
        </p>
      </motion.div>

      <div className="grid lg:grid-cols-3 gap-8">
        {/* Game Canvas */}
        <div className="lg:col-span-2">
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            className="bg-white dark:bg-slate-800 rounded-xl shadow-lg border border-slate-200 dark:border-slate-700 p-6"
          >
            {/* Controls */}
            <div className="flex flex-wrap items-center justify-between mb-6 gap-4">
              <div className="flex items-center gap-4">
                <div className="flex items-center gap-2">
                  <label className="text-sm font-medium text-slate-700 dark:text-slate-300">
                    Learning Rate:
                  </label>
                  <input
                    type="range"
                    min="0.01"
                    max="0.5"
                    step="0.01"
                    value={learningRate}
                    onChange={(e) =>
                      setLearningRate(parseFloat(e.target.value))
                    }
                    disabled={isRunning}
                    className="w-32"
                  />
                  <span className="text-sm font-mono text-slate-600 dark:text-slate-400 min-w-[60px]">
                    {learningRate.toFixed(2)}
                  </span>
                </div>
              </div>

              <div className="flex gap-3">
                <motion.button
                  onClick={() => setIsRunning(!isRunning)}
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                  className={`px-6 py-2 rounded-lg font-semibold transition-colors ${
                    isRunning
                      ? "bg-red-600 hover:bg-red-700 text-white"
                      : "bg-green-600 hover:bg-green-700 text-white"
                  }`}
                >
                  {isRunning ? "⏸️ Stop" : "▶️ Start Descent"}
                </motion.button>
                <motion.button
                  onClick={resetGame}
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                  className="px-6 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg font-semibold transition-colors"
                >
                  🔄 Reset
                </motion.button>
              </div>
            </div>

            {/* Canvas */}
            <div className="relative">
              <canvas
                ref={canvasRef}
                width={canvasWidth}
                height={canvasHeight}
                onClick={handleCanvasClick}
                className="border border-slate-300 dark:border-slate-600 rounded-lg cursor-pointer w-full max-w-full"
                style={{ aspectRatio: `${canvasWidth}/${canvasHeight}` }}
              />

              {/* Mario Message */}
              <AnimatePresence>
                {showMessage && (
                  <motion.div
                    initial={{ opacity: 0, scale: 0.8, y: 20 }}
                    animate={{ opacity: 1, scale: 1, y: 0 }}
                    exit={{ opacity: 0, scale: 0.8, y: -20 }}
                    className="absolute top-4 left-4 bg-yellow-100 border border-yellow-300 rounded-lg px-4 py-2 shadow-lg"
                  >
                    <div className="flex items-center gap-2">
                      <span className="text-2xl">🍄</span>
                      <span className="font-semibold text-yellow-800">
                        {marioMessage}
                      </span>
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>
            </div>

            <p className="text-sm text-slate-500 dark:text-slate-400 mt-4 text-center">
              💡 Click anywhere on the curve to place Mario, then start the
              descent!
            </p>
          </motion.div>
        </div>

        {/* Stats Panel */}
        <div className="lg:col-span-1">
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            className="bg-white dark:bg-slate-800 rounded-xl shadow-lg border border-slate-200 dark:border-slate-700 p-6 h-fit"
          >
            <h3 className="text-2xl font-bold text-slate-900 dark:text-slate-100 mb-6 text-center">
              📊 Mario's Stats
            </h3>

            <div className="space-y-4">
              <div className="bg-slate-50 dark:bg-slate-700 rounded-lg p-4">
                <div className="text-sm font-medium text-slate-600 dark:text-slate-400">
                  Iteration
                </div>
                <div className="text-2xl font-bold text-blue-600">
                  {stats.iteration}
                </div>
              </div>

              <div className="bg-slate-50 dark:bg-slate-700 rounded-lg p-4">
                <div className="text-sm font-medium text-slate-600 dark:text-slate-400">
                  Position (x)
                </div>
                <div className="text-xl font-mono text-slate-900 dark:text-slate-100">
                  {stats.currentX.toFixed(3)}
                </div>
              </div>

              <div className="bg-slate-50 dark:bg-slate-700 rounded-lg p-4">
                <div className="text-sm font-medium text-slate-600 dark:text-slate-400">
                  Loss (y)
                </div>
                <div className="text-xl font-mono text-slate-900 dark:text-slate-100">
                  {stats.currentY.toFixed(3)}
                </div>
              </div>

              <div className="bg-slate-50 dark:bg-slate-700 rounded-lg p-4">
                <div className="text-sm font-medium text-slate-600 dark:text-slate-400">
                  Gradient
                </div>
                <div
                  className={`text-xl font-mono ${
                    Math.abs(stats.gradient) < 0.1
                      ? "text-green-600"
                      : "text-red-600"
                  }`}
                >
                  {stats.gradient.toFixed(3)}
                </div>
              </div>

              <div className="bg-yellow-50 dark:bg-yellow-900/20 rounded-lg p-4">
                <div className="text-sm font-medium text-yellow-600 dark:text-yellow-400">
                  Coins Collected
                </div>
                <div className="text-2xl font-bold text-yellow-600">
                  🪙 {stats.coinsCollected}
                </div>
              </div>
            </div>

            {/* Learning Tips */}
            <div className="mt-8 p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
              <h4 className="font-semibold text-blue-800 dark:text-blue-300 mb-2">
                💡 Tips
              </h4>
              <ul className="text-sm text-blue-700 dark:text-blue-300 space-y-1">
                <li>• Lower learning rate = smaller, safer steps</li>
                <li>• Higher learning rate = bigger steps, may overshoot</li>
                <li>• Mario stops when gradient ≈ 0</li>
                <li>• Coins appear when loss decreases!</li>
              </ul>
            </div>
          </motion.div>
        </div>
      </div>

      {/* Educational Footer */}
      <motion.div
        initial={{ opacity: 0, y: 40 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.6 }}
        className="mt-16 bg-gradient-to-r from-blue-600 to-purple-600 rounded-xl p-8 text-white"
      >
        <div className="text-center mb-6">
          <h3 className="text-3xl font-bold mb-4">
            🎓 Learn More About Gradient Descent
          </h3>
          <p className="text-lg opacity-90 max-w-3xl mx-auto">
            Gradient descent is the foundation of machine learning! It's how
            neural networks learn by finding the optimal parameters that
            minimize the loss function.
          </p>
        </div>

        <div className="grid md:grid-cols-2 gap-6">
          <motion.a
            href="https://en.wikipedia.org/wiki/Gradient_descent"
            target="_blank"
            rel="noopener noreferrer"
            whileHover={{ scale: 1.02 }}
            className="bg-white/10 backdrop-blur-sm rounded-lg p-6 border border-white/20 hover:bg-white/20 transition-colors"
          >
            <div className="text-2xl mb-3">📚</div>
            <h4 className="text-xl font-semibold mb-2">
              Learn More About Gradient Descent
            </h4>
            <p className="opacity-90">
              Dive deeper into the mathematics and theory behind gradient
              descent optimization.
            </p>
          </motion.a>

          <motion.div
            whileHover={{ scale: 1.02 }}
            className="bg-white/10 backdrop-blur-sm rounded-lg p-6 border border-white/20 hover:bg-white/20 transition-colors cursor-pointer"
          >
            <div className="text-2xl mb-3">🧠</div>
            <h4 className="text-xl font-semibold mb-2">
              See Backpropagation Demo Next!
            </h4>
            <p className="opacity-90">
              Ready for the next level? Learn how gradients flow through neural
              networks.
            </p>
          </motion.div>
        </div>
      </motion.div>
    </div>
  );
}
