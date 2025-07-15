import React, { useState, useEffect } from "react";
import { Link } from "react-router-dom";
import { motion, AnimatePresence } from "framer-motion";
import Navigation from "../components/Navigation";

interface Discovery {
  id: number;
  title: string;
  year: string;
  discoverer: string;
  discovererBio: string;
  paperTitle: string;
  paperLink: string;
  description: string;
  impact: string;
  demoType: string;
}

const discoveries: Discovery[] = [
  {
    id: 1,
    title: "The Perceptron",
    year: "1957",
    discoverer: "Frank Rosenblatt",
    discovererBio:
      "American psychologist and computer scientist at Cornell University who pioneered artificial neural networks.",
    paperTitle:
      "The Perceptron: A Probabilistic Model for Information Storage and Organization in the Brain",
    paperLink: "https://psycnet.apa.org/record/1959-09865-001",
    description:
      "The first artificial neural network algorithm that could learn to classify inputs into categories.",
    impact: "Foundation for all modern neural networks and deep learning.",
    demoType: "perceptron",
  },
  {
    id: 2,
    title: "Backpropagation",
    year: "1986",
    discoverer: "Geoffrey Hinton, David Rumelhart, Ronald Williams",
    discovererBio:
      "Geoffrey Hinton is known as the 'Godfather of Deep Learning' and won the 2018 Turing Award for his work on neural networks.",
    paperTitle: "Learning representations by back-propagating errors",
    paperLink: "https://www.nature.com/articles/323533a0",
    description:
      "Algorithm for training multi-layer neural networks by calculating gradients and updating weights.",
    impact:
      "Enabled training of deep neural networks, making modern AI possible.",
    demoType: "backprop",
  },
  {
    id: 3,
    title: "Convolutional Neural Networks",
    year: "1989",
    discoverer: "Yann LeCun",
    discovererBio:
      "French computer scientist, Chief AI Scientist at Meta, and 2018 Turing Award winner for deep learning breakthroughs.",
    paperTitle: "Backpropagation Applied to Handwritten Zip Code Recognition",
    paperLink: "http://yann.lecun.com/exdb/publis/pdf/lecun-89e.pdf",
    description:
      "Neural networks designed to process grid-like data such as images using convolution operations.",
    impact:
      "Revolutionized computer vision and image recognition, enabling modern visual AI.",
    demoType: "cnn",
  },
  {
    id: 4,
    title: "Long Short-Term Memory (LSTM)",
    year: "1997",
    discoverer: "Sepp Hochreiter & Jürgen Schmidhuber",
    discovererBio:
      "Sepp Hochreiter is a German computer scientist known for LSTM networks and modern AI contributions.",
    paperTitle: "Long Short-Term Memory",
    paperLink:
      "https://www.mitpressjournals.org/doi/abs/10.1162/neco.1997.9.8.1735",
    description:
      "Recurrent neural network architecture that can learn long-term dependencies in sequential data.",
    impact:
      "Enabled breakthroughs in natural language processing, speech recognition, and time series prediction.",
    demoType: "lstm",
  },
  {
    id: 5,
    title: "Q-Learning (Reinforcement Learning)",
    year: "1989",
    discoverer: "Chris Watkins",
    discovererBio:
      "British computer scientist who developed Q-learning as part of his PhD thesis at Cambridge University.",
    paperTitle: "Learning from Delayed Rewards",
    paperLink: "https://www.cs.rhul.ac.uk/~chrisw/new_thesis.pdf",
    description:
      "Algorithm that enables agents to learn optimal actions in an environment through trial and error.",
    impact:
      "Foundation for game-playing AI, robotics, and autonomous systems like AlphaGo.",
    demoType: "qlearning",
  },
  {
    id: 6,
    title: "Attention Mechanism",
    year: "2014",
    discoverer: "Dzmitry Bahdanau, Kyunghyun Cho, Yoshua Bengio",
    discovererBio:
      "Yoshua Bengio is a Canadian computer scientist and 2018 Turing Award winner for deep learning.",
    paperTitle:
      "Neural Machine Translation by Jointly Learning to Align and Translate",
    paperLink: "https://arxiv.org/abs/1409.0473",
    description:
      "Mechanism allowing neural networks to focus on relevant parts of input when making predictions.",
    impact:
      "Enabled breakthrough improvements in machine translation and became foundation for Transformers.",
    demoType: "attention",
  },
  {
    id: 7,
    title: "Transformers",
    year: "2017",
    discoverer: "Ashish Vaswani et al. (Google)",
    discovererBio:
      "Team of Google researchers led by Ashish Vaswani who revolutionized natural language processing.",
    paperTitle: "Attention Is All You Need",
    paperLink: "https://arxiv.org/abs/1706.03762",
    description:
      "Architecture relying entirely on attention mechanisms, dispensing with recurrence and convolutions.",
    impact:
      "Enabled GPT, BERT, and modern large language models that power ChatGPT and similar systems.",
    demoType: "transformer",
  },
  {
    id: 8,
    title: "Generative Adversarial Networks (GANs)",
    year: "2014",
    discoverer: "Ian Goodfellow",
    discovererBio:
      "Canadian computer scientist and former Google researcher, known for inventing GANs.",
    paperTitle: "Generative Adversarial Networks",
    paperLink: "https://arxiv.org/abs/1406.2661",
    description:
      "Two neural networks competing against each other to generate realistic synthetic data.",
    impact:
      "Revolutionized image generation, art creation, and synthetic media production.",
    demoType: "gan",
  },
  {
    id: 9,
    title: "ResNet (Residual Networks)",
    year: "2015",
    discoverer: "Kaiming He et al. (Microsoft)",
    discovererBio:
      "Kaiming He is a Chinese computer scientist at Meta AI Research, known for breakthrough deep learning architectures.",
    paperTitle: "Deep Residual Learning for Image Recognition",
    paperLink: "https://arxiv.org/abs/1512.03385",
    description:
      "Deep neural networks with skip connections that allow training of very deep networks.",
    impact:
      "Enabled training of networks with hundreds of layers, dramatically improving computer vision.",
    demoType: "resnet",
  },
  {
    id: 10,
    title: "Deep Q-Networks (DQN)",
    year: "2015",
    discoverer: "Volodymyr Mnih et al. (DeepMind)",
    discovererBio:
      "Research team at DeepMind led by Volodymyr Mnih that combined deep learning with reinforcement learning.",
    paperTitle: "Human-level control through deep reinforcement learning",
    paperLink: "https://www.nature.com/articles/nature14236",
    description:
      "Combination of deep learning and Q-learning that can master complex games from raw pixels.",
    impact:
      "Achieved superhuman performance in Atari games and paved way for AlphaGo and game-playing AI.",
    demoType: "dqn",
  },
];

const PerceptronDemo = () => {
  const [input1, setInput1] = useState(0.5);
  const [input2, setInput2] = useState(0.3);
  const [weight1, setWeight1] = useState(0.7);
  const [weight2, setWeight2] = useState(0.4);
  const [bias, setBias] = useState(-0.2);
  const [animateSignal, setAnimateSignal] = useState(false);

  const output = input1 * weight1 + input2 * weight2 + bias;
  const activated = output > 0 ? 1 : 0;

  const triggerAnimation = () => {
    setAnimateSignal(true);
    setTimeout(() => setAnimateSignal(false), 2000);
  };

  return (
    <div className="bg-white p-6 rounded-lg border border-gray-200">
      <h4 className="text-lg font-semibold mb-4">Interactive Perceptron</h4>

      {/* Neural Network Visualization */}
      <div className="mb-6 bg-gray-50 p-6 rounded-lg">
        <div className="flex items-center justify-between relative">
          {/* Inputs */}
          <div className="space-y-8">
            <motion.div
              className="w-12 h-12 bg-blue-500 rounded-full flex items-center justify-center text-white font-bold"
              animate={{ scale: animateSignal ? [1, 1.2, 1] : 1 }}
              transition={{ duration: 0.5 }}
            >
              {input1.toFixed(1)}
            </motion.div>
            <motion.div
              className="w-12 h-12 bg-blue-500 rounded-full flex items-center justify-center text-white font-bold"
              animate={{ scale: animateSignal ? [1, 1.2, 1] : 1 }}
              transition={{ duration: 0.5, delay: 0.1 }}
            >
              {input2.toFixed(1)}
            </motion.div>
          </div>

          {/* Connections with weights */}
          <div className="flex-1 relative mx-8">
            <svg className="w-full h-32" viewBox="0 0 200 120">
              {/* Weight lines */}
              <motion.line
                x1="10"
                y1="20"
                x2="180"
                y2="60"
                stroke={weight1 > 0 ? "#10b981" : "#ef4444"}
                strokeWidth={Math.abs(weight1) * 4 + 1}
                initial={{ pathLength: 0 }}
                animate={{
                  pathLength: animateSignal ? [0, 1] : 1,
                  stroke: animateSignal
                    ? ["#3b82f6", weight1 > 0 ? "#10b981" : "#ef4444"]
                    : weight1 > 0
                      ? "#10b981"
                      : "#ef4444",
                }}
                transition={{ duration: 0.8 }}
              />
              <motion.line
                x1="10"
                y1="100"
                x2="180"
                y2="60"
                stroke={weight2 > 0 ? "#10b981" : "#ef4444"}
                strokeWidth={Math.abs(weight2) * 4 + 1}
                initial={{ pathLength: 0 }}
                animate={{
                  pathLength: animateSignal ? [0, 1] : 1,
                  stroke: animateSignal
                    ? ["#3b82f6", weight2 > 0 ? "#10b981" : "#ef4444"]
                    : weight2 > 0
                      ? "#10b981"
                      : "#ef4444",
                }}
                transition={{ duration: 0.8, delay: 0.1 }}
              />

              {/* Weight labels */}
              <text
                x="60"
                y="30"
                fill="#374151"
                fontSize="12"
                fontWeight="bold"
              >
                w₁: {weight1.toFixed(1)}
              </text>
              <text
                x="60"
                y="90"
                fill="#374151"
                fontSize="12"
                fontWeight="bold"
              >
                w₂: {weight2.toFixed(1)}
              </text>
            </svg>
          </div>

          {/* Output neuron */}
          <motion.div
            className={`w-16 h-16 rounded-full flex items-center justify-center text-white font-bold text-lg ${
              activated ? "bg-green-500" : "bg-red-500"
            }`}
            animate={{
              scale: animateSignal ? [1, 1.3, 1] : 1,
              backgroundColor: animateSignal
                ? ["#3b82f6", activated ? "#10b981" : "#ef4444"]
                : activated
                  ? "#10b981"
                  : "#ef4444",
            }}
            transition={{ duration: 0.5, delay: 0.8 }}
          >
            {activated}
          </motion.div>
        </div>

        {/* Bias visualization */}
        <div className="mt-4 text-center">
          <div className="inline-flex items-center gap-2">
            <span className="text-sm font-medium">Bias:</span>
            <motion.div
              className={`px-3 py-1 rounded-full text-white font-bold ${
                bias > 0 ? "bg-green-500" : "bg-red-500"
              }`}
              animate={{ scale: animateSignal ? [1, 1.1, 1] : 1 }}
              transition={{ duration: 0.5, delay: 0.6 }}
            >
              {bias.toFixed(1)}
            </motion.div>
          </div>
        </div>

        <button
          onClick={triggerAnimation}
          className="mt-4 w-full button-primary"
        >
          🔥 Activate Signal Flow
        </button>
      </div>

      <div className="space-y-4">
        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="block text-sm font-medium mb-1">
              Input 1: {input1}
            </label>
            <input
              type="range"
              min="-1"
              max="1"
              step="0.1"
              value={input1}
              onChange={(e) => setInput1(parseFloat(e.target.value))}
              className="w-full"
            />
          </div>
          <div>
            <label className="block text-sm font-medium mb-1">
              Weight 1: {weight1}
            </label>
            <input
              type="range"
              min="-1"
              max="1"
              step="0.1"
              value={weight1}
              onChange={(e) => setWeight1(parseFloat(e.target.value))}
              className="w-full"
            />
          </div>
        </div>
        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="block text-sm font-medium mb-1">
              Input 2: {input2}
            </label>
            <input
              type="range"
              min="-1"
              max="1"
              step="0.1"
              value={input2}
              onChange={(e) => setInput2(parseFloat(e.target.value))}
              className="w-full"
            />
          </div>
          <div>
            <label className="block text-sm font-medium mb-1">
              Weight 2: {weight2}
            </label>
            <input
              type="range"
              min="-1"
              max="1"
              step="0.1"
              value={weight2}
              onChange={(e) => setWeight2(parseFloat(e.target.value))}
              className="w-full"
            />
          </div>
        </div>
        <div>
          <label className="block text-sm font-medium mb-1">Bias: {bias}</label>
          <input
            type="range"
            min="-1"
            max="1"
            step="0.1"
            value={bias}
            onChange={(e) => setBias(parseFloat(e.target.value))}
            className="w-full"
          />
        </div>
        <div className="bg-gray-50 p-4 rounded">
          <p>
            Sum: {input1} × {weight1} + {input2} × {weight2} + {bias} ={" "}
            {output.toFixed(2)}
          </p>
          <p className="font-semibold">
            Output: {activated} {activated ? "(Activated)" : "(Not Activated)"}
          </p>
        </div>
      </div>
    </div>
  );
};

const AttentionDemo = () => {
  const [sentence] = useState("The cat sat on the mat");
  const [focusWord, setFocusWord] = useState("cat");
  const words = sentence.split(" ");

  const getAttentionWeight = (word: string) => {
    if (word === focusWord) return 1.0;
    if (word === "The" && focusWord === "cat") return 0.3;
    if (word === "sat" && focusWord === "cat") return 0.7;
    return 0.1;
  };

  return (
    <div className="bg-white p-6 rounded-lg border border-gray-200">
      <h4 className="text-lg font-semibold mb-4">Attention Mechanism Demo</h4>
      <p className="mb-4">Click on words to see how attention focuses:</p>
      <div className="space-y-4">
        <div className="flex flex-wrap gap-2">
          {words.map((word, idx) => (
            <button
              key={idx}
              onClick={() => setFocusWord(word)}
              className={`px-3 py-2 rounded transition-all ${
                word === focusWord
                  ? "bg-blue-500 text-white"
                  : "bg-gray-100 hover:bg-gray-200"
              }`}
              style={{
                opacity: 0.3 + 0.7 * getAttentionWeight(word),
                transform: `scale(${0.9 + 0.1 * getAttentionWeight(word)})`,
              }}
            >
              {word}
            </button>
          ))}
        </div>
        <div className="bg-gray-50 p-4 rounded">
          <p>
            <strong>Focused on:</strong> "{focusWord}"
          </p>
          <p>
            <strong>Attention weights:</strong>
          </p>
          {words.map((word, idx) => (
            <span key={idx} className="inline-block mr-4">
              {word}: {getAttentionWeight(word).toFixed(1)}
            </span>
          ))}
        </div>
      </div>
    </div>
  );
};

const QLearningDemo = () => {
  const [currentState, setCurrentState] = useState(0);
  const [episode, setEpisode] = useState(0);
  const [qTable, setQTable] = useState([
    [0, 0],
    [0, 0],
    [0, 0],
  ]);

  const states = ["Start", "Middle", "Goal"];
  const actions = ["Left", "Right"];

  const takeAction = (action: number) => {
    let newState = currentState;
    let reward = 0;

    if (currentState === 0 && action === 1) {
      // Start -> Right -> Middle
      newState = 1;
      reward = 0;
    } else if (currentState === 1 && action === 1) {
      // Middle -> Right -> Goal
      newState = 2;
      reward = 10;
    } else if (currentState === 1 && action === 0) {
      // Middle -> Left -> Start
      newState = 0;
      reward = -1;
    }

    // Update Q-table
    const newQTable = [...qTable];
    const learningRate = 0.1;
    const discountFactor = 0.9;
    const maxNextQ = Math.max(...newQTable[newState]);

    newQTable[currentState][action] =
      newQTable[currentState][action] +
      learningRate *
        (reward + discountFactor * maxNextQ - newQTable[currentState][action]);

    setQTable(newQTable);
    setCurrentState(newState);
    setEpisode(episode + 1);

    if (newState === 2) {
      setTimeout(() => setCurrentState(0), 1000); // Reset to start
    }
  };

  return (
    <div className="bg-white p-6 rounded-lg border border-gray-200">
      <h4 className="text-lg font-semibold mb-4">Q-Learning Demo</h4>
      <div className="space-y-4">
        <div className="flex gap-4 items-center">
          {states.map((state, idx) => (
            <div
              key={idx}
              className={`px-4 py-2 rounded-lg border-2 ${
                idx === currentState
                  ? "border-blue-500 bg-blue-100"
                  : "border-gray-300"
              }`}
            >
              {state}
            </div>
          ))}
        </div>

        <div className="flex gap-2">
          <button
            onClick={() => takeAction(0)}
            disabled={currentState === 0}
            className="button-primary disabled:opacity-50"
          >
            Go Left
          </button>
          <button
            onClick={() => takeAction(1)}
            disabled={currentState === 2}
            className="button-primary disabled:opacity-50"
          >
            Go Right
          </button>
        </div>

        <div className="bg-gray-50 p-4 rounded">
          <p>
            <strong>Episode:</strong> {episode}
          </p>
          <p>
            <strong>Current State:</strong> {states[currentState]}
          </p>
          <p>
            <strong>Q-Table:</strong>
          </p>
          {qTable.map((row, stateIdx) => (
            <div key={stateIdx}>
              {states[stateIdx]}: Left={row[0].toFixed(2)}, Right=
              {row[1].toFixed(2)}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

const CNNDemo = () => {
  const [filterPosition, setFilterPosition] = useState(0);
  const [isAnimating, setIsAnimating] = useState(false);

  const inputImage = [
    [1, 0, 1, 0, 1],
    [0, 1, 0, 1, 0],
    [1, 0, 1, 0, 1],
    [0, 1, 0, 1, 0],
    [1, 0, 1, 0, 1],
  ];

  const filter = [
    [1, 0, 1],
    [0, 1, 0],
    [1, 0, 1],
  ];

  const startConvolution = () => {
    setIsAnimating(true);
    setFilterPosition(0);
    let pos = 0;
    const interval = setInterval(() => {
      pos += 1;
      setFilterPosition(pos);
      if (pos >= 9) {
        clearInterval(interval);
        setIsAnimating(false);
      }
    }, 500);
  };

  const getFilterX = (pos: number) => (pos % 3) * 40;
  const getFilterY = (pos: number) => Math.floor(pos / 3) * 40;

  return (
    <div className="bg-white p-6 rounded-lg border border-gray-200">
      <h4 className="text-lg font-semibold mb-4">CNN Convolution Animation</h4>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-6">
        {/* Input Image */}
        <div className="text-center">
          <h5 className="text-sm font-semibold mb-2">Input Image (5×5)</h5>
          <div className="relative inline-block bg-gray-100 p-4 rounded">
            <div className="grid grid-cols-5 gap-1">
              {inputImage.flat().map((pixel, idx) => (
                <div
                  key={idx}
                  className={`w-8 h-8 border border-gray-300 flex items-center justify-center text-xs font-bold ${
                    pixel ? "bg-blue-500 text-white" : "bg-white text-gray-500"
                  }`}
                >
                  {pixel}
                </div>
              ))}
            </div>

            {/* Animated Filter Overlay */}
            <motion.div
              className="absolute top-4 left-4 border-4 border-red-500 bg-red-200/30 rounded"
              style={{
                width: "120px",
                height: "120px",
                transform: `translate(${getFilterX(filterPosition)}px, ${getFilterY(filterPosition)}px)`,
              }}
              animate={{
                x: getFilterX(filterPosition),
                y: getFilterY(filterPosition),
              }}
              transition={{ duration: 0.4 }}
            />
          </div>
        </div>

        {/* Filter */}
        <div className="text-center">
          <h5 className="text-sm font-semibold mb-2">Filter/Kernel (3×3)</h5>
          <div className="inline-block bg-red-100 p-4 rounded">
            <div className="grid grid-cols-3 gap-1">
              {filter.flat().map((weight, idx) => (
                <motion.div
                  key={idx}
                  className={`w-10 h-10 border-2 border-red-500 flex items-center justify-center text-sm font-bold ${
                    weight ? "bg-red-500 text-white" : "bg-white text-gray-500"
                  }`}
                  animate={{ scale: isAnimating ? [1, 1.1, 1] : 1 }}
                  transition={{
                    duration: 0.4,
                    repeat: isAnimating ? Infinity : 0,
                  }}
                >
                  {weight}
                </motion.div>
              ))}
            </div>
          </div>
        </div>

        {/* Feature Map */}
        <div className="text-center">
          <h5 className="text-sm font-semibold mb-2">Feature Map (3×3)</h5>
          <div className="inline-block bg-green-100 p-4 rounded">
            <div className="grid grid-cols-3 gap-1">
              {Array.from({ length: 9 }).map((_, idx) => (
                <motion.div
                  key={idx}
                  className={`w-10 h-10 border-2 border-green-500 flex items-center justify-center text-sm font-bold ${
                    idx <= filterPosition && isAnimating
                      ? "bg-green-500 text-white"
                      : "bg-white text-gray-500"
                  }`}
                  animate={{
                    backgroundColor:
                      idx <= filterPosition && isAnimating
                        ? "#10b981"
                        : "#ffffff",
                    scale:
                      idx === filterPosition && isAnimating ? [1, 1.2, 1] : 1,
                  }}
                  transition={{ duration: 0.3 }}
                >
                  {idx <= filterPosition ? "✓" : "?"}
                </motion.div>
              ))}
            </div>
          </div>
        </div>
      </div>

      <div className="text-center">
        <button
          onClick={startConvolution}
          disabled={isAnimating}
          className="button-primary disabled:opacity-50"
        >
          {isAnimating ? "🔄 Computing..." : "🚀 Start Convolution"}
        </button>
      </div>

      <div className="mt-4 text-sm text-gray-600">
        <p>
          <strong>How it works:</strong> The filter slides over the input image,
          computing dot products at each position to create feature maps that
          detect patterns like edges and shapes.
        </p>
      </div>
    </div>
  );
};

const LSTMDemo = () => {
  const [timeStep, setTimeStep] = useState(0);
  const [isRunning, setIsRunning] = useState(false);
  const [cellState, setCellState] = useState(0.5);
  const [hiddenState, setHiddenState] = useState(0.3);

  const sequence = ["Hello", "world", "this", "is", "LSTM"];
  const gates = {
    forget: 0.7,
    input: 0.8,
    output: 0.6,
  };

  const runLSTM = () => {
    setIsRunning(true);
    setTimeStep(0);

    let step = 0;
    const interval = setInterval(() => {
      step += 1;
      setTimeStep(step);

      // Simulate LSTM computations
      setCellState((prev) => prev * gates.forget + Math.random() * 0.3);
      setHiddenState((prev) => Math.tanh(cellState) * gates.output);

      if (step >= sequence.length) {
        clearInterval(interval);
        setIsRunning(false);
      }
    }, 1500);
  };

  return (
    <div className="bg-white p-6 rounded-lg border border-gray-200">
      <h4 className="text-lg font-semibold mb-4">LSTM Memory Gates</h4>

      {/* Sequence Input */}
      <div className="mb-6">
        <h5 className="text-sm font-semibold mb-2">Input Sequence:</h5>
        <div className="flex gap-2 justify-center">
          {sequence.map((word, idx) => (
            <motion.div
              key={idx}
              className={`px-4 py-2 rounded border-2 font-medium ${
                idx < timeStep
                  ? "border-blue-500 bg-blue-100 text-blue-700"
                  : idx === timeStep
                    ? "border-orange-500 bg-orange-100 text-orange-700"
                    : "border-gray-300 bg-gray-100 text-gray-500"
              }`}
              animate={{
                scale: idx === timeStep ? [1, 1.1, 1] : 1,
                y: idx === timeStep ? [0, -5, 0] : 0,
              }}
              transition={{ duration: 0.5 }}
            >
              {word}
            </motion.div>
          ))}
        </div>
      </div>

      {/* LSTM Architecture */}
      <div className="bg-gray-50 p-6 rounded-lg mb-6">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {/* Forget Gate */}
          <div className="text-center">
            <motion.div
              className="w-16 h-16 bg-red-500 rounded-full flex items-center justify-center text-white font-bold mx-auto mb-2"
              animate={{
                scale: isRunning ? [1, 1.2, 1] : 1,
                opacity: gates.forget,
              }}
              transition={{ duration: 0.8, repeat: isRunning ? Infinity : 0 }}
            >
              f
            </motion.div>
            <h6 className="font-semibold text-sm">Forget Gate</h6>
            <p className="text-xs text-gray-600">Decides what to forget</p>
            <div className="mt-2 text-sm font-mono">{gates.forget}</div>
          </div>

          {/* Input Gate */}
          <div className="text-center">
            <motion.div
              className="w-16 h-16 bg-green-500 rounded-full flex items-center justify-center text-white font-bold mx-auto mb-2"
              animate={{
                scale: isRunning ? [1, 1.2, 1] : 1,
                opacity: gates.input,
              }}
              transition={{
                duration: 0.8,
                delay: 0.2,
                repeat: isRunning ? Infinity : 0,
              }}
            >
              i
            </motion.div>
            <h6 className="font-semibold text-sm">Input Gate</h6>
            <p className="text-xs text-gray-600">Decides what to store</p>
            <div className="mt-2 text-sm font-mono">{gates.input}</div>
          </div>

          {/* Output Gate */}
          <div className="text-center">
            <motion.div
              className="w-16 h-16 bg-purple-500 rounded-full flex items-center justify-center text-white font-bold mx-auto mb-2"
              animate={{
                scale: isRunning ? [1, 1.2, 1] : 1,
                opacity: gates.output,
              }}
              transition={{
                duration: 0.8,
                delay: 0.4,
                repeat: isRunning ? Infinity : 0,
              }}
            >
              o
            </motion.div>
            <h6 className="font-semibold text-sm">Output Gate</h6>
            <p className="text-xs text-gray-600">Controls output</p>
            <div className="mt-2 text-sm font-mono">{gates.output}</div>
          </div>
        </div>

        {/* Cell State and Hidden State */}
        <div className="mt-6 grid grid-cols-2 gap-4">
          <div className="text-center">
            <motion.div
              className="w-full h-8 bg-blue-200 rounded-full relative overflow-hidden"
              animate={{ opacity: isRunning ? [0.5, 1, 0.5] : 1 }}
              transition={{ duration: 1, repeat: isRunning ? Infinity : 0 }}
            >
              <motion.div
                className="h-full bg-blue-500 rounded-full"
                style={{ width: `${cellState * 100}%` }}
                animate={{ width: `${cellState * 100}%` }}
                transition={{ duration: 0.5 }}
              />
            </motion.div>
            <p className="text-sm font-semibold mt-2">
              Cell State: {cellState.toFixed(2)}
            </p>
            <p className="text-xs text-gray-600">Long-term memory</p>
          </div>

          <div className="text-center">
            <motion.div
              className="w-full h-8 bg-orange-200 rounded-full relative overflow-hidden"
              animate={{ opacity: isRunning ? [0.5, 1, 0.5] : 1 }}
              transition={{
                duration: 1,
                delay: 0.3,
                repeat: isRunning ? Infinity : 0,
              }}
            >
              <motion.div
                className="h-full bg-orange-500 rounded-full"
                style={{ width: `${hiddenState * 100}%` }}
                animate={{ width: `${hiddenState * 100}%` }}
                transition={{ duration: 0.5 }}
              />
            </motion.div>
            <p className="text-sm font-semibold mt-2">
              Hidden State: {hiddenState.toFixed(2)}
            </p>
            <p className="text-xs text-gray-600">Short-term memory</p>
          </div>
        </div>
      </div>

      <div className="text-center">
        <button
          onClick={runLSTM}
          disabled={isRunning}
          className="button-primary disabled:opacity-50"
        >
          {isRunning ? "🧠 Processing..." : "▶️ Run LSTM Sequence"}
        </button>
      </div>
    </div>
  );
};

const BackpropDemo = () => {
  const [isTraining, setIsTraining] = useState(false);
  const [error, setError] = useState(0.8);
  const [epoch, setEpoch] = useState(0);

  const runBackprop = () => {
    setIsTraining(true);
    setEpoch(0);
    setError(0.8);

    let currentEpoch = 0;
    let currentError = 0.8;

    const interval = setInterval(() => {
      currentEpoch += 1;
      currentError = Math.max(
        0.05,
        currentError * 0.85 + Math.random() * 0.1 - 0.05,
      );

      setEpoch(currentEpoch);
      setError(currentError);

      if (currentEpoch >= 10) {
        clearInterval(interval);
        setIsTraining(false);
      }
    }, 800);
  };

  return (
    <div className="bg-white p-6 rounded-lg border border-gray-200">
      <h4 className="text-lg font-semibold mb-4">Backpropagation Training</h4>

      {/* Neural Network Layers */}
      <div className="mb-6 bg-gray-50 p-6 rounded-lg">
        <div className="flex justify-between items-center">
          <div className="text-center">
            <div className="space-y-2">
              {[0, 1, 2].map((i) => (
                <motion.div
                  key={i}
                  className="w-8 h-8 bg-blue-500 rounded-full"
                  animate={{
                    scale: isTraining ? [1, 1.2, 1] : 1,
                    backgroundColor: isTraining
                      ? ["#3b82f6", "#10b981", "#3b82f6"]
                      : "#3b82f6",
                  }}
                  transition={{
                    duration: 0.6,
                    delay: i * 0.1,
                    repeat: isTraining ? Infinity : 0,
                  }}
                />
              ))}
            </div>
            <p className="text-xs mt-2 font-semibold">Input</p>
          </div>

          <div className="text-center">
            <div className="space-y-2">
              {[0, 1, 2, 3].map((i) => (
                <motion.div
                  key={i}
                  className="w-8 h-8 bg-purple-500 rounded-full"
                  animate={{
                    scale: isTraining ? [1, 1.2, 1] : 1,
                    backgroundColor: isTraining
                      ? ["#a855f7", "#f59e0b", "#a855f7"]
                      : "#a855f7",
                  }}
                  transition={{
                    duration: 0.6,
                    delay: 0.2 + i * 0.1,
                    repeat: isTraining ? Infinity : 0,
                  }}
                />
              ))}
            </div>
            <p className="text-xs mt-2 font-semibold">Hidden</p>
          </div>

          <div className="text-center">
            <div className="space-y-2">
              {[0, 1].map((i) => (
                <motion.div
                  key={i}
                  className="w-8 h-8 bg-green-500 rounded-full"
                  animate={{
                    scale: isTraining ? [1, 1.2, 1] : 1,
                    backgroundColor: isTraining
                      ? ["#10b981", "#ef4444", "#10b981"]
                      : "#10b981",
                  }}
                  transition={{
                    duration: 0.6,
                    delay: 0.4 + i * 0.1,
                    repeat: isTraining ? Infinity : 0,
                  }}
                />
              ))}
            </div>
            <p className="text-xs mt-2 font-semibold">Output</p>
          </div>
        </div>

        {/* Error Signal */}
        {isTraining && (
          <motion.div
            className="mt-4 text-center"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
          >
            <motion.div
              className="inline-flex items-center gap-2 px-4 py-2 bg-red-100 border border-red-300 rounded-full"
              animate={{ x: [200, 0, -200] }}
              transition={{ duration: 2, repeat: Infinity, ease: "linear" }}
            >
              <span className="text-red-600 font-bold">
                📉 Error: {error.toFixed(3)}
              </span>
            </motion.div>
          </motion.div>
        )}
      </div>

      {/* Training Progress */}
      <div className="grid grid-cols-2 gap-4 mb-6">
        <div className="text-center">
          <div className="text-2xl font-bold text-blue-600">{epoch}</div>
          <div className="text-sm text-gray-600">Epoch</div>
        </div>
        <div className="text-center">
          <div className="text-2xl font-bold text-red-600">
            {error.toFixed(3)}
          </div>
          <div className="text-sm text-gray-600">Error</div>
        </div>
      </div>

      <div className="text-center">
        <button
          onClick={runBackprop}
          disabled={isTraining}
          className="button-primary disabled:opacity-50"
        >
          {isTraining ? "🔄 Training..." : "🎯 Start Training"}
        </button>
      </div>

      <div className="mt-4 text-sm text-gray-600">
        <p>
          <strong>Backpropagation:</strong> The error flows backward through the
          network, adjusting weights to minimize prediction errors through
          gradient descent.
        </p>
      </div>
    </div>
  );
};

const TransformerDemo = () => {
  const [isProcessing, setIsProcessing] = useState(false);
  const [currentStep, setCurrentStep] = useState(0);

  const sentence = ["The", "cat", "sat", "on", "mat"];
  const steps = ["Self-Attention", "Feed Forward", "Layer Norm", "Output"];

  const runTransformer = () => {
    setIsProcessing(true);
    setCurrentStep(0);

    let step = 0;
    const interval = setInterval(() => {
      step += 1;
      setCurrentStep(step);

      if (step >= steps.length) {
        clearInterval(interval);
        setTimeout(() => setIsProcessing(false), 1000);
      }
    }, 1500);
  };

  return (
    <div className="bg-white p-6 rounded-lg border border-gray-200">
      <h4 className="text-lg font-semibold mb-4">Transformer Architecture</h4>

      {/* Input Tokens */}
      <div className="mb-6">
        <h5 className="text-sm font-semibold mb-2">Input Sequence:</h5>
        <div className="flex gap-2 justify-center">
          {sentence.map((word, idx) => (
            <motion.div
              key={idx}
              className="px-3 py-2 bg-blue-100 border border-blue-300 rounded font-medium"
              animate={{
                y: isProcessing ? [0, -10, 0] : 0,
                backgroundColor: isProcessing
                  ? ["#dbeafe", "#bfdbfe", "#dbeafe"]
                  : "#dbeafe",
              }}
              transition={{
                duration: 1,
                delay: idx * 0.2,
                repeat: isProcessing ? Infinity : 0,
              }}
            >
              {word}
            </motion.div>
          ))}
        </div>
      </div>

      {/* Multi-Head Attention Visualization */}
      <div className="mb-6 bg-gray-50 p-4 rounded-lg">
        <h5 className="text-sm font-semibold mb-3">
          Multi-Head Self-Attention
        </h5>
        <div className="grid grid-cols-8 gap-1 mb-4">
          {Array.from({ length: 40 }).map((_, idx) => {
            const row = Math.floor(idx / 8);
            const col = idx % 8;
            const isActive = currentStep >= 1 && isProcessing;
            const intensity = Math.random();

            return (
              <motion.div
                key={idx}
                className="w-6 h-6 border border-gray-300 rounded-sm"
                animate={{
                  backgroundColor: isActive
                    ? `rgba(59, 130, 246, ${intensity})`
                    : "#ffffff",
                  scale: isActive && intensity > 0.7 ? [1, 1.2, 1] : 1,
                }}
                transition={{ duration: 0.5, delay: (row + col) * 0.05 }}
              />
            );
          })}
        </div>
        <p className="text-xs text-gray-600">
          Attention matrix showing relationships between all token pairs
        </p>
      </div>

      {/* Processing Steps */}
      <div className="mb-6">
        <h5 className="text-sm font-semibold mb-3">Processing Pipeline:</h5>
        <div className="flex justify-between">
          {steps.map((step, idx) => (
            <motion.div
              key={idx}
              className={`px-4 py-2 rounded-lg border-2 text-sm font-medium ${
                idx < currentStep
                  ? "border-green-500 bg-green-100 text-green-700"
                  : idx === currentStep
                    ? "border-orange-500 bg-orange-100 text-orange-700"
                    : "border-gray-300 bg-gray-100 text-gray-500"
              }`}
              animate={{
                scale: idx === currentStep ? [1, 1.1, 1] : 1,
                rotate: idx === currentStep ? [0, 5, -5, 0] : 0,
              }}
              transition={{
                duration: 0.5,
                repeat: idx === currentStep ? 3 : 0,
              }}
            >
              {step}
            </motion.div>
          ))}
        </div>
      </div>

      <div className="text-center">
        <button
          onClick={runTransformer}
          disabled={isProcessing}
          className="button-primary disabled:opacity-50"
        >
          {isProcessing ? "⚡ Processing..." : "🧠 Run Transformer"}
        </button>
      </div>

      <div className="mt-4 text-sm text-gray-600">
        <p>
          <strong>Key Innovation:</strong> Parallel processing using
          self-attention allows understanding of relationships between all words
          simultaneously, not sequentially.
        </p>
      </div>
    </div>
  );
};

const GANDemo = () => {
  const [isTraining, setIsTraining] = useState(false);
  const [epoch, setEpoch] = useState(0);
  const [generatorLoss, setGeneratorLoss] = useState(2.5);
  const [discriminatorLoss, setDiscriminatorLoss] = useState(0.3);
  const [imageQuality, setImageQuality] = useState(0.1);

  const startGANTraining = () => {
    setIsTraining(true);
    setEpoch(0);
    setGeneratorLoss(2.5);
    setDiscriminatorLoss(0.3);
    setImageQuality(0.1);

    let currentEpoch = 0;
    const interval = setInterval(() => {
      currentEpoch += 1;
      setEpoch(currentEpoch);

      // Simulate adversarial training dynamics
      setGeneratorLoss((prev) =>
        Math.max(0.1, prev * 0.95 + Math.random() * 0.2 - 0.1),
      );
      setDiscriminatorLoss((prev) =>
        Math.max(0.1, prev + Math.random() * 0.1 - 0.05),
      );
      setImageQuality((prev) =>
        Math.min(1.0, prev + 0.08 + Math.random() * 0.02),
      );

      if (currentEpoch >= 15) {
        clearInterval(interval);
        setIsTraining(false);
      }
    }, 600);
  };

  return (
    <div className="bg-white p-6 rounded-lg border border-gray-200">
      <h4 className="text-lg font-semibold mb-4">GAN Training Battle</h4>

      {/* Generator vs Discriminator */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
        {/* Generator */}
        <div className="text-center">
          <motion.div
            className="w-24 h-24 bg-gradient-to-r from-purple-500 to-purple-700 rounded-lg mx-auto mb-4 flex items-center justify-center text-white font-bold text-xl"
            animate={{
              scale: isTraining ? [1, 1.1, 1] : 1,
              rotate: isTraining ? [0, 5, -5, 0] : 0,
            }}
            transition={{ duration: 1, repeat: isTraining ? Infinity : 0 }}
          >
            G
          </motion.div>
          <h5 className="font-semibold text-purple-700">Generator</h5>
          <p className="text-sm text-gray-600">Creates fake images</p>
          <div className="mt-2">
            <div className="text-sm font-mono">
              Loss: {generatorLoss.toFixed(2)}
            </div>
            <motion.div
              className="w-full h-2 bg-purple-200 rounded-full mt-1 overflow-hidden"
              animate={{ opacity: isTraining ? [0.5, 1, 0.5] : 1 }}
              transition={{ duration: 0.8, repeat: isTraining ? Infinity : 0 }}
            >
              <motion.div
                className="h-full bg-purple-500 rounded-full"
                animate={{ width: `${Math.max(0, 100 - generatorLoss * 20)}%` }}
                transition={{ duration: 0.5 }}
              />
            </motion.div>
          </div>
        </div>

        {/* VS */}
        <div className="hidden md:flex items-center justify-center">
          <motion.div
            className="text-4xl font-bold text-red-500"
            animate={{
              scale: isTraining ? [1, 1.3, 1] : 1,
              rotate: isTraining ? [0, 180, 360] : 0,
            }}
            transition={{ duration: 2, repeat: isTraining ? Infinity : 0 }}
          >
            ⚔️
          </motion.div>
        </div>

        {/* Discriminator */}
        <div className="text-center">
          <motion.div
            className="w-24 h-24 bg-gradient-to-r from-green-500 to-green-700 rounded-lg mx-auto mb-4 flex items-center justify-center text-white font-bold text-xl"
            animate={{
              scale: isTraining ? [1, 1.1, 1] : 1,
              rotate: isTraining ? [0, -5, 5, 0] : 0,
            }}
            transition={{
              duration: 1,
              delay: 0.5,
              repeat: isTraining ? Infinity : 0,
            }}
          >
            D
          </motion.div>
          <h5 className="font-semibold text-green-700">Discriminator</h5>
          <p className="text-sm text-gray-600">Detects fake images</p>
          <div className="mt-2">
            <div className="text-sm font-mono">
              Loss: {discriminatorLoss.toFixed(2)}
            </div>
            <motion.div
              className="w-full h-2 bg-green-200 rounded-full mt-1 overflow-hidden"
              animate={{ opacity: isTraining ? [0.5, 1, 0.5] : 1 }}
              transition={{
                duration: 0.8,
                delay: 0.4,
                repeat: isTraining ? Infinity : 0,
              }}
            >
              <motion.div
                className="h-full bg-green-500 rounded-full"
                animate={{
                  width: `${Math.min(100, discriminatorLoss * 100)}%`,
                }}
                transition={{ duration: 0.5 }}
              />
            </motion.div>
          </div>
        </div>
      </div>

      {/* Generated Images Quality */}
      <div className="mb-6 bg-gray-50 p-4 rounded-lg">
        <h5 className="text-sm font-semibold mb-3">Generated Image Quality</h5>
        <div className="grid grid-cols-5 gap-2 mb-4">
          {Array.from({ length: 10 }).map((_, idx) => {
            const quality = Math.min(
              1,
              imageQuality + (Math.random() - 0.5) * 0.2,
            );
            return (
              <motion.div
                key={idx}
                className="w-12 h-12 border border-gray-300 rounded"
                style={{
                  background: `linear-gradient(45deg,
                    rgba(255,255,255,${1 - quality}),
                    rgba(59,130,246,${quality}),
                    rgba(147,51,234,${quality * 0.8})
                  )`,
                }}
                animate={{
                  opacity: isTraining ? [0.3, 1, 0.3] : 1,
                  scale: isTraining ? [0.9, 1.1, 0.9] : 1,
                }}
                transition={{
                  duration: 1,
                  delay: idx * 0.1,
                  repeat: isTraining ? Infinity : 0,
                }}
              />
            );
          })}
        </div>
        <div className="text-center">
          <div className="text-lg font-bold text-blue-600">
            {(imageQuality * 100).toFixed(0)}%
          </div>
          <div className="text-sm text-gray-600">Quality Score</div>
        </div>
      </div>

      {/* Training Stats */}
      <div className="text-center mb-6">
        <div className="text-xl font-bold text-gray-800">Epoch: {epoch}</div>
      </div>

      <div className="text-center">
        <button
          onClick={startGANTraining}
          disabled={isTraining}
          className="button-primary disabled:opacity-50"
        >
          {isTraining ? "⚔️ Training Battle..." : "🎨 Start GAN Training"}
        </button>
      </div>

      <div className="mt-4 text-sm text-gray-600">
        <p>
          <strong>Adversarial Training:</strong> The generator learns to create
          better fakes while the discriminator learns to detect them better,
          pushing both to improve.
        </p>
      </div>
    </div>
  );
};

const ResNetDemo = () => {
  const [showSkipConnections, setShowSkipConnections] = useState(false);
  const [layerDepth, setLayerDepth] = useState(8);

  const toggleDemo = () => {
    setShowSkipConnections(!showSkipConnections);
  };

  return (
    <div className="bg-white p-6 rounded-lg border border-gray-200">
      <h4 className="text-lg font-semibold mb-4">ResNet Skip Connections</h4>

      <div className="mb-6">
        <label className="block text-sm font-medium mb-2">
          Network Depth: {layerDepth} layers
        </label>
        <input
          type="range"
          min="4"
          max="16"
          value={layerDepth}
          onChange={(e) => setLayerDepth(parseInt(e.target.value))}
          className="w-full"
        />
      </div>

      {/* Network Visualization */}
      <div className="mb-6 bg-gray-50 p-6 rounded-lg overflow-x-auto">
        <div className="flex items-center gap-4 min-w-max">
          {Array.from({ length: layerDepth }).map((_, idx) => (
            <div key={idx} className="relative">
              {/* Layer */}
              <motion.div
                className="w-12 h-16 bg-blue-500 rounded flex items-center justify-center text-white font-bold"
                animate={{
                  opacity: showSkipConnections ? [0.5, 1, 0.5] : 1,
                  y: showSkipConnections ? [0, -5, 0] : 0,
                }}
                transition={{
                  duration: 0.8,
                  delay: idx * 0.1,
                  repeat: showSkipConnections ? Infinity : 0,
                }}
              >
                L{idx + 1}
              </motion.div>

              {/* Skip Connection */}
              {showSkipConnections && idx % 2 === 0 && idx < layerDepth - 2 && (
                <motion.div
                  className="absolute top-0 left-6"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                >
                  <svg className="w-24 h-16" viewBox="0 0 96 64">
                    <motion.path
                      d="M 0 32 Q 48 0 96 32"
                      stroke="#ef4444"
                      strokeWidth="3"
                      fill="none"
                      strokeDasharray="5,5"
                      initial={{ pathLength: 0 }}
                      animate={{ pathLength: 1 }}
                      transition={{ duration: 1, delay: idx * 0.2 }}
                    />
                    <motion.circle
                      cx="48"
                      cy="16"
                      r="3"
                      fill="#ef4444"
                      initial={{ scale: 0 }}
                      animate={{ scale: [0, 1.5, 1] }}
                      transition={{ duration: 0.5, delay: 1 + idx * 0.2 }}
                    />
                  </svg>
                  <div className="absolute top-0 left-8 text-xs text-red-600 font-bold bg-white px-1 rounded">
                    Skip
                  </div>
                </motion.div>
              )}
            </div>
          ))}
        </div>
      </div>

      {/* Information Flow */}
      <div className="mb-6">
        <div className="grid grid-cols-2 gap-4">
          <div className="text-center">
            <div className="text-lg font-bold text-blue-600">
              Without Skip Connections
            </div>
            <div className="text-sm text-gray-600">
              Information degrades through deep layers
            </div>
            <motion.div
              className="w-full h-4 bg-gradient-to-r from-blue-500 to-gray-300 rounded-full mt-2"
              animate={{ opacity: showSkipConnections ? 0.3 : 1 }}
            />
          </div>
          <div className="text-center">
            <div className="text-lg font-bold text-red-600">
              With Skip Connections
            </div>
            <div className="text-sm text-gray-600">
              Information preserved across layers
            </div>
            <motion.div
              className="w-full h-4 bg-gradient-to-r from-red-500 to-red-400 rounded-full mt-2"
              animate={{ opacity: showSkipConnections ? 1 : 0.3 }}
            />
          </div>
        </div>
      </div>

      <div className="text-center">
        <button onClick={toggleDemo} className="button-primary">
          {showSkipConnections
            ? "🚫 Hide Skip Connections"
            : "✨ Show Skip Connections"}
        </button>
      </div>

      <div className="mt-4 text-sm text-gray-600">
        <p>
          <strong>Innovation:</strong> Skip connections allow gradients to flow
          directly to earlier layers, enabling training of very deep networks
          (50+ layers).
        </p>
      </div>
    </div>
  );
};

const DQNDemo = () => {
  const [gameState, setGameState] = useState({
    agent: 2,
    goal: 7,
    obstacles: [4, 5],
  });
  const [qValues, setQValues] = useState(
    Array(10)
      .fill(null)
      .map(() => [Math.random(), Math.random()]),
  );
  const [episode, setEpisode] = useState(0);
  const [isTraining, setIsTraining] = useState(false);

  const actions = ["Left", "Right"];

  const trainDQN = () => {
    setIsTraining(true);
    setEpisode(0);

    let currentEpisode = 0;
    const interval = setInterval(() => {
      currentEpisode += 1;
      setEpisode(currentEpisode);

      // Simulate learning
      setQValues((prev) =>
        prev.map((values, idx) => {
          if (idx === gameState.goal) return [5, 5]; // High value for goal
          if (gameState.obstacles.includes(idx)) return [-5, -5]; // Low value for obstacles
          return values.map((v) => v + (Math.random() - 0.5) * 0.1);
        }),
      );

      if (currentEpisode >= 20) {
        clearInterval(interval);
        setIsTraining(false);
      }
    }, 300);
  };

  const moveAgent = (action: number) => {
    if (isTraining) return;

    setGameState((prev) => {
      const newPos =
        action === 0
          ? Math.max(0, prev.agent - 1)
          : Math.min(9, prev.agent + 1);
      return { ...prev, agent: newPos };
    });
  };

  return (
    <div className="bg-white p-6 rounded-lg border border-gray-200">
      <h4 className="text-lg font-semibold mb-4">DQN Game Environment</h4>

      {/* Game Grid */}
      <div className="mb-6">
        <h5 className="text-sm font-semibold mb-2">Game State:</h5>
        <div className="grid grid-cols-10 gap-1 mb-4">
          {Array.from({ length: 10 }).map((_, idx) => {
            const isAgent = idx === gameState.agent;
            const isGoal = idx === gameState.goal;
            const isObstacle = gameState.obstacles.includes(idx);

            return (
              <motion.div
                key={idx}
                className={`w-10 h-10 border-2 rounded flex items-center justify-center font-bold ${
                  isAgent
                    ? "bg-blue-500 border-blue-700 text-white"
                    : isGoal
                      ? "bg-green-500 border-green-700 text-white"
                      : isObstacle
                        ? "bg-red-500 border-red-700 text-white"
                        : "bg-gray-100 border-gray-300"
                }`}
                animate={{
                  scale: isAgent ? [1, 1.2, 1] : 1,
                  rotate: isTraining && isAgent ? [0, 10, -10, 0] : 0,
                }}
                transition={{
                  duration: 0.5,
                  repeat: isTraining && isAgent ? Infinity : 0,
                }}
              >
                {isAgent ? "🤖" : isGoal ? "🏆" : isObstacle ? "⚠️" : idx}
              </motion.div>
            );
          })}
        </div>
      </div>

      {/* Q-Values */}
      <div className="mb-6 bg-gray-50 p-4 rounded-lg">
        <h5 className="text-sm font-semibold mb-2">
          Q-Values (State-Action Values):
        </h5>
        <div className="grid grid-cols-5 gap-2 text-xs">
          {qValues.slice(0, 5).map((values, idx) => (
            <div key={idx} className="text-center">
              <div className="font-semibold">State {idx}</div>
              <div className="space-y-1">
                {values.map((value, actionIdx) => (
                  <motion.div
                    key={actionIdx}
                    className={`px-2 py-1 rounded text-white font-mono ${
                      value > 0 ? "bg-green-500" : "bg-red-500"
                    }`}
                    animate={{
                      scale: isTraining ? [1, 1.1, 1] : 1,
                      backgroundColor: isTraining
                        ? value > 0
                          ? ["#10b981", "#059669", "#10b981"]
                          : ["#ef4444", "#dc2626", "#ef4444"]
                        : value > 0
                          ? "#10b981"
                          : "#ef4444",
                    }}
                    transition={{
                      duration: 0.5,
                      delay: idx * 0.1 + actionIdx * 0.05,
                    }}
                  >
                    {actions[actionIdx]}: {value.toFixed(1)}
                  </motion.div>
                ))}
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Controls */}
      <div className="grid grid-cols-2 gap-4 mb-6">
        <button
          onClick={() => moveAgent(0)}
          disabled={isTraining}
          className="button-secondary disabled:opacity-50"
        >
          ← Move Left
        </button>
        <button
          onClick={() => moveAgent(1)}
          disabled={isTraining}
          className="button-secondary disabled:opacity-50"
        >
          Move Right →
        </button>
      </div>

      <div className="text-center mb-4">
        <div className="text-lg font-bold text-blue-600">
          Episode: {episode}
        </div>
      </div>

      <div className="text-center">
        <button
          onClick={trainDQN}
          disabled={isTraining}
          className="button-primary disabled:opacity-50"
        >
          {isTraining ? "🧠 Learning..." : "🎮 Start DQN Training"}
        </button>
      </div>

      <div className="mt-4 text-sm text-gray-600">
        <p>
          <strong>Deep Q-Learning:</strong> Combines neural networks with
          reinforcement learning to learn optimal policies for complex
          environments from raw observations.
        </p>
      </div>
    </div>
  );
};

const SimpleDemo = ({ type }: { type: string }) => {
  switch (type) {
    case "transformer":
      return <TransformerDemo />;
    case "gan":
      return <GANDemo />;
    case "resnet":
      return <ResNetDemo />;
    case "dqn":
      return <DQNDemo />;
    default:
      return (
        <div className="bg-white p-6 rounded-lg border border-gray-200">
          <h4 className="text-lg font-semibold mb-4">How it Works</h4>
          <p className="text-gray-700">
            This fundamental AI discovery shaped the field in significant ways.
          </p>
        </div>
      );
  }
};

const renderDemo = (discovery: Discovery) => {
  switch (discovery.demoType) {
    case "perceptron":
      return <PerceptronDemo />;
    case "attention":
      return <AttentionDemo />;
    case "qlearning":
      return <QLearningDemo />;
    default:
      return <SimpleDemo type={discovery.demoType} />;
  }
};

export default function AIDiscoveries() {
  const [selectedDiscovery, setSelectedDiscovery] = useState<Discovery | null>(
    null,
  );

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 to-white">
      <Navigation />

      <div className="container mx-auto px-6 py-12">
        <div className="text-center mb-16">
          <h1 className="text-6xl font-bold text-black mb-6">
            AI Landmark Discoveries
          </h1>
          <p className="text-xl text-gray-600 max-w-3xl mx-auto">
            Explore the 10 fundamental breakthroughs that shaped artificial
            intelligence, with interactive demos and insights into the brilliant
            minds behind them.
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-8 mb-12">
          {discoveries.map((discovery) => (
            <div
              key={discovery.id}
              className="bg-white rounded-xl shadow-lg border border-gray-200 overflow-hidden hover:shadow-xl transition-all duration-300 cursor-pointer"
              onClick={() => setSelectedDiscovery(discovery)}
            >
              <div className="p-6">
                <div className="flex items-center justify-between mb-4">
                  <span className="text-sm font-semibold text-blue-600 bg-blue-50 px-3 py-1 rounded-full">
                    {discovery.year}
                  </span>
                  <span className="text-2xl font-bold text-gray-400">
                    #{discovery.id}
                  </span>
                </div>

                <h3 className="text-xl font-bold text-black mb-3">
                  {discovery.title}
                </h3>

                <p className="text-gray-600 mb-4 line-clamp-3">
                  {discovery.description}
                </p>

                <div className="border-t pt-4">
                  <p className="text-sm font-medium text-gray-800 mb-1">
                    {discovery.discoverer}
                  </p>
                  <p className="text-xs text-gray-500">
                    Click to explore interactive demo
                  </p>
                </div>
              </div>
            </div>
          ))}
        </div>

        {selectedDiscovery && (
          <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50">
            <div className="bg-white rounded-xl max-w-6xl w-full max-h-[90vh] overflow-y-auto">
              <div className="p-8">
                <div className="flex justify-between items-start mb-6">
                  <div>
                    <h2 className="text-3xl font-bold text-black mb-2">
                      {selectedDiscovery.title}
                    </h2>
                    <p className="text-lg text-blue-600 font-semibold">
                      {selectedDiscovery.year} • {selectedDiscovery.discoverer}
                    </p>
                  </div>
                  <button
                    onClick={() => setSelectedDiscovery(null)}
                    className="text-gray-500 hover:text-gray-700 text-2xl"
                  >
                    ×
                  </button>
                </div>

                <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                  <div className="space-y-6">
                    <div>
                      <h3 className="text-xl font-semibold mb-3">
                        Description
                      </h3>
                      <p className="text-gray-700">
                        {selectedDiscovery.description}
                      </p>
                    </div>

                    <div>
                      <h3 className="text-xl font-semibold mb-3">Impact</h3>
                      <p className="text-gray-700">
                        {selectedDiscovery.impact}
                      </p>
                    </div>

                    <div>
                      <h3 className="text-xl font-semibold mb-3">
                        About the Discoverer
                      </h3>
                      <p className="text-gray-700">
                        {selectedDiscovery.discovererBio}
                      </p>
                    </div>

                    <div>
                      <h3 className="text-xl font-semibold mb-3">
                        Original Paper
                      </h3>
                      <p className="text-gray-700 mb-2">
                        {selectedDiscovery.paperTitle}
                      </p>
                      <a
                        href={selectedDiscovery.paperLink}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="button-primary inline-block"
                      >
                        Read Original Paper →
                      </a>
                    </div>
                  </div>

                  <div>
                    <h3 className="text-xl font-semibold mb-4">
                      Interactive Demo
                    </h3>
                    {renderDemo(selectedDiscovery)}
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        <div className="text-center">
          <Link to="/" className="button-secondary">
            ← Back to Portfolio
          </Link>
        </div>
      </div>
    </div>
  );
}
