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

const SimpleDemo = ({ type }: { type: string }) => {
  const demoContent = {
    backprop:
      "Backpropagation calculates how much each weight in a neural network contributed to the error, then adjusts weights to minimize that error. Think of it like learning from mistakes by tracing back through each decision.",
    cnn: "CNNs use filters that slide over images to detect features like edges, shapes, and patterns. Multiple layers detect increasingly complex features, from simple lines to complete objects.",
    lstm: "LSTMs have gates that control what information to remember, forget, or output. This allows them to maintain important context across long sequences, like remembering the subject of a sentence.",
    transformer:
      "Transformers process all parts of a sequence simultaneously using attention, allowing them to understand relationships between any two words regardless of distance.",
    gan: "GANs pit two networks against each other: a generator creates fake data while a discriminator tries to detect fakes. They improve together until the generator creates realistic data.",
    resnet:
      "ResNets use skip connections that allow information to flow directly to later layers, solving the problem of very deep networks losing information during training.",
    dqn: "DQNs combine deep learning with Q-learning to play games from raw pixels. They learn to predict the value of taking different actions in different game states.",
  };

  return (
    <div className="bg-white p-6 rounded-lg border border-gray-200">
      <h4 className="text-lg font-semibold mb-4">How it Works</h4>
      <p className="text-gray-700">
        {demoContent[type as keyof typeof demoContent]}
      </p>
    </div>
  );
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
