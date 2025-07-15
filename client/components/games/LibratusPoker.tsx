import { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";

interface Card {
  suit: "hearts" | "diamonds" | "clubs" | "spades";
  rank:
    | "2"
    | "3"
    | "4"
    | "5"
    | "6"
    | "7"
    | "8"
    | "9"
    | "10"
    | "J"
    | "Q"
    | "K"
    | "A";
}

interface Player {
  name: string;
  chips: number;
  cards: Card[];
  currentBet: number;
  isAI: boolean;
  lastAction: string;
}

const suits = {
  hearts: "♥",
  diamonds: "♦",
  clubs: "♣",
  spades: "♠",
};

const createDeck = (): Card[] => {
  const suits = ["hearts", "diamonds", "clubs", "spades"] as const;
  const ranks = [
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "10",
    "J",
    "Q",
    "K",
    "A",
  ] as const;

  const deck: Card[] = [];
  for (const suit of suits) {
    for (const rank of ranks) {
      deck.push({ suit, rank });
    }
  }
  return deck;
};

const shuffleDeck = (deck: Card[]): Card[] => {
  const shuffled = [...deck];
  for (let i = shuffled.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
  }
  return shuffled;
};

export default function LibratusPoker() {
  const [gamePhase, setGamePhase] = useState<
    "setup" | "playing" | "demo" | "ended"
  >("setup");
  const [players, setPlayers] = useState<Player[]>([
    {
      name: "You",
      chips: 10000,
      cards: [],
      currentBet: 0,
      isAI: false,
      lastAction: "",
    },
    {
      name: "Libratus",
      chips: 10000,
      cards: [],
      currentBet: 0,
      isAI: true,
      lastAction: "",
    },
  ]);
  const [communityCards, setCommunityCards] = useState<Card[]>([]);
  const [deck, setDeck] = useState<Card[]>([]);
  const [pot, setPot] = useState(0);
  const [currentPlayerIndex, setCurrentPlayerIndex] = useState(0);
  const [gameRound, setGameRound] = useState<
    "preflop" | "flop" | "turn" | "river" | "showdown"
  >("preflop");
  const [isThinking, setIsThinking] = useState(false);
  const [gameHistory, setGameHistory] = useState<string[]>([]);
  const [blinds, setBlinds] = useState({ small: 50, big: 100 });

  const resetGame = () => {
    const newDeck = shuffleDeck(createDeck());
    const newPlayers = [
      {
        name: "You",
        chips: 10000,
        cards: [],
        currentBet: 0,
        isAI: false,
        lastAction: "",
      },
      {
        name: "Libratus",
        chips: 10000,
        cards: [],
        currentBet: 0,
        isAI: true,
        lastAction: "",
      },
    ];

    setPlayers(newPlayers);
    setCommunityCards([]);
    setDeck(newDeck);
    setPot(0);
    setCurrentPlayerIndex(0);
    setGameRound("preflop");
    setIsThinking(false);
    setGameHistory([]);
    setGamePhase("setup");
  };

  const startGame = () => {
    const newDeck = shuffleDeck(createDeck());
    const newPlayers = [...players];

    // Deal cards
    for (let i = 0; i < 2; i++) {
      for (let j = 0; j < newPlayers.length; j++) {
        newPlayers[j].cards.push(newDeck.pop()!);
      }
    }

    // Set blinds
    newPlayers[0].currentBet = blinds.small;
    newPlayers[0].chips -= blinds.small;
    newPlayers[1].currentBet = blinds.big;
    newPlayers[1].chips -= blinds.big;

    setPlayers(newPlayers);
    setDeck(newDeck);
    setPot(blinds.small + blinds.big);
    setGamePhase("playing");
    setCurrentPlayerIndex(0);
    setGameHistory([
      "Game started",
      `Blinds posted: ${blinds.small}/${blinds.big}`,
    ]);
  };

  const startDemo = () => {
    startGame();
    setGamePhase("demo");
    setGameHistory([
      "Demo mode: Watch Libratus play",
      `Blinds posted: ${blinds.small}/${blinds.big}`,
    ]);
  };

  // AI Decision Making (Simplified)
  const makeAIDecision = (
    player: Player,
  ): "fold" | "call" | "raise" | "check" => {
    const handStrength = evaluateHandStrength(player.cards, communityCards);
    const potOdds = pot > 0 ? player.currentBet / pot : 0;

    // Simplified AI logic (real Libratus is much more sophisticated)
    if (handStrength > 0.7) {
      return Math.random() > 0.3 ? "raise" : "call";
    } else if (handStrength > 0.4) {
      return Math.random() > 0.5 ? "call" : "check";
    } else {
      return Math.random() > 0.6 ? "fold" : "check";
    }
  };

  const evaluateHandStrength = (
    holeCards: Card[],
    communityCards: Card[],
  ): number => {
    // Simplified hand evaluation (returns 0-1)
    const allCards = [...holeCards, ...communityCards];

    // Check for pairs, face cards, etc.
    let strength = 0;

    // High cards bonus
    for (const card of holeCards) {
      if (["J", "Q", "K", "A"].includes(card.rank)) {
        strength += 0.1;
      }
    }

    // Pair bonus
    const ranks = holeCards.map((c) => c.rank);
    if (ranks[0] === ranks[1]) {
      strength += 0.3;
    }

    // Suited bonus
    const suits = holeCards.map((c) => c.suit);
    if (suits[0] === suits[1]) {
      strength += 0.1;
    }

    return Math.min(strength, 1);
  };

  const handlePlayerAction = (
    action: "fold" | "call" | "raise" | "check",
    raiseAmount?: number,
  ) => {
    if (gamePhase !== "playing") return;

    const newPlayers = [...players];
    const currentPlayer = newPlayers[currentPlayerIndex];

    let actionDescription = "";

    switch (action) {
      case "fold":
        currentPlayer.lastAction = "Fold";
        actionDescription = `${currentPlayer.name} folds`;
        setGamePhase("ended");
        break;

      case "call":
        const callAmount = Math.max(
          0,
          Math.max(...newPlayers.map((p) => p.currentBet)) -
            currentPlayer.currentBet,
        );
        currentPlayer.chips -= callAmount;
        currentPlayer.currentBet += callAmount;
        currentPlayer.lastAction = `Call ${callAmount}`;
        actionDescription = `${currentPlayer.name} calls ${callAmount}`;
        setPot((prev) => prev + callAmount);
        break;

      case "raise":
        const raiseTotal = raiseAmount || 200;
        const currentBet = Math.max(...newPlayers.map((p) => p.currentBet));
        const totalRaise = raiseTotal - currentPlayer.currentBet;
        currentPlayer.chips -= totalRaise;
        currentPlayer.currentBet = raiseTotal;
        currentPlayer.lastAction = `Raise to ${raiseTotal}`;
        actionDescription = `${currentPlayer.name} raises to ${raiseTotal}`;
        setPot((prev) => prev + totalRaise);
        break;

      case "check":
        currentPlayer.lastAction = "Check";
        actionDescription = `${currentPlayer.name} checks`;
        break;
    }

    setPlayers(newPlayers);
    setGameHistory((prev) => [...prev, actionDescription]);
    setCurrentPlayerIndex((prev) => (prev + 1) % newPlayers.length);
  };

  // AI Turn
  useEffect(() => {
    if (gamePhase === "playing" && players[currentPlayerIndex]?.isAI) {
      setIsThinking(true);

      const timer = setTimeout(() => {
        const aiPlayer = players[currentPlayerIndex];
        const decision = makeAIDecision(aiPlayer);
        handlePlayerAction(decision, decision === "raise" ? 300 : undefined);
        setIsThinking(false);
      }, 2000);

      return () => clearTimeout(timer);
    }
  }, [currentPlayerIndex, gamePhase]);

  const getCardDisplay = (card: Card, isHidden: boolean = false) => {
    if (isHidden) {
      return (
        <div className="w-12 h-16 bg-blue-900 border border-blue-700 rounded-lg flex items-center justify-center">
          <span className="text-white text-xs">🂠</span>
        </div>
      );
    }

    const isRed = card.suit === "hearts" || card.suit === "diamonds";

    return (
      <div
        className={`w-12 h-16 bg-white border border-gray-300 rounded-lg flex flex-col items-center justify-center shadow-md ${isRed ? "text-red-600" : "text-black"}`}
      >
        <span className="text-xs font-bold">{card.rank}</span>
        <span className="text-lg">{suits[card.suit]}</span>
      </div>
    );
  };

  return (
    <div className="w-full max-w-6xl mx-auto p-6">
      {/* Header */}
      <div className="text-center mb-8">
        <h3 className="text-3xl font-bold text-gray-800 mb-2">
          Libratus Poker Demo
        </h3>
        <p className="text-gray-600">
          Experience the 2017 AI that mastered imperfect information games
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Poker Table */}
        <div className="lg:col-span-2">
          <div className="bg-green-800 rounded-2xl p-8 shadow-2xl relative">
            {/* Community Cards */}
            <div className="text-center mb-8">
              <h4 className="text-white text-lg font-bold mb-4">
                Community Cards
              </h4>
              <div className="flex justify-center gap-2 mb-4">
                {Array.from({ length: 5 }).map((_, idx) => (
                  <div key={idx}>
                    {communityCards[idx] ? (
                      getCardDisplay(communityCards[idx])
                    ) : (
                      <div className="w-12 h-16 border-2 border-dashed border-white/30 rounded-lg"></div>
                    )}
                  </div>
                ))}
              </div>

              {/* Pot */}
              <div className="bg-yellow-600 text-white px-4 py-2 rounded-full inline-block font-bold">
                Pot: ${pot}
              </div>
            </div>

            {/* Players */}
            <div className="grid grid-cols-2 gap-8">
              {players.map((player, idx) => (
                <div
                  key={idx}
                  className={`text-center ${currentPlayerIndex === idx ? "ring-2 ring-yellow-400 rounded-lg p-4" : "p-4"}`}
                >
                  <h5 className="text-white font-bold mb-2 flex items-center justify-center gap-2">
                    {player.name}
                    {player.isAI && (
                      <span className="text-xs bg-blue-600 px-2 py-1 rounded">
                        AI
                      </span>
                    )}
                    {isThinking && currentPlayerIndex === idx && (
                      <span className="text-xs bg-yellow-600 px-2 py-1 rounded animate-pulse">
                        Thinking...
                      </span>
                    )}
                  </h5>

                  {/* Player Cards */}
                  <div className="flex justify-center gap-2 mb-3">
                    {player.cards.map((card, cardIdx) => (
                      <div key={cardIdx}>
                        {getCardDisplay(
                          card,
                          player.isAI && gamePhase !== "ended",
                        )}
                      </div>
                    ))}
                  </div>

                  {/* Player Info */}
                  <div className="text-white text-sm space-y-1">
                    <div>Chips: ${player.chips}</div>
                    <div>Current Bet: ${player.currentBet}</div>
                    {player.lastAction && (
                      <div className="text-yellow-300 font-semibold">
                        {player.lastAction}
                      </div>
                    )}
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Control Panel */}
        <div className="space-y-6">
          {/* Game Status */}
          <div className="bg-white rounded-xl p-6 shadow-lg border border-gray-200">
            <h4 className="text-xl font-bold text-gray-800 mb-4">
              Game Status
            </h4>

            <div className="space-y-3">
              <div className="flex justify-between">
                <span className="text-gray-600">Round:</span>
                <span className="font-bold capitalize">{gameRound}</span>
              </div>

              <div className="flex justify-between">
                <span className="text-gray-600">Current Player:</span>
                <span className="font-bold">
                  {players[currentPlayerIndex]?.name || "None"}
                </span>
              </div>

              <div className="flex justify-between">
                <span className="text-gray-600">Pot Size:</span>
                <span className="font-bold">${pot}</span>
              </div>

              <div className="flex justify-between">
                <span className="text-gray-600">Blinds:</span>
                <span className="font-bold">
                  ${blinds.small}/${blinds.big}
                </span>
              </div>
            </div>

            {isThinking && (
              <div className="mt-4 p-3 bg-blue-50 rounded-lg">
                <div className="text-sm text-blue-800">
                  🤖 Libratus is calculating optimal strategy...
                </div>
                <div className="text-xs text-blue-600 mt-1">
                  Analyzing game tree and opponent modeling
                </div>
              </div>
            )}
          </div>

          {/* Player Actions */}
          {gamePhase === "playing" && !players[currentPlayerIndex]?.isAI && (
            <div className="bg-white rounded-xl p-6 shadow-lg border border-gray-200">
              <h4 className="text-xl font-bold text-gray-800 mb-4">
                Your Actions
              </h4>

              <div className="space-y-3">
                <motion.button
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.98 }}
                  onClick={() => handlePlayerAction("check")}
                  className="w-full py-2 bg-green-500 text-white rounded-lg font-semibold hover:bg-green-600 transition-all"
                >
                  Check
                </motion.button>

                <motion.button
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.98 }}
                  onClick={() => handlePlayerAction("call")}
                  className="w-full py-2 bg-blue-500 text-white rounded-lg font-semibold hover:bg-blue-600 transition-all"
                >
                  Call
                </motion.button>

                <motion.button
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.98 }}
                  onClick={() => handlePlayerAction("raise", 300)}
                  className="w-full py-2 bg-yellow-600 text-white rounded-lg font-semibold hover:bg-yellow-700 transition-all"
                >
                  Raise ($300)
                </motion.button>

                <motion.button
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.98 }}
                  onClick={() => handlePlayerAction("fold")}
                  className="w-full py-2 bg-red-500 text-white rounded-lg font-semibold hover:bg-red-600 transition-all"
                >
                  Fold
                </motion.button>
              </div>
            </div>
          )}

          {/* Game Controls */}
          <div className="bg-white rounded-xl p-6 shadow-lg border border-gray-200 space-y-3">
            <h4 className="text-xl font-bold text-gray-800 mb-4">Controls</h4>

            <motion.button
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
              onClick={startDemo}
              className="w-full py-3 bg-gradient-to-r from-red-600 to-red-700 text-white rounded-lg font-semibold hover:from-red-700 hover:to-red-800 transition-all"
            >
              🎬 Watch Libratus Demo
            </motion.button>

            <motion.button
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
              onClick={startGame}
              className="w-full py-3 bg-gradient-to-r from-green-500 to-green-600 text-white rounded-lg font-semibold hover:from-green-600 hover:to-green-700 transition-all"
            >
              🎮 Play vs Libratus
            </motion.button>

            <motion.button
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
              onClick={resetGame}
              className="w-full py-3 bg-gradient-to-r from-gray-500 to-gray-600 text-white rounded-lg font-semibold hover:from-gray-600 hover:to-gray-700 transition-all"
            >
              🔄 Reset Game
            </motion.button>
          </div>

          {/* Game History */}
          {gameHistory.length > 0 && (
            <div className="bg-white rounded-xl p-6 shadow-lg border border-gray-200">
              <h4 className="text-xl font-bold text-gray-800 mb-4">Game Log</h4>
              <div className="space-y-1 max-h-40 overflow-y-auto">
                {gameHistory.map((entry, idx) => (
                  <div key={idx} className="text-sm text-gray-700">
                    {entry}
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Historical Context */}
      <div className="mt-8 bg-gradient-to-r from-red-50 to-orange-50 rounded-xl p-6 border border-red-200">
        <h4 className="text-xl font-bold text-gray-800 mb-4">
          🏆 The Historic Match
        </h4>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div>
            <h5 className="font-bold text-gray-800 mb-2">Libratus' Triumph</h5>
            <p className="text-gray-700 text-sm leading-relaxed">
              In January 2017, Carnegie Mellon's Libratus defeated four of the
              world's best no-limit Texas Hold'em players in a 20-day tournament
              at Rivers Casino in Pittsburgh, winning by a statistically
              significant margin of over $1.8 million in chips.
            </p>
          </div>
          <div>
            <h5 className="font-bold text-gray-800 mb-2">
              Imperfect Information Mastery
            </h5>
            <p className="text-gray-700 text-sm leading-relaxed">
              Unlike chess or Go, poker involves hidden information, bluffing,
              and psychological warfare. Libratus used counterfactual regret
              minimization and real-time strategy computation to master these
              complex dynamics, proving AI could excel in deception-based games.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
