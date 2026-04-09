export type DemoAction = "BUY" | "SELL" | "HOLD";
export type DemoPositionAction = "OPEN" | "CLOSE" | "REDUCE" | "HOLD_POSITION";

export interface DemoRecommendation {
  action: DemoAction;
  positionAction: DemoPositionAction;
  tradeStyle: "swing_trade" | "day_trade";
  timeframeMode: "daily_only" | "intraday_ready";
  entryPrice?: number;
  stopPrice?: number;
  target1?: number;
  positionSize?: number;
  forecastConfidence: number;
  ftaScore: number;
  probabilityOfSuccess: number;
  rationale: string;
  rejectionReason?: string;
}

export interface DemoTickerFrame {
  ticker: string;
  latestClose: number;
  changePct: number;
  recommendation: DemoRecommendation;
  priceSeries: { label: string; close: number }[];
  decisionLog: {
    time: string;
    action: DemoAction;
    positionAction: DemoPositionAction;
    note: string;
  }[];
  inventory: {
    dailyBars: number;
    intradayBars: number;
    hasRealIntraday: boolean;
    fundamentalsFresh: boolean;
    earningsFresh: boolean;
    newsFresh: boolean;
    providerIndicators: string[];
  };
}

export interface DemoFrame {
  id: string;
  label: string;
  summary: string;
  generatedAt: string;
  marketSession: "pre-market" | "market-hours" | "after-hours";
  loopStatus: "idle" | "running";
  requestBudget: {
    used: number;
    remaining: number;
    limit: number;
  };
  systemPortfolio: {
    equity: number;
    cash: number;
    dailyPnl: number;
    realizedPnl: number;
    unrealizedPnl: number;
    openPositions: number;
    closedTrades: number;
  };
  ingestEvents: {
    time: string;
    ticker: string;
    endpoint: string;
    result: string;
    rows: number;
  }[];
  tickers: DemoTickerFrame[];
  eodSummary?: {
    tradesFinalized: number;
    adaptiveUpdate: string;
    metaModelRetrained: boolean;
    summary: string;
  };
}

const baseSeries = {
  AAPL: [
    { label: "Mon", close: 197.2 },
    { label: "Tue", close: 198.1 },
    { label: "Wed", close: 199.4 },
    { label: "Thu", close: 200.3 },
    { label: "Fri", close: 201.6 },
  ],
  NVDA: [
    { label: "Mon", close: 861.5 },
    { label: "Tue", close: 872.2 },
    { label: "Wed", close: 881.9 },
    { label: "Thu", close: 887.1 },
    { label: "Fri", close: 894.7 },
  ],
  MSFT: [
    { label: "Mon", close: 421.4 },
    { label: "Tue", close: 423.9 },
    { label: "Wed", close: 424.5 },
    { label: "Thu", close: 426.8 },
    { label: "Fri", close: 427.2 },
  ],
};

export const TRADE_SYSTEM_FRAMES: DemoFrame[] = [
  {
    id: "preopen-ingest",
    label: "09:20 ET · pre-market ingest",
    summary:
      "Local-first ingest backfills missing daily tails, refreshes high-value indicators, and keeps intraday disabled because no real 1h feed is present.",
    generatedAt: "2026-04-09T13:20:00.000Z",
    marketSession: "pre-market",
    loopStatus: "idle",
    requestBudget: { used: 5, remaining: 20, limit: 25 },
    systemPortfolio: {
      equity: 100000,
      cash: 100000,
      dailyPnl: 0,
      realizedPnl: 0,
      unrealizedPnl: 0,
      openPositions: 0,
      closedTrades: 0,
    },
    ingestEvents: [
      { time: "09:12", ticker: "AAPL", endpoint: "TIME_SERIES_DAILY", result: "tail refreshed", rows: 100 },
      { time: "09:14", ticker: "NVDA", endpoint: "TIME_SERIES_DAILY", result: "tail refreshed", rows: 100 },
      { time: "09:16", ticker: "AAPL", endpoint: "RSI", result: "stored locally", rows: 2 },
      { time: "09:17", ticker: "AAPL", endpoint: "OVERVIEW", result: "snapshot updated", rows: 9 },
    ],
    tickers: [
      {
        ticker: "AAPL",
        latestClose: 201.6,
        changePct: 0.0042,
        recommendation: {
          action: "HOLD",
          positionAction: "HOLD_POSITION",
          tradeStyle: "swing_trade",
          timeframeMode: "daily_only",
          forecastConfidence: 0.61,
          ftaScore: 0.58,
          probabilityOfSuccess: 0.57,
          rationale:
            "Daily features are loaded locally, but the opening scan has not run yet. The system remains in paper-only standby before the recommendation loop starts.",
          rejectionReason: "loop_not_started",
        },
        priceSeries: baseSeries.AAPL,
        decisionLog: [
          { time: "09:20", action: "HOLD", positionAction: "HOLD_POSITION", note: "Waiting for market-hours loop." },
        ],
        inventory: {
          dailyBars: 252,
          intradayBars: 0,
          hasRealIntraday: false,
          fundamentalsFresh: true,
          earningsFresh: true,
          newsFresh: false,
          providerIndicators: ["RSI", "SMA", "EMA"],
        },
      },
      {
        ticker: "NVDA",
        latestClose: 894.7,
        changePct: 0.0086,
        recommendation: {
          action: "HOLD",
          positionAction: "HOLD_POSITION",
          tradeStyle: "swing_trade",
          timeframeMode: "daily_only",
          forecastConfidence: 0.63,
          ftaScore: 0.61,
          probabilityOfSuccess: 0.59,
          rationale:
            "Inventory is fresh locally, but no recommendation is issued until the loop starts. Free-tier policy keeps the system from forcing unnecessary calls.",
          rejectionReason: "loop_not_started",
        },
        priceSeries: baseSeries.NVDA,
        decisionLog: [
          { time: "09:20", action: "HOLD", positionAction: "HOLD_POSITION", note: "Waiting for market-hours loop." },
        ],
        inventory: {
          dailyBars: 252,
          intradayBars: 0,
          hasRealIntraday: false,
          fundamentalsFresh: true,
          earningsFresh: true,
          newsFresh: false,
          providerIndicators: ["RSI", "SMA", "EMA"],
        },
      },
      {
        ticker: "MSFT",
        latestClose: 427.2,
        changePct: 0.0009,
        recommendation: {
          action: "HOLD",
          positionAction: "HOLD_POSITION",
          tradeStyle: "swing_trade",
          timeframeMode: "daily_only",
          forecastConfidence: 0.55,
          ftaScore: 0.49,
          probabilityOfSuccess: 0.52,
          rationale:
            "The local data store is healthy, but the score stack is below the open threshold before the first loop decision.",
          rejectionReason: "loop_not_started",
        },
        priceSeries: baseSeries.MSFT,
        decisionLog: [
          { time: "09:20", action: "HOLD", positionAction: "HOLD_POSITION", note: "Waiting for market-hours loop." },
        ],
        inventory: {
          dailyBars: 252,
          intradayBars: 0,
          hasRealIntraday: false,
          fundamentalsFresh: true,
          earningsFresh: true,
          newsFresh: false,
          providerIndicators: ["RSI", "SMA", "EMA"],
        },
      },
    ],
  },
  {
    id: "opening-scan",
    label: "09:35 ET · opening scan",
    summary:
      "The continuous loop starts, reads local daily features only, scores AAPL as an opening swing trade, and leaves intraday adaptation disabled.",
    generatedAt: "2026-04-09T13:35:00.000Z",
    marketSession: "market-hours",
    loopStatus: "running",
    requestBudget: { used: 6, remaining: 19, limit: 25 },
    systemPortfolio: {
      equity: 100182,
      cash: 94420,
      dailyPnl: 182,
      realizedPnl: 0,
      unrealizedPnl: 182,
      openPositions: 1,
      closedTrades: 0,
    },
    ingestEvents: [
      { time: "09:32", ticker: "AAPL", endpoint: "TIME_SERIES_DAILY", result: "served from local store", rows: 0 },
      { time: "09:33", ticker: "AAPL", endpoint: "NEWS_SENTIMENT", result: "budget guarded", rows: 0 },
      { time: "09:34", ticker: "AAPL", endpoint: "LOCAL_FEATURES", result: "assembled from local data", rows: 252 },
    ],
    tickers: [
      {
        ticker: "AAPL",
        latestClose: 202.4,
        changePct: 0.004,
        recommendation: {
          action: "BUY",
          positionAction: "OPEN",
          tradeStyle: "swing_trade",
          timeframeMode: "daily_only",
          entryPrice: 202.4,
          stopPrice: 198.8,
          target1: 209.6,
          positionSize: 28,
          forecastConfidence: 0.74,
          ftaScore: 0.82,
          probabilityOfSuccess: 0.71,
          rationale:
            "Daily-only TimesFM forecast points up, FTA accepted the structure, and the meta-model clears the moderate risk threshold. Position opened in paper mode with deterministic execution logic.",
        },
        priceSeries: [...baseSeries.AAPL, { label: "09:35", close: 202.4 }],
        decisionLog: [
          { time: "09:20", action: "HOLD", positionAction: "HOLD_POSITION", note: "Waiting for market-hours loop." },
          { time: "09:35", action: "BUY", positionAction: "OPEN", note: "FTA + meta-model accepted. Paper swing trade opened." },
        ],
        inventory: {
          dailyBars: 252,
          intradayBars: 0,
          hasRealIntraday: false,
          fundamentalsFresh: true,
          earningsFresh: true,
          newsFresh: false,
          providerIndicators: ["RSI", "SMA", "EMA"],
        },
      },
      {
        ticker: "NVDA",
        latestClose: 896.1,
        changePct: 0.0016,
        recommendation: {
          action: "HOLD",
          positionAction: "HOLD_POSITION",
          tradeStyle: "swing_trade",
          timeframeMode: "daily_only",
          forecastConfidence: 0.66,
          ftaScore: 0.58,
          probabilityOfSuccess: 0.55,
          rationale:
            "Forecast is constructive, but reward-to-risk is still below threshold. The system explicitly avoids weakening FTA just to force more trades.",
          rejectionReason: "reward_risk_below_threshold",
        },
        priceSeries: [...baseSeries.NVDA, { label: "09:35", close: 896.1 }],
        decisionLog: [
          { time: "09:20", action: "HOLD", positionAction: "HOLD_POSITION", note: "Waiting for market-hours loop." },
          { time: "09:35", action: "HOLD", positionAction: "HOLD_POSITION", note: "FTA reward-to-risk still below threshold." },
        ],
        inventory: {
          dailyBars: 252,
          intradayBars: 0,
          hasRealIntraday: false,
          fundamentalsFresh: true,
          earningsFresh: true,
          newsFresh: false,
          providerIndicators: ["RSI", "SMA", "EMA"],
        },
      },
      {
        ticker: "MSFT",
        latestClose: 427.0,
        changePct: -0.0005,
        recommendation: {
          action: "HOLD",
          positionAction: "HOLD_POSITION",
          tradeStyle: "swing_trade",
          timeframeMode: "daily_only",
          forecastConfidence: 0.52,
          ftaScore: 0.44,
          probabilityOfSuccess: 0.49,
          rationale:
            "The forecast confidence is too soft and the meta-model does not support a new trade. The recommendation stays deterministic and conservative.",
          rejectionReason: "meta_model_probability_below_threshold",
        },
        priceSeries: [...baseSeries.MSFT, { label: "09:35", close: 427.0 }],
        decisionLog: [
          { time: "09:20", action: "HOLD", positionAction: "HOLD_POSITION", note: "Waiting for market-hours loop." },
          { time: "09:35", action: "HOLD", positionAction: "HOLD_POSITION", note: "Meta-model gate rejected the setup." },
        ],
        inventory: {
          dailyBars: 252,
          intradayBars: 0,
          hasRealIntraday: false,
          fundamentalsFresh: true,
          earningsFresh: true,
          newsFresh: false,
          providerIndicators: ["RSI", "SMA", "EMA"],
        },
      },
    ],
  },
  {
    id: "mid-session",
    label: "12:10 ET · mid-session refresh",
    summary:
      "The loop refreshes recommendations during market hours, but still uses daily-derived signals. NVDA turns actionable while AAPL remains an open paper position.",
    generatedAt: "2026-04-09T16:10:00.000Z",
    marketSession: "market-hours",
    loopStatus: "running",
    requestBudget: { used: 7, remaining: 18, limit: 25 },
    systemPortfolio: {
      equity: 100486,
      cash: 90110,
      dailyPnl: 486,
      realizedPnl: 0,
      unrealizedPnl: 486,
      openPositions: 2,
      closedTrades: 0,
    },
    ingestEvents: [
      { time: "12:05", ticker: "NVDA", endpoint: "TIME_SERIES_DAILY", result: "served from local store", rows: 0 },
      { time: "12:06", ticker: "NVDA", endpoint: "LOCAL_FEATURES", result: "assembled from local data", rows: 252 },
      { time: "12:07", ticker: "NVDA", endpoint: "NEWS_SENTIMENT", result: "skipped to preserve budget", rows: 0 },
    ],
    tickers: [
      {
        ticker: "AAPL",
        latestClose: 203.8,
        changePct: 0.0069,
        recommendation: {
          action: "HOLD",
          positionAction: "HOLD_POSITION",
          tradeStyle: "swing_trade",
          timeframeMode: "daily_only",
          entryPrice: 202.4,
          stopPrice: 198.8,
          target1: 209.6,
          positionSize: 28,
          forecastConfidence: 0.76,
          ftaScore: 0.84,
          probabilityOfSuccess: 0.73,
          rationale:
            "The open paper trade remains valid. The loop marks the position to market and persists restart-safe state after each iteration.",
        },
        priceSeries: [...baseSeries.AAPL, { label: "09:35", close: 202.4 }, { label: "12:10", close: 203.8 }],
        decisionLog: [
          { time: "09:20", action: "HOLD", positionAction: "HOLD_POSITION", note: "Waiting for market-hours loop." },
          { time: "09:35", action: "BUY", positionAction: "OPEN", note: "FTA + meta-model accepted. Paper swing trade opened." },
          { time: "12:10", action: "HOLD", positionAction: "HOLD_POSITION", note: "Open long remains inside risk and structure bounds." },
        ],
        inventory: {
          dailyBars: 252,
          intradayBars: 0,
          hasRealIntraday: false,
          fundamentalsFresh: true,
          earningsFresh: true,
          newsFresh: false,
          providerIndicators: ["RSI", "SMA", "EMA"],
        },
      },
      {
        ticker: "NVDA",
        latestClose: 901.6,
        changePct: 0.0061,
        recommendation: {
          action: "BUY",
          positionAction: "OPEN",
          tradeStyle: "swing_trade",
          timeframeMode: "daily_only",
          entryPrice: 901.6,
          stopPrice: 882.2,
          target1: 940.4,
          positionSize: 8,
          forecastConfidence: 0.78,
          ftaScore: 0.86,
          probabilityOfSuccess: 0.74,
          rationale:
            "The refreshed daily snapshot confirms a cleaner swing continuation. The system opens a second paper position without relaxing the outer safety envelope.",
        },
        priceSeries: [...baseSeries.NVDA, { label: "09:35", close: 896.1 }, { label: "12:10", close: 901.6 }],
        decisionLog: [
          { time: "09:20", action: "HOLD", positionAction: "HOLD_POSITION", note: "Waiting for market-hours loop." },
          { time: "09:35", action: "HOLD", positionAction: "HOLD_POSITION", note: "FTA reward-to-risk still below threshold." },
          { time: "12:10", action: "BUY", positionAction: "OPEN", note: "Reward-to-risk cleared. Paper swing trade opened." },
        ],
        inventory: {
          dailyBars: 252,
          intradayBars: 0,
          hasRealIntraday: false,
          fundamentalsFresh: true,
          earningsFresh: true,
          newsFresh: false,
          providerIndicators: ["RSI", "SMA", "EMA"],
        },
      },
      {
        ticker: "MSFT",
        latestClose: 428.4,
        changePct: 0.0033,
        recommendation: {
          action: "HOLD",
          positionAction: "HOLD_POSITION",
          tradeStyle: "swing_trade",
          timeframeMode: "daily_only",
          forecastConfidence: 0.58,
          ftaScore: 0.5,
          probabilityOfSuccess: 0.51,
          rationale:
            "MSFT still lacks enough edge after the gates. The system preserves capital rather than spreading into marginal setups.",
          rejectionReason: "fta_score_below_threshold",
        },
        priceSeries: [...baseSeries.MSFT, { label: "09:35", close: 427.0 }, { label: "12:10", close: 428.4 }],
        decisionLog: [
          { time: "09:20", action: "HOLD", positionAction: "HOLD_POSITION", note: "Waiting for market-hours loop." },
          { time: "09:35", action: "HOLD", positionAction: "HOLD_POSITION", note: "Meta-model gate rejected the setup." },
          { time: "12:10", action: "HOLD", positionAction: "HOLD_POSITION", note: "FTA score remains below the moderate profile threshold." },
        ],
        inventory: {
          dailyBars: 252,
          intradayBars: 0,
          hasRealIntraday: false,
          fundamentalsFresh: true,
          earningsFresh: true,
          newsFresh: false,
          providerIndicators: ["RSI", "SMA", "EMA"],
        },
      },
    ],
  },
  {
    id: "eod-closeout",
    label: "16:12 ET · end-of-day closeout",
    summary:
      "After the market closes, the EOD cycle finalizes paper outcomes, writes its summary, updates adaptive context, and keeps the system ready for the next session.",
    generatedAt: "2026-04-09T20:12:00.000Z",
    marketSession: "after-hours",
    loopStatus: "idle",
    requestBudget: { used: 7, remaining: 18, limit: 25 },
    systemPortfolio: {
      equity: 100742,
      cash: 100742,
      dailyPnl: 742,
      realizedPnl: 742,
      unrealizedPnl: 0,
      openPositions: 0,
      closedTrades: 2,
    },
    ingestEvents: [
      { time: "16:05", ticker: "AAPL", endpoint: "EOD_SUMMARY", result: "saved to local logs", rows: 1 },
      { time: "16:07", ticker: "NVDA", endpoint: "ADAPTIVE_CONTEXT", result: "updated", rows: 1 },
      { time: "16:10", ticker: "SYSTEM", endpoint: "REQUEST_BUDGET", result: "no additional calls made", rows: 0 },
    ],
    tickers: [
      {
        ticker: "AAPL",
        latestClose: 205.6,
        changePct: 0.0088,
        recommendation: {
          action: "SELL",
          positionAction: "CLOSE",
          tradeStyle: "swing_trade",
          timeframeMode: "daily_only",
          entryPrice: 202.4,
          stopPrice: 198.8,
          target1: 209.6,
          positionSize: 28,
          forecastConfidence: 0.71,
          ftaScore: 0.79,
          probabilityOfSuccess: 0.69,
          rationale:
            "The replay closes the position into the end-of-day process and records the outcome for the adaptive loop.",
        },
        priceSeries: [...baseSeries.AAPL, { label: "09:35", close: 202.4 }, { label: "12:10", close: 203.8 }, { label: "Close", close: 205.6 }],
        decisionLog: [
          { time: "09:35", action: "BUY", positionAction: "OPEN", note: "FTA + meta-model accepted. Paper swing trade opened." },
          { time: "12:10", action: "HOLD", positionAction: "HOLD_POSITION", note: "Open long remains inside risk and structure bounds." },
          { time: "16:12", action: "SELL", positionAction: "CLOSE", note: "Session closed and outcome finalized for EOD learning." },
        ],
        inventory: {
          dailyBars: 252,
          intradayBars: 0,
          hasRealIntraday: false,
          fundamentalsFresh: true,
          earningsFresh: true,
          newsFresh: false,
          providerIndicators: ["RSI", "SMA", "EMA"],
        },
      },
      {
        ticker: "NVDA",
        latestClose: 909.8,
        changePct: 0.0091,
        recommendation: {
          action: "SELL",
          positionAction: "CLOSE",
          tradeStyle: "swing_trade",
          timeframeMode: "daily_only",
          entryPrice: 901.6,
          stopPrice: 882.2,
          target1: 940.4,
          positionSize: 8,
          forecastConfidence: 0.73,
          ftaScore: 0.81,
          probabilityOfSuccess: 0.71,
          rationale:
            "The second paper position is finalized into the EOD summary, and the adaptive context is updated without enabling intraday fine-tuning.",
        },
        priceSeries: [...baseSeries.NVDA, { label: "09:35", close: 896.1 }, { label: "12:10", close: 901.6 }, { label: "Close", close: 909.8 }],
        decisionLog: [
          { time: "09:35", action: "HOLD", positionAction: "HOLD_POSITION", note: "FTA reward-to-risk still below threshold." },
          { time: "12:10", action: "BUY", positionAction: "OPEN", note: "Reward-to-risk cleared. Paper swing trade opened." },
          { time: "16:12", action: "SELL", positionAction: "CLOSE", note: "Session closed and outcome finalized for EOD learning." },
        ],
        inventory: {
          dailyBars: 252,
          intradayBars: 0,
          hasRealIntraday: false,
          fundamentalsFresh: true,
          earningsFresh: true,
          newsFresh: false,
          providerIndicators: ["RSI", "SMA", "EMA"],
        },
      },
      {
        ticker: "MSFT",
        latestClose: 429.1,
        changePct: 0.0016,
        recommendation: {
          action: "HOLD",
          positionAction: "HOLD_POSITION",
          tradeStyle: "swing_trade",
          timeframeMode: "daily_only",
          forecastConfidence: 0.57,
          ftaScore: 0.48,
          probabilityOfSuccess: 0.5,
          rationale:
            "No trade was opened in MSFT. The deterministic filter chain kept the system focused on higher-conviction ideas.",
          rejectionReason: "no_open_trade",
        },
        priceSeries: [...baseSeries.MSFT, { label: "09:35", close: 427.0 }, { label: "12:10", close: 428.4 }, { label: "Close", close: 429.1 }],
        decisionLog: [
          { time: "09:35", action: "HOLD", positionAction: "HOLD_POSITION", note: "Meta-model gate rejected the setup." },
          { time: "12:10", action: "HOLD", positionAction: "HOLD_POSITION", note: "FTA score remains below the moderate profile threshold." },
          { time: "16:12", action: "HOLD", positionAction: "HOLD_POSITION", note: "No new trade entered; capital preserved." },
        ],
        inventory: {
          dailyBars: 252,
          intradayBars: 0,
          hasRealIntraday: false,
          fundamentalsFresh: true,
          earningsFresh: true,
          newsFresh: false,
          providerIndicators: ["RSI", "SMA", "EMA"],
        },
      },
    ],
    eodSummary: {
      tradesFinalized: 2,
      adaptiveUpdate: "thresholds tightened slightly after a strong day",
      metaModelRetrained: false,
      summary:
        "EOD orchestration finalized trade outcomes, wrote the summary, updated adaptive context, and saved the next-session state without spending additional request budget.",
    },
  },
];
