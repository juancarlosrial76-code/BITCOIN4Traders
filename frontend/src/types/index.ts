// ====================
// Core Types
// ====================

export interface User {
  id: string;
  email: string;
  name: string;
  role: 'admin' | 'trader' | 'analyst' | 'viewer';
  createdAt: string;
  lastLogin: string;
}

export interface Portfolio {
  id: string;
  totalEquity: number;
  cashBalance: number;
  positionsValue: number;
  dailyPnL: number;
  dailyPnLPercent: number;
  totalPnL: number;
  totalPnLPercent: number;
  dailyReturn: number;
}

export interface Position {
  id: string;
  symbol: string;
  side: 'LONG' | 'SHORT';
  quantity: number;
  entryPrice: number;
  currentPrice: number;
  unrealizedPnL: number;
  unrealizedPnLPercent: number;
  leverage: number;
  liquidationPrice?: number;
  openedAt: string;
}

export interface Trade {
  id: string;
  orderId: string;
  symbol: string;
  side: 'BUY' | 'SELL';
  orderType: 'MARKET' | 'LIMIT';
  quantity: number;
  price: number;
  commission: number;
  realizedPnL: number;
  executedAt: string;
  status: 'PENDING' | 'FILLED' | 'CANCELLED' | 'REJECTED';
}

export interface Order {
  id: string;
  symbol: string;
  side: 'BUY' | 'SELL';
  type: 'MARKET' | 'LIMIT';
  quantity: number;
  price?: number;
  status: OrderStatus;
  createdAt: string;
  filledAt?: string;
  filledQuantity?: number;
  filledPrice?: number;
}

export type OrderStatus = 
  | 'PENDING' 
  | 'OPEN' 
  | 'FILLED' 
  | 'PARTIALLY_FILLED' 
  | 'CANCELLED' 
  | 'REJECTED';

export type BotStatus = 
  | 'IDLE' 
  | 'STARTING' 
  | 'RUNNING' 
  | 'STOPPING' 
  | 'STOPPED' 
  | 'ERROR';

export interface BotState {
  status: BotStatus;
  uptime: number;
  currentPosition?: Position;
  lastTrade?: Trade;
  errors: BotError[];
}

export interface BotError {
  code: string;
  message: string;
  timestamp: string;
  resolved: boolean;
}

// ====================
// Performance & Analytics
// ====================

export interface PerformanceMetrics {
  totalReturn: number;
  sharpeRatio: number;
  sortinoRatio: number;
  calmarRatio: number;
  maxDrawdown: number;
  maxDrawdownDuration: number;
  winRate: number;
  profitFactor: number;
  totalTrades: number;
  winningTrades: number;
  losingTrades: number;
  avgWin: number;
  avgLoss: number;
  largestWin: number;
  largestLoss: number;
  avgHoldingPeriod: number;
}

export interface EquityPoint {
  timestamp: string;
  value: number;
}

export interface DrawdownPoint {
  timestamp: string;
  drawdown: number;
}

export interface MonthlyReturn {
  month: string;
  year: number;
  monthNum: number;
  return: number;
}

export interface TradeDistribution {
  range: string;
  count: number;
}

// ====================
// Models
// ====================

export interface Model {
  id: string;
  name: string;
  filePath: string;
  fileSize: number;
  trainedAt: string;
  trainingConfig: {
    iterations: number;
    batchSize: number;
    learningRate: number;
    hiddenDim: number;
  };
  validationMetrics: PerformanceMetrics;
  isActive: boolean;
}

// ====================
// Configuration
// ====================

export interface Config {
  id: string;
  name: string;
  version: string;
  updatedAt: string;
  updatedBy: string;
  environment: {
    initialCapital: number;
    transactionCostBps: number;
    slippageModel: string;
  };
  risk: {
    maxPositionSize: number;
    maxDrawdown: number;
    stopLoss?: number;
    takeProfit?: number;
  };
  agent: {
    modelPath: string;
    inferenceSteps: number;
    temperature: number;
  };
  exchange: {
    apiKey: string;
    testnet: boolean;
  };
  notifications: {
    telegram: boolean;
    telegramChatId?: string;
    email: boolean;
    emailAddress?: string;
  };
}

// ====================
// System
// ====================

export interface SystemMetrics {
  cpu: {
    usage: number;
    cores: number;
  };
  memory: {
    total: number;
    used: number;
    percent: number;
  };
  gpu?: {
    name: string;
    usage: number;
    memoryTotal: number;
    memoryUsed: number;
    temperature: number;
  };
  disk: {
    total: number;
    used: number;
    percent: number;
  };
  network: {
    bytesSent: number;
    bytesReceived: number;
  };
  process: {
    uptime: number;
    threads: number;
  };
}

export interface LogEntry {
  id: string;
  timestamp: string;
  level: 'DEBUG' | 'INFO' | 'WARNING' | 'ERROR';
  logger: string;
  message: string;
  context?: Record<string, unknown>;
}

// ====================
// API Response Types
// ====================

export interface ApiResponse<T> {
  data: T;
  success: boolean;
  error?: {
    code: string;
    message: string;
  };
}

export interface PaginatedResponse<T> extends ApiResponse<T[]> {
  meta: {
    page: number;
    pageSize: number;
    total: number;
    totalPages: number;
  };
}

// ====================
// WebSocket Messages
// ====================

export interface WSMessage {
  type: WSMessageType;
  payload: unknown;
  timestamp: string;
}

export type WSMessageType = 
  | 'price_update'
  | 'position_update'
  | 'trade_executed'
  | 'bot_status'
  | 'alert'
  | 'metrics_update';

export interface PriceUpdate {
  symbol: string;
  price: number;
  change: number;
  changePercent: number;
  volume: number;
  timestamp: string;
}

// ====================
// Form Types
// ====================

export interface OrderFormData {
  symbol: string;
  side: 'BUY' | 'SELL';
  orderType: 'MARKET' | 'LIMIT';
  quantity: number;
  price?: number;
}

export interface LoginFormData {
  email: string;
  password: string;
}

// ====================
// Utility Types
// ====================

export type Nullable<T> = T | null;
export type Optional<T> = T | undefined;
export type DeepPartial<T> = {
  [P in keyof T]?: T[P] extends object ? DeepPartial<T[P]> : T[P];
};

export type ChangeCallback<T> = (prev: T, next: T) => void;

// ====================
// Store Types
// ====================

export interface EquityCurvePoint {
  timestamp: number;
  value: number;
}

export interface TradingConfig {
  maxPositionSize: number;
  stopLoss: number;
  takeProfit: number;
  riskPerTrade: number;
  leverage: number;
}

export interface BotConfig {
  symbol: string;
  timeframe: string;
  maxPositions: number;
  agentType: string;
  modelPath: string;
}

export interface RiskConfig {
  maxDrawdown: number;
  stopLoss: number;
  takeProfit: number;
  positionSizePercent: number;
}

export interface DataConfig {
  dataSource: string;
  startDate: string;
  endDate: string;
  trainTestSplit: number;
}
