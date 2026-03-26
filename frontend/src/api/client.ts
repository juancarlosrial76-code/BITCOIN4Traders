const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export interface ApiError extends Error {
  status?: number;
  code?: string;
  endpoint?: string;
}

export interface RetryConfig {
  maxRetries: number;
  retryDelay: number;
  retryCondition?: (error: ApiError) => boolean;
}

const defaultRetryConfig: RetryConfig = {
  maxRetries: 3,
  retryDelay: 1000,
  retryCondition: error => {
    return error.status === undefined || error.status >= 500;
  },
};

type RequestInterceptor = (config: RequestInit) => RequestInit | Promise<RequestInit>;
type ResponseInterceptor = (response: Response) => Response | Promise<Response>;
type ErrorInterceptor = (error: ApiError) => ApiError | Promise<ApiError>;

const requestInterceptors: RequestInterceptor[] = [];
const responseInterceptors: ResponseInterceptor[] = [];
const errorInterceptors: ErrorInterceptor[] = [];

export const apiInterceptors: {
  addRequestInterceptor: (interceptor: RequestInterceptor) => () => void;
  addResponseInterceptor: (interceptor: ResponseInterceptor) => () => void;
  addErrorInterceptor: (interceptor: ErrorInterceptor) => () => void;
} = {
  addRequestInterceptor: (interceptor: RequestInterceptor) => {
    requestInterceptors.push(interceptor);
    return () => {
      const index = requestInterceptors.indexOf(interceptor);
      if (index > -1) requestInterceptors.splice(index, 1);
    };
  },
  addResponseInterceptor: (interceptor: ResponseInterceptor) => {
    responseInterceptors.push(interceptor);
    return () => {
      const index = responseInterceptors.indexOf(interceptor);
      if (index > -1) responseInterceptors.splice(index, 1);
    };
  },
  addErrorInterceptor: (interceptor: ErrorInterceptor) => {
    errorInterceptors.push(interceptor);
    return () => {
      const index = errorInterceptors.indexOf(interceptor);
      if (index > -1) errorInterceptors.splice(index, 1);
    };
  },
};

function sleep(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms));
}

function calculateRetryDelay(attempt: number, baseDelay: number): number {
  return baseDelay * Math.pow(2, attempt);
}

export async function fetchWithRetry<T>(
  endpoint: string,
  options: RequestInit,
  retryConfig: RetryConfig = defaultRetryConfig
): Promise<T> {
  let lastError: ApiError;

  for (let attempt = 0; attempt <= retryConfig.maxRetries; attempt++) {
    try {
      let config = { ...options };

      for (const interceptor of requestInterceptors) {
        config = await interceptor(config);
      }

      const token = localStorage.getItem('access_token');
      const headers: Record<string, string> = {
        'Content-Type': 'application/json',
        ...(token ? { Authorization: `Bearer ${token}` } : {}),
        ...(config.headers as Record<string, string>),
      };

      const response = await fetch(`${API_BASE}${endpoint}`, {
        ...config,
        headers,
      });

      for (const interceptor of responseInterceptors) {
        await interceptor(response);
      }

      if (response.status === 401) {
        localStorage.removeItem('access_token');
        localStorage.removeItem('username');
        window.location.href = '/login';
        const error: ApiError = new Error('Session expired. Please log in again.');
        error.status = 401;
        throw error;
      }

      if (!response.ok) {
        const error: ApiError = new Error(`API Error: ${response.status}`);
        error.status = response.status;
        error.endpoint = endpoint;

        try {
          const data = await response.json();
          error.message = data.message || error.message;
          error.code = data.code;
        } catch {
          // Ignore JSON parse errors
        }

        throw error;
      }

      if (response.status === 204) {
        return undefined as T;
      }

      return response.json();
    } catch (error) {
      lastError = error as ApiError;

      const shouldRetry = attempt < retryConfig.maxRetries && retryConfig.retryCondition(lastError);

      if (shouldRetry) {
        const delay = calculateRetryDelay(attempt, retryConfig.retryDelay);
        await sleep(delay);
        continue;
      }

      break;
    }
  }

  for (const interceptor of errorInterceptors) {
    lastError = await interceptor(lastError);
  }

  throw lastError;
}

export const api = {
  status: () =>
    fetchWithRetry<{ status: string; timestamp: string; version: string }>('/api/status', {}),

  trading: {
    getStatus: () => fetchWithRetry<TradingStatus>('/api/trading/status', {}),
    start: () =>
      fetchWithRetry<{ status: string; timestamp: string }>('/api/trading/start', {
        method: 'POST',
      }),
    stop: () =>
      fetchWithRetry<{ status: string; timestamp: string }>('/api/trading/stop', {
        method: 'POST',
      }),
    getSignal: () =>
      fetchWithRetry<{ signal: string; confidence: number; timestamp: string }>(
        '/api/trading/signal',
        {}
      ),
    getOrders: () => fetchWithRetry<Order[]>('/api/trading/orders', {}),
    placeOrder: (order: PlaceOrderRequest) =>
      fetchWithRetry<Order>('/api/trading/order', {
        method: 'POST',
        body: JSON.stringify(order),
      }),
    cancelOrder: (orderId: string) =>
      fetchWithRetry<{ status: string }>(`/api/trading/orders/${orderId}`, { method: 'DELETE' }),
    getConfig: () => fetchWithRetry<TradingConfig>('/api/trading/config', {}),
    updateConfig: (config: Partial<TradingConfig>) =>
      fetchWithRetry<{ status: string }>('/api/trading/config', {
        method: 'PUT',
        body: JSON.stringify(config),
      }),
    getBalance: () => fetchWithRetry<Balance>('/api/trading/balance', {}),
  },

  config: {
    get: () => fetchWithRetry<FullConfig>('/api/config/', {}),
    getBot: () => fetchWithRetry<BotConfig>('/api/config/bot', {}),
    updateBot: (config: Partial<BotConfig>) =>
      fetchWithRetry<{ status: string }>('/api/config/bot', {
        method: 'PUT',
        body: JSON.stringify(config),
      }),
    getRisk: () => fetchWithRetry<RiskConfig>('/api/config/risk', {}),
    updateRisk: (config: Partial<RiskConfig>) =>
      fetchWithRetry<{ status: string }>('/api/config/risk', {
        method: 'PUT',
        body: JSON.stringify(config),
      }),
    getData: () => fetchWithRetry<DataConfig>('/api/config/data', {}),
    updateData: (config: Partial<DataConfig>) =>
      fetchWithRetry<{ status: string }>('/api/config/data', {
        method: 'PUT',
        body: JSON.stringify(config),
      }),
  },

  analytics: {
    getMetrics: () => fetchWithRetry<PerformanceMetrics>('/api/analytics/metrics', {}),
    getEquityCurve: () => fetchWithRetry<EquityPoint[]>('/api/analytics/equity-curve', {}),
    getMonthlyReturns: () => fetchWithRetry<MonthlyReturn[]>('/api/analytics/monthly-returns', {}),
    getTradeDistribution: () =>
      fetchWithRetry<TradeDistribution[]>('/api/analytics/trade-distribution', {}),
  },

  models: {
    list: () => fetchWithRetry<Model[]>('/api/models/', {}),
    get: (id: number) => fetchWithRetry<Model>(`/api/models/${id}`, {}),
    train: (config?: TrainingConfig) =>
      fetchWithRetry<{ jobId: string }>('/api/models/train', {
        method: 'POST',
        body: config ? JSON.stringify(config) : '{}',
      }),
    delete: (id: number) =>
      fetchWithRetry<{ status: string }>(`/api/models/${id}`, { method: 'DELETE' }),
    getTrainingHistory: () => fetchWithRetry<TrainingJob[]>('/api/models/training/history', {}),
    getTrainingStatus: (jobId: string) =>
      fetchWithRetry<TrainingStatus>(`/api/models/train/${jobId}/status`, {}),
  },

  system: {
    getMetrics: () => fetchWithRetry<SystemMetrics>('/api/system/metrics', {}),
    getLogs: () => fetchWithRetry<LogEntry[]>('/api/system/logs', {}),
    getEndpoints: () => fetchWithRetry<EndpointInfo[]>('/api/system/endpoints', {}),
    getEnv: () => fetchWithRetry<EnvVariable[]>('/api/system/env', {}),
  },
};

interface TradingStatus {
  is_running: boolean;
  current_position: number;
  unrealized_pnl: number;
  timestamp: string;
  champion_signal?: string;
  champion_name?: string;
}

interface Order {
  id: string;
  symbol: string;
  side: 'buy' | 'sell';
  type: 'market' | 'limit';
  quantity: number;
  price?: number;
  status: 'open' | 'filled' | 'cancelled';
  filled_quantity?: number;
  created_at: string;
  updated_at: string;
}

interface PlaceOrderRequest {
  symbol: string;
  side: 'buy' | 'sell';
  order_type: 'market' | 'limit';
  quantity: number;
  price?: number;
  stop_price?: number;
}

interface TradingConfig {
  symbol: string;
  timeframe: string;
  max_positions: number;
  agent_type: string;
  model_path: string;
  leverage: number;
}

interface Balance {
  USDT: number;
  BTC: number;
  totalUSD: number;
  mode: string;
}

interface FullConfig {
  bot: BotConfig;
  risk: RiskConfig;
  data: DataConfig;
}

interface BotConfig {
  symbol: string;
  timeframe: string;
  maxPositions: number;
  agentType: string;
  modelPath: string;
}

interface RiskConfig {
  maxDrawdown: number;
  stopLoss: number;
  takeProfit: number;
  positionSizePercent: number;
  riskPerTrade: number;
}

interface DataConfig {
  dataSource: string;
  startDate: string;
  endDate: string;
  trainTestSplit: number;
}

interface PerformanceMetrics {
  total_trades: number;
  win_rate: number;
  profit_factor: number;
  sharpe_ratio: number;
  max_drawdown: number;
  avg_trade_duration: number;
  total_pnl: number;
}

interface EquityPoint {
  timestamp: string;
  value: number;
}

interface MonthlyReturn {
  month: string;
  return: number;
}

interface TradeDistribution {
  range: string;
  count: number;
}

interface Model {
  id: number;
  name: string;
  type: string;
  created: string;
  size: string;
  status: 'active' | 'trained' | 'not_trained';
  sharpe?: number;
  source: string;
}

interface TrainingConfig {
  modelType: string;
  episodes: number;
  learningRate: number;
  batchSize: number;
  trainSteps: number;
}

interface TrainingJob {
  id: string;
  modelName: string;
  status: 'pending' | 'running' | 'completed' | 'failed';
  progress: number;
  startTime: string;
  endTime?: string;
  metrics?: {
    loss: number;
    reward: number;
    sharpe?: number;
  };
}

interface TrainingStatus {
  status: 'pending' | 'running' | 'completed' | 'failed';
  progress: number;
  metrics?: {
    loss: number;
    reward: number;
  };
}

interface SystemMetrics {
  cpu: number;
  memory: number;
  disk: number;
  uptime: number;
  activeConnections: number;
  requestsPerSecond: number;
  timestamp: string;
}

interface LogEntry {
  id: string;
  timestamp: string;
  level: 'DEBUG' | 'INFO' | 'WARNING' | 'ERROR';
  message: string;
  source?: string;
}

interface EndpointInfo {
  path: string;
  method: string;
  description: string;
}

interface EnvVariable {
  key: string;
  value: string;
}
