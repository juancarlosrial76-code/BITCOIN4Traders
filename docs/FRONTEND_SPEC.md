# BITCOIN4Traders Frontend - Detailed Technical Specification

**Version:** 2.0  
**Status:** Detailed Planning  
**Last Updated:** 2026-02-27

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [User Stories & Use Cases](#2-user-stories--use-cases)
3. [Feature Specifications](#3-feature-specifications)
4. [UI Component Library](#4-ui-component-library)
5. [Data Models & Types](#5-data-models--types)
6. [API Specification](#6-api-specification)
7. [State Management](#7-state-management)
8. [Real-time Communication](#8-real-time-communication)
9. [Security & Authentication](#9-security--authentication)
10. [File Structure](#10-file-structure)
11. [Implementation Phases](#11-implementation-phases)
12. [Testing Strategy](#12-testing-strategy)
13. [Deployment Pipeline](#13-deployment-pipeline)

---

## 1. Executive Summary

### 1.1 Purpose

The BITCOIN4Traders Control Center is a web-based dashboard for monitoring and controlling the trading bot. It provides real-time insights into trading performance, allows manual intervention, and enables configuration management without CLI access.

### 1.2 Target Users

| User Type | Description | Access Level |
|----------|-------------|--------------|
| **Admin** | Full system control | All features |
| **Trader** | Active trading management | Trading, Analytics, Config |
| **Analyst** | Performance analysis only | Read-only Analytics |
| **Viewer** | Monitoring only | Dashboard view only |

### 1.3 Technology Stack

#### Frontend Stack

| Technology | Version | Purpose |
|------------|---------|---------|
| React | 18.x | UI Framework |
| TypeScript | 5.x | Type Safety |
| Vite | 5.x | Build Tool |
| Tailwind CSS | 3.x | Styling |
| shadcn/ui | latest | Component Library |
| Zustand | 4.x | State Management |
| React Query | 5.x | Server State |
| TanStack Table | 8.x | Data Tables |
| TradingView Lightweight Charts | 4.x | Financial Charts |
| Recharts | 2.x | Analytics Charts |
| React Hook Form | 7.x | Form Handling |
| Zod | 3.x | Schema Validation |
| React Router | 6.x | Routing |
| Socket.io Client | 4.x | WebSocket |

#### Backend Stack (New)

| Technology | Version | Purpose |
|------------|---------|---------|
| FastAPI | 0.109.x | REST API |
| Uvicorn | 0.27.x | ASGI Server |
| SQLAlchemy | 2.x | ORM |
| Pydantic | 2.x | Data Validation |
| Python-Jose | 3.x | JWT |
| Passlib | 1.7.x | Password Hashing |
| Socket.io | 4.x | WebSocket |
| Redis | 7.x | Cache/Sessions |

---

## 2. User Stories & Use Cases

### 2.1 User Stories

#### Story 1: Dashboard Overview
```
As a trader,
I want to see my portfolio value and performance at a glance,
So that I can quickly assess the bot's health.

Acceptance Criteria:
- [ ] Show total equity in USD
- [ ] Show today's P&L
- [ ] Show current open positions
- [ ] Show win rate (last 30 days)
- [ ] Show live BTC price
- [ ] Show bot status (Running/Stopped/Error)
```

#### Story 2: Start Trading
```
As an admin,
I want to start the trading bot with one click,
So that I can begin automated trading.

Acceptance Criteria:
- [ ] Show confirmation dialog before starting
- [ ] Verify API keys are configured
- [ ] Show loading state during startup
- [ ] Display success/error notification
- [ ] Update dashboard status to "Running"
```

#### Story 3: Emergency Stop
```
As an admin,
I want to immediately stop all trading and close positions,
So that I can prevent further losses in extreme market conditions.

Acceptance Criteria:
- [ ] One-click emergency stop button (always visible)
- [ ] No confirmation required (speed is critical)
- [ ] Automatically close all open positions
- [ ] Send Telegram alert
- [ ] Log emergency event
- [ ] Update status immediately
```

#### Story 4: Manual Trade
```
As a trader,
I want to place manual orders,
So that I can intervene in the market when needed.

Acceptance Criteria:
- [ ] Select order type (Market, Limit)
- [ ] Enter quantity
- [ ] Enter price (for Limit orders)
- [ ] Select side (Buy/Sell)
- [ ] Preview estimated cost
- [ ] Submit order
- [ ] Show order status
```

#### Story 5: Configuration Management
```
As an admin,
I want to edit bot configuration through the UI,
So that I don't need to edit YAML files manually.

Acceptance Criteria:
- [ ] Load current config from YAML
- [ ] Display form with validation
- [ ] Validate inputs before save
- [ ] Backup previous config
- [ ] Apply changes (restart bot if needed)
- [ ] Show diff of changes
```

#### Story 6: Performance Analytics
```
As an analyst,
I want to view detailed performance metrics,
So that I can evaluate the strategy's effectiveness.

Acceptance Criteria:
- [ ] Equity curve chart
- [ ] Drawdown chart
- [ ] Monthly returns heatmap
- [ ] Trade distribution histogram
- [ ] Key metrics: Sharpe, Sortino, Calmar, Max DD
- [ ] Filter by date range
```

#### Story 7: Model Management
```
As an admin,
I want to switch between different trained models,
So that I can test different strategies.

Acceptance Criteria:
- [ ] List all available models
- [ ] Show model metadata (date, metrics, config)
- [ ] Preview model performance
- [ ] One-click model switching
- [ ] Automatic bot restart with new model
- [ ] Rollback to previous model
```

#### Story 8: Real-time Monitoring
```
As a trader,
I want to see live updates without refreshing,
So that I can react quickly to market changes.

Acceptance Criteria:
- [ ] Live price updates (WebSocket)
- [ ] Position updates in real-time
- [ ] Trade execution notifications
- [ ] Alert notifications
- [ ] Connection status indicator
```

---

## 3. Feature Specifications

### 3.1 Dashboard Module

#### Layout
```
┌─────────────────────────────────────────────────────────────┐
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐       │
│  │ Total Equity│  │ Today's P&L │  │ Open Pos   │       │
│  │   $125,430  │  │   +$1,250   │  │     2      │       │
│  └─────────────┘  └─────────────┘  └─────────────┘       │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────────────────────────────────────────────────┐   │
│  │              LIVE PRICE CHART (BTC/USDT)            │   │
│  │                                                      │   │
│  └──────────────────────────────────────────────────────┘   │
├─────────────────────────────────────────────────────────────┤
│  Recent Trades                    │ Bot Status             │
│  ┌────────────────────────────┐ │ ┌──────────────────┐  │
│  │ BUY  0.5 BTC @ $67,230    │ │ │ 🟢 Running      │  │
│  │ SELL 0.3 BTC @ $67,450    │ │ │ Uptime: 2h 30m  │  │
│  │ BUY  0.2 BTC @ $67,100    │ │ │ [⏹ Stop]       │  │
│  └────────────────────────────┘ │ └──────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

#### Components

| Component | Description | Props |
|-----------|-------------|-------|
| StatCard | Metric display card | value, label, change, changeType |
| PriceChart | TradingView candlestick | data, symbol, interval |
| TradeList | Recent trades table | trades[], onRefresh |
| StatusBadge | Bot status indicator | status, uptime |
| ActionButton | Primary actions | label, onClick, variant |

### 3.2 Trading Module

#### Manual Order Form

| Field | Type | Validation | Default |
|-------|------|------------|---------|
| Symbol | Select | From available pairs | BTC/USDT |
| Side | Radio | BUY or SELL | Required |
| Order Type | Select | MARKET or LIMIT | MARKET |
| Quantity | Number | > 0, decimals | Required |
| Price | Number | > 0 (LIMIT only) | Current price |
| Total | Calculated | Read-only | quantity × price |

#### Order Validation Rules

```typescript
interface OrderValidation {
  quantity: {
    min: 0.0001;  // Min order size
    max: 100;     // Max order size
    decimals: 4;  // Max decimal places
  };
  price: {
    maxSlippage: 0.01; // Max 1% slippage warning
  };
}
```

### 3.3 Configuration Module

#### Configuration Sections

| Section | Fields | Type |
|---------|--------|------|
| **Environment** | initial_capital, transaction_cost_bps | Number |
| **Risk** | max_position_size, max_drawdown, stop_loss | Number |
| **Agent** | model_path, inference_steps | String/Number |
| **Exchange** | api_key (masked), testnet | String/Boolean |
| **Notifications** | telegram_enabled, email_alerts | Boolean |

#### Config Editor Features

- [ ] Schema-based form generation
- [ ] Real-time validation
- [ ] Default value reset
- [ ] Diff view (before/after)
- [ ] Backup history (last 5)
- [ ] Import/Export JSON/YAML

### 3.4 Analytics Module

#### Charts

| Chart | Library | Data Source |
|-------|---------|-------------|
| Equity Curve | Recharts | /api/analytics/equity |
| Drawdown | Recharts | /api/analytics/drawdown |
| Monthly Heatmap | Custom | /api/analytics/monthly |
| Trade Distribution | Recharts | /api/analytics/trades |
| Price Chart | TradingView | WebSocket |

#### Metrics Display

| Metric | Formula | Update Frequency |
|--------|---------|------------------|
| Total Return | (Current - Initial) / Initial | Real-time |
| Sharpe Ratio | (Return - RiskFree) / StdDev | Daily |
| Sortino Ratio | (Return - Target) / DownsideDev | Daily |
| Calmar Ratio | Annual Return / MaxDD | Daily |
| Win Rate | Wins / Total Trades | Real-time |
| Profit Factor | Gross Profit / Gross Loss | Real-time |
| Max Drawdown | (Peak - Trough) / Peak | Daily |

### 3.5 Models Module

#### Model Card

```
┌────────────────────────────────────────┐
│  📊 ppo_best_v3.pt                     │
│  ───────────────────────────────────  │
│  Trained: 2026-02-15                  │
│  Iterations: 500                       │
│  Val Sharpe: 1.23                     │
│  ───────────────────────────────────  │
│  [Activate] [Delete] [View Details]   │
└────────────────────────────────────────┘
```

#### Model Actions

- **Activate**: Switch to model (requires restart)
- **Delete**: Remove model file
- **View Details**: Show training config, metrics
- **Download**: Export model file

### 3.6 System Module

#### System Metrics

| Metric | Source | Update |
|--------|--------|--------|
| CPU Usage | psutil.cpu_percent() | 5s |
| RAM Usage | psutil.virtual_memory() | 5s |
| GPU Usage | nvidia-smi (if available) | 5s |
| Disk Usage | psutil.disk_usage() | 60s |
| Network | psutil.net_io_counters() | 60s |

#### Log Viewer

| Feature | Implementation |
|---------|---------------|
| Source | /api/system/logs |
| Format | JSON logs |
| Levels | DEBUG, INFO, WARNING, ERROR |
| Filtering | By level, time, keyword |
| Search | Full-text search |
| Pagination | Infinite scroll |
| Export | Download as .log |

---

## 4. UI Component Library

### 4.1 Core Components

```typescript
// Button variants
type ButtonVariant = 'default' | 'destructive' | 'outline' | 'secondary' | 'ghost' | 'link';
type ButtonSize = 'sm' | 'default' | 'lg' | 'icon';

// Input types
type InputType = 'text' | 'number' | 'password' | 'email' | 'search';

// Form components
<Button variant="default" size="lg">Start Trading</Button>
<Input type="number" placeholder="0.00" />
<Select>
  <SelectTrigger><SelectValue placeholder="Select" /></SelectTrigger>
  <SelectContent>
    <SelectItem value="btc">BTC/USDT</SelectItem>
  </SelectContent>
</Select>
<Switch checked={enabled} onCheckedChange={setEnabled} />
<Slider defaultValue={[50]} max={100} step={1} />
```

### 4.2 Custom Trading Components

```typescript
// Price Display
<PriceDisplay 
  value={67230.50} 
  currency="USD" 
  format="compact"  // $67K
  change={2.5}      // +2.5%
  size="lg"
/>

// Position Card
<PositionCard 
  position={{
    symbol: 'BTC/USDT',
    side: 'LONG',
    quantity: 0.5,
    entryPrice: 65000,
    currentPrice: 67230,
    pnl: 1115.25,
    pnlPercent: 3.43
  }}
/>

// Order Form
<OrderForm 
  onSubmit={handleOrder}
  validation={orderValidation}
  balance={50000}
/>

// Trade Table
<TradeTable 
  trades={trades}
  columns={['time', 'side', 'symbol', 'quantity', 'price', 'total']}
  sortable
  filterable
/>
```

### 4.3 Chart Components

```typescript
// Candlestick Chart
<CandlestickChart 
  data={candleData}
  symbol="BTCUSDT"
  interval="1h"
  height={400}
/>

// Equity Curve
<EquityChart 
  data={equityData}
  showDrawdown
  showBenchmark
  height={300}
/>

// Performance Metrics
<MetricsGrid 
  metrics={[
    { label: 'Sharpe', value: 1.45, format: 'number' },
    { label: 'Max DD', value: -12.5, format: 'percent' },
    { label: 'Win Rate', value: 58, format: 'percent' }
  ]}
/>
```

---

## 5. Data Models & Types

### 5.1 TypeScript Interfaces

```typescript
// ====================
// Core Types
// ====================

interface User {
  id: string;
  email: string;
  name: string;
  role: 'admin' | 'trader' | 'analyst' | 'viewer';
  createdAt: Date;
  lastLogin: Date;
}

interface Portfolio {
  id: string;
  totalEquity: number;
  cashBalance: number;
  positionsValue: number;
  dailyPnL: number;
  totalPnL: number;
  dailyReturn: number;
}

interface Position {
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
  openedAt: Date;
}

interface Trade {
  id: string;
  orderId: string;
  symbol: string;
  side: 'BUY' | 'SELL';
  orderType: 'MARKET' | 'LIMIT';
  quantity: number;
  price: number;
  commission: number;
  realizedPnL: number;
  executedAt: Date;
  status: 'PENDING' | 'FILLED' | 'CANCELLED' | 'REJECTED';
}

interface Order {
  id: string;
  symbol: string;
  side: 'BUY' | 'SELL';
  type: 'MARKET' | 'LIMIT';
  quantity: number;
  price?: number;
  status: OrderStatus;
  createdAt: Date;
  filledAt?: Date;
  filledQuantity?: number;
  filledPrice?: number;
}

type OrderStatus = 'PENDING' | 'OPEN' | 'FILLED' | 'PARTIALLY_FILLED' | 'CANCELLED' | 'REJECTED';

interface BotStatus {
  state: 'IDLE' | 'STARTING' | 'RUNNING' | 'STOPPING' | 'STOPPED' | 'ERROR';
  uptime: number;  // seconds
  currentPosition?: Position;
  lastTrade?: Trade;
  errors: BotError[];
}

interface BotError {
  code: string;
  message: string;
  timestamp: Date;
  resolved: boolean;
}

interface PerformanceMetrics {
  totalReturn: number;
  sharpeRatio: number;
  sortinoRatio: number;
  calmarRatio: number;
  maxDrawdown: number;
  maxDrawdownDuration: number;  // days
  winRate: number;
  profitFactor: number;
  totalTrades: number;
  winningTrades: number;
  losingTrades: number;
  avgWin: number;
  avgLoss: number;
  largestWin: number;
  largestLoss: number;
  avgHoldingPeriod: number;  // hours
}

interface Model {
  id: string;
  name: string;
  filePath: string;
  fileSize: number;
  trainedAt: Date;
  trainingConfig: {
    iterations: number;
    batchSize: number;
    learningRate: number;
    hiddenDim: number;
  };
  validationMetrics: PerformanceMetrics;
  isActive: boolean;
}

interface Config {
  id: string;
  name: string;
  version: string;
  updatedAt: Date;
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
    apiKey: string;  // masked
    testnet: boolean;
  };
  notifications: {
    telegram: boolean;
    telegramChatId?: string;
    email: boolean;
    emailAddress?: string;
  };
}

interface SystemMetrics {
  cpu: {
    usage: number;  // percent
    cores: number;
  };
  memory: {
    total: number;  // bytes
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
    openFiles: number;
  };
}

interface LogEntry {
  id: string;
  timestamp: Date;
  level: 'DEBUG' | 'INFO' | 'WARNING' | 'ERROR';
  logger: string;
  message: string;
  context?: Record<string, unknown>;
}
```

### 5.2 API Response Types

```typescript
// Generic API Response
interface ApiResponse<T> {
  data: T;
  success: boolean;
  error?: {
    code: string;
    message: string;
  };
  meta?: {
    page?: number;
    pageSize?: number;
    total?: number;
  };
}

// Pagination
interface PaginatedResponse<T> extends ApiResponse<T[]> {
  meta: {
    page: number;
    pageSize: number;
    total: number;
    totalPages: number;
  };
}

// WebSocket Messages
interface WSMessage {
  type: string;
  payload: unknown;
  timestamp: string;
}

type WSMessageType = 
  | 'price_update'
  | 'position_update'
  | 'trade_executed'
  | 'bot_status'
  | 'alert'
  | 'metrics_update';
```

---

## 6. API Specification

### 6.1 Authentication

```
┌─────────────────────────────────────────────────────────────┐
│  Authentication Flow                                        │
└─────────────────────────────────────────────────────────────┘

POST /api/v1/auth/login
  Body: { email, password }
  Response: { access_token, refresh_token, user }

POST /api/v1/auth/logout
  Headers: Authorization: Bearer <token>
  Response: { success }

POST /api/v1/auth/refresh
  Body: { refresh_token }
  Response: { access_token, refresh_token }

GET /api/v1/auth/me
  Headers: Authorization: Bearer <token>
  Response: { user }
```

### 6.2 Trading Endpoints

```
# Bot Control
─────────────────────────────────────────────
GET    /api/v1/trading/status        → BotStatus
POST   /api/v1/trading/start         → { success, message }
POST   /api/v1/trading/stop          → { success, message }
POST   /api/v1/trading/emergency    → { success, message }

# Orders
─────────────────────────────────────────────
GET    /api/v1/orders                 → Order[]
GET    /api/v1/orders/:id            → Order
POST   /api/v1/orders                 → Order
DELETE /api/v1/orders/:id             → { success }

# Portfolio
─────────────────────────────────────────────
GET    /api/v1/portfolio              → Portfolio
GET    /api/v1/portfolio/positions    → Position[]
GET    /api/v1/portfolio/balance      → { available, locked }
```

### 6.3 Configuration Endpoints

```
GET    /api/v1/config                  → Config
PUT    /api/v1/config                  → Config
GET    /api/v1/config/schema          → JSON Schema
GET    /api/v1/config/versions        → Config[]
POST   /api/v1/config/versions/:id/rollback → Config
POST   /api/v1/config/export          → YAML/JSON file
POST   /api/v1/config/import          → Config
```

### 6.4 Analytics Endpoints

```
GET    /api/v1/analytics/performance   → PerformanceMetrics
GET    /api/v1/analytics/equity        → { timestamp, value }[]
GET    /api/v1/analytics/drawdown      → { timestamp, drawdown }[]
GET    /api/v1/analytics/monthly       → { month, return }[]
GET    /api/v1/analytics/trades       → Trade[]
GET    /api/v1/analytics/distribution  → { range, count }[]
```

### 6.5 Models Endpoints

```
GET    /api/v1/models                  → Model[]
GET    /api/v1/models/:id            → Model
POST   /api/v1/models/:id/activate   → { success }
DELETE /api/v1/models/:id            → { success }
POST   /api/v1/models/upload         → Model
GET    /models/:id/download           → File
```

### 6.6 System Endpoints

```
GET    /api/v1/system/health          → SystemMetrics
GET    /api/v1/system/logs           → LogEntry[]
GET    /api/v1/system/logs/stream    → WebSocket
GET    /api/v1/system/metrics         → SystemMetrics
POST   /api/v1/system/backup          → { filename }
POST   /api/v1/system/restore         → { success }
```

### 6.7 WebSocket Events

```
Server → Client:
─────────────────────────────────────────────
{ type: 'price_update', payload: { symbol, price, change } }
{ type: 'position_update', payload: Position }
{ type: 'trade_executed', payload: Trade }
{ type: 'bot_status', payload: BotStatus }
{ type: 'alert', payload: { level, message } }
{ type: 'metrics_update', payload: SystemMetrics }

Client → Server:
─────────────────────────────────────────────
{ type: 'subscribe', channels: ['trading', 'metrics'] }
{ type: 'unsubscribe', channels: ['trading'] }
```

---

## 7. State Management

### 7.1 Zustand Stores

```typescript
// ====================
// authStore.ts
// ====================
interface AuthState {
  user: User | null;
  token: string | null;
  isAuthenticated: boolean;
  isLoading: boolean;
  
  // Actions
  login: (email: string, password: string) => Promise<void>;
  logout: () => void;
  refreshToken: () => Promise<void>;
}

// ====================
// tradingStore.ts
// ====================
interface TradingState {
  status: BotStatus;
  positions: Position[];
  orders: Order[];
  portfolio: Portfolio | null;
  
  // Actions
  startTrading: () => Promise<void>;
  stopTrading: () => Promise<void>;
  placeOrder: (order: OrderRequest) => Promise<Order>;
  cancelOrder: (orderId: string) => Promise<void>;
  refreshStatus: () => Promise<void>;
}

// ====================
// configStore.ts
// ====================
interface ConfigState {
  config: Config | null;
  isLoading: boolean;
  isDirty: boolean;
  
  loadConfig: () => Promise<void>;
  updateConfig: (updates: Partial<Config>) => void;
  saveConfig: () => Promise<void>;
  resetConfig: () => Promise<void>;
}

// ====================
// analyticsStore.ts
// ====================
interface AnalyticsState {
  metrics: PerformanceMetrics | null;
  equityCurve: { timestamp: Date; value: number }[];
  trades: Trade[];
  dateRange: { start: Date; end: Date };
  
  fetchMetrics: () => Promise<void>;
  fetchEquityCurve: (range: DateRange) => Promise<void>;
  fetchTrades: (params: TradeQuery) => Promise<void>;
}

// ====================
// modelsStore.ts
// ====================
interface ModelsState {
  models: Model[];
  activeModel: Model | null;
  isLoading: boolean;
  
  fetchModels: () => Promise<void>;
  activateModel: (modelId: string) => Promise<void>;
  deleteModel: (modelId: string) => Promise<void>;
  uploadModel: (file: File) => Promise<void>;
}

// ====================
// systemStore.ts
// ====================
interface SystemState {
  metrics: SystemMetrics | null;
  logs: LogEntry[];
  connectionStatus: 'connected' | 'disconnected' | 'reconnecting';
  
  connect: () => void;
  disconnect: () => void;
  fetchMetrics: () => Promise<void>;
  fetchLogs: (filters: LogFilters) => Promise<void>;
}
```

### 7.2 React Query Configuration

```typescript
// queryClient.ts
export const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 5000,           // 5 seconds
      cacheTime: 300000,         // 5 minutes
      refetchOnWindowFocus: true,
      retry: 1,
    },
    mutations: {
      onError: (error) => {
        toast.error(error.message);
      },
    },
  },
});
```

---

## 8. Real-time Communication

### 8.1 WebSocket Connection

```typescript
// hooks/useWebSocket.ts
import { useEffect, useCallback } from 'react';
import { useTradingStore } from '@/stores/tradingStore';
import { useSystemStore } from '@/stores/systemStore';

export function useWebSocket() {
  const [socket, setSocket] = useState<WebSocket | null>(null);
  const updateStatus = useTradingStore(s => s.updateStatus);
  const updateMetrics = useSystemStore(s => s.updateMetrics);
  
  useEffect(() => {
    const ws = new WebSocket('ws://localhost:8000/ws');
    
    ws.onopen = () => {
      console.log('WebSocket connected');
      // Subscribe to channels
      ws.send(JSON.stringify({
        type: 'subscribe',
        channels: ['trading', 'metrics']
      }));
    };
    
    ws.onmessage = (event) => {
      const message = JSON.parse(event.data);
      
      switch (message.type) {
        case 'bot_status':
          updateStatus(message.payload);
          break;
        case 'metrics_update':
          updateMetrics(message.payload);
          break;
        case 'trade_executed':
          // Handle trade notification
          break;
        case 'alert':
          // Handle alert notification
          break;
      }
    };
    
    ws.onclose = () => {
      console.log('WebSocket disconnected');
    };
    
    setSocket(ws);
    
    return () => {
      ws.close();
    };
  }, []);
  
  return socket;
}
```

### 8.2 Reconnection Strategy

```typescript
const RECONNECT_INTERVAL = 3000;  // 3 seconds
const MAX_RECONNECT_ATTEMPTS = 10;

function connectWithRetry(url: string) {
  let attempts = 0;
  
  function tryConnect() {
    const ws = new WebSocket(url);
    
    ws.onclose = () => {
      if (attempts < MAX_RECONNECT_ATTEMPTS) {
        attempts++;
        setTimeout(tryConnect, RECONNECT_INTERVAL * attempts);
      }
    };
    
    return ws;
  }
  
  return tryConnect();
}
```

---

## 9. Security & Authentication

### 9.1 JWT Implementation

```python
# backend/auth/jwt.py
from datetime import datetime, timedelta
from jose import JWTError, jwt

SECRET_KEY = "your-secret-key"  # From environment
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE = 30  # minutes
REFRESH_TOKEN_EXPIRE = 7  # days

def create_access_token(data: dict) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE)
    to_encode.update({"exp": expire, "type": "access"})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def create_refresh_token(data: dict) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(days=REFRESH_TOKEN_EXPIRE)
    to_encode.update({"exp": expire, "type": "refresh"})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def verify_token(token: str) -> dict:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")
```

### 9.2 Role-Based Access Control

```typescript
// Frontend route protection
const ProtectedRoute = ({ 
  children, 
  requiredRole 
}: { 
  children: React.ReactNode; 
  requiredRole: UserRole; 
}) => {
  const { user } = useAuth();
  
  if (!user || !hasRole(user, requiredRole)) {
    return <Navigate to="/unauthorized" />;
  }
  
  return <>{children}</>;
};

// Usage
<ProtectedRoute requiredRole="admin">
  <SettingsPage />
</ProtectedRoute>
```

### 9.3 API Key Security

```typescript
// Mask API key in UI
function maskApiKey(key: string): string {
  if (!key || key.length < 8) return '***';
  return `${key.slice(0, 4)}...${key.slice(-4)}`;
}

// Display: "API Key: Binan***8f2d"
```

---

## 10. File Structure

```
BITCOIN4Traders/
├── frontend/                         # New frontend project
│   ├── public/
│   │   ├── favicon.svg
│   │   ├── logo.svg
│   │   └── robots.txt
│   ├── src/
│   │   ├── components/              # Reusable components
│   │   │   ├── ui/                 # Base UI components (shadcn)
│   │   │   │   ├── button.tsx
│   │   │   │   ├── input.tsx
│   │   │   │   ├── select.tsx
│   │   │   │   ├── card.tsx
│   │   │   │   ├── dialog.tsx
│   │   │   │   ├── table.tsx
│   │   │   │   └── ...
│   │   │   ├── charts/             # Chart components
│   │   │   │   ├── candlestick.tsx
│   │   │   │   ├── equity.tsx
│   │   │   │   └── metrics.tsx
│   │   │   ├── trading/            # Trading-specific
│   │   │   │   ├── order-form.tsx
│   │   │   │   ├── position-card.tsx
│   │   │   │   ├── trade-list.tsx
│   │   │   │   └── price-display.tsx
│   │   │   └── layout/             # Layout components
│   │   │       ├── header.tsx
│   │   │       ├── sidebar.tsx
│   │   │       └── page-container.tsx
│   │   ├── pages/                  # Page components
│   │   │   ├── dashboard/
│   │   │   │   └── page.tsx
│   │   │   ├── trading/
│   │   │   │   ├── components/
│   │   │   │   └── page.tsx
│   │   │   ├── config/
│   │   │   │   └── page.tsx
│   │   │   ├── analytics/
│   │   │   │   └── page.tsx
│   │   │   ├── models/
│   │   │   │   └── page.tsx
│   │   │   ├── system/
│   │   │   │   └── page.tsx
│   │   │   └── auth/
│   │   │       ├── login.tsx
│   │   │       └── protected-route.tsx
│   │   ├── hooks/                  # Custom React hooks
│   │   │   ├── use-auth.ts
│   │   │   ├── use-websocket.ts
│   │   │   ├── use-trading.ts
│   │   │   └── use-config.ts
│   │   ├── stores/                 # Zustand stores
│   │   │   ├── auth-store.ts
│   │   │   ├── trading-store.ts
│   │   │   ├── config-store.ts
│   │   │   ├── analytics-store.ts
│   │   │   ├── models-store.ts
│   │   │   └── system-store.ts
│   │   ├── lib/                   # Utilities
│   │   │   ├── api.ts             # Axios client
│   │   │   ├── utils.ts
│   │   │   ├── formatters.ts      # Number/date formatters
│   │   │   └── validators.ts       # Zod schemas
│   │   ├── types/                 # TypeScript types
│   │   │   └── index.ts
│   │   ├── App.tsx
│   │   ├── main.tsx
│   │   └── router.tsx
│   ├── package.json
│   ├── tsconfig.json
│   ├── tailwind.config.js
│   ├── vite.config.ts
│   └── index.html
│
├── backend/                         # New backend API (future)
│   ├── app/
│   │   ├── api/
│   │   │   └── v1/
│   │   │       ├── auth.py
│   │   │       ├── trading.py
│   │   │       ├── config.py
│   │   │       ├── analytics.py
│   │   │       ├── models.py
│   │   │       └── system.py
│   │   ├── core/
│   │   │   ├── config.py
│   │   │   ├── security.py
│   │   │   └── websocket.py
│   │   ├── db/
│   │   │   ├── base.py
│   │   │   └── session.py
│   │   ├── models/
│   │   │   └── schemas.py
│   │   └── main.py
│   ├── requirements.txt
│   └── pyproject.toml
│
└── docs/
    └── FRONTEND_SPEC.md           # This document
```

---

## 11. Implementation Phases

### Phase 1: MVP (2 weeks)

| Task | Duration | Priority |
|------|----------|----------|
| Project setup (Vite + React + TS) | 1 day | Must |
| UI component library setup | 2 days | Must |
| Authentication flow | 2 days | Must |
| Dashboard page | 2 days | Must |
| Trading start/stop | 1 day | Must |
| Basic portfolio view | 2 days | Must |
| WebSocket connection | 2 days | Must |

### Phase 2: Core Features (3 weeks)

| Task | Duration | Priority |
|------|----------|----------|
| Manual order form | 2 days | Must |
| Configuration editor | 3 days | Must |
| Trade history table | 2 days | Must |
| Performance charts | 3 days | Should |
| Model management | 2 days | Should |
| Log viewer | 2 days | Should |
| System metrics | 1 day | Could |

### Phase 3: Advanced (3 weeks)

| Task | Duration | Priority |
|------|----------|----------|
| Advanced analytics | 3 days | Should |
| Multiple model support | 2 days | Should |
| Backup/restore | 2 days | Could |
| Notifications | 2 days | Could |
| Mobile responsive | 3 days | Could |
| Theme customization | 1 day | Could |

---

## 12. Testing Strategy

### 12.1 Unit Tests

```bash
# Run with Vitest
npm run test:unit

# Coverage target: 70%
```

### 12.2 Component Tests

```bash
# Run with React Testing Library
npm run test:components

# Test key components:
# - OrderForm validation
# - PositionCard display
# - Chart rendering
```

### 12.3 E2E Tests

```bash
# Run with Playwright
npm run test:e2e

# Critical flows:
# 1. Login → Dashboard
# 2. Start trading → Status updates
# 3. Place order → Trade executes
# 4. Edit config → Bot restarts
```

---

## 13. Deployment Pipeline

### 13.1 Docker Configuration

```yaml
# docker-compose.yml
version: '3.8'

services:
  frontend:
    build: ./frontend
    ports:
      - "3000:80"
    depends_on:
      - backend
    
  backend:
    build: ./backend
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://user:pass@db:5432/trading
      - REDIS_URL=redis://redis:6379
    depends_on:
      - db
      - redis
    
  db:
    image: postgres:15
    volumes:
      - postgres_data:/var/lib/postgresql/data
    
  redis:
    image: redis:7-alpine
    
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
```

### 13.2 Environment Variables

```bash
# backend/.env
DATABASE_URL=postgresql://user:pass@localhost:5432/trading
REDIS_URL=redis://localhost:6379
SECRET_KEY=your-secret-key-here
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

# Frontend (.env.production)
VITE_API_URL=http://localhost:8000
VITE_WS_URL=ws://localhost:8000
```

---

## 14. Future Enhancements

### 14.1 Phase 4 - Advanced Features

| Feature | Description |
|---------|-------------|
| Multi-account support | Manage multiple trading accounts |
| Strategy builder | No-code strategy creation |
| Mobile app | React Native companion |
| Paper trading simulation | Test strategies without real money |
| Custom indicators | User-defined technical indicators |
| AI Assistant | Chat with the bot |

### 14.2 Phase 5 - Enterprise

| Feature | Description |
|---------|-------------|
| Team collaboration | Multi-user workspaces |
| Audit logging | Complete action history |
| API for external tools | REST API for third-party |
| Webhook integrations | Zapier, IFTTT |
| SLA monitoring | Uptime guarantees |

---

## Appendix A: Color Palette

```css
:root {
  /* Primary - Bitcoin Blue */
  --primary: #0ea5e9;
  --primary-hover: #0284c7;
  --primary-light: #38bdf8;
  
  /* Semantic */
  --success: #22c55e;
  --success-light: #4ade80;
  --danger: #ef4444;
  --danger-light: #f87171;
  --warning: #f59e0b;
  --warning-light: #fbbf24;
  --info: #3b82f6;
  
  /* Background - Dark Mode */
  --background: #0f172a;
  --background-secondary: #1e293b;
  --background-tertiary: #334155;
  
  /* Surface */
  --surface: #1e293b;
  --surface-hover: #334155;
  
  /* Border */
  --border: #334155;
  --border-light: #475569;
  
  /* Text */
  --text-primary: #f8fafc;
  --text-secondary: #94a3b8;
  --text-muted: #64748b;
  
  /* Profit/Loss */
  --profit: #22c55e;
  --loss: #ef4444;
  
  /* Charts */
  --chart-green: #22c55e;
  --chart-red: #ef4444;
  --chart-blue: #3b82f6;
}
```

---

## Appendix B: Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `Ctrl + S` | Start trading |
| `Ctrl + X` | Stop trading |
| `Ctrl + E` | Emergency stop |
| `Ctrl + K` | Command palette |
| `Ctrl + /` | Toggle sidebar |
| `Esc` | Close modal |

---

## Appendix C: Error Codes

| Code | Description | Action |
|------|-------------|--------|
| AUTH_001 | Invalid credentials | Show login form |
| AUTH_002 | Token expired | Redirect to login |
| AUTH_003 | Insufficient permissions | Show unauthorized |
| TRADE_001 | Insufficient balance | Show error message |
| TRADE_002 | Invalid quantity | Validate input |
| TRADE_003 | Market closed | Show warning |
| CONFIG_001 | Invalid config | Show validation errors |
| CONFIG_002 | Config locked | Show lock message |
| SYSTEM_001 | Bot not running | Show start button |
| SYSTEM_002 | Connection lost | Show reconnecting |

---

**Document Version:** 2.0  
**Created:** 2026-02-27  
**Last Updated:** 2026-02-27  
**Status:** Ready for Implementation
