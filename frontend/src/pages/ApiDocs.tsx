import { Link } from 'react-router-dom';
import { useTranslation } from 'react-i18next';
import { Card } from '../components/ui';
import { ArrowLeft } from 'lucide-react';

interface Endpoint {
  method: 'GET' | 'POST' | 'PUT' | 'DELETE';
  path: string;
  description: string;
  auth?: boolean;
}

const methodColor: Record<string, string> = {
  GET: 'bg-blue-500/10 text-blue-400',
  POST: 'bg-green-500/10 text-green-400',
  PUT: 'bg-yellow-500/10 text-yellow-400',
  DELETE: 'bg-red-500/10 text-red-400',
};

function EndpointRow({ method, path, description, auth = true }: Endpoint) {
  return (
    <div className="flex items-start gap-3 py-3 border-b border-border/50 last:border-0">
      <span className={`px-2 py-0.5 text-xs font-mono font-bold rounded flex-shrink-0 mt-0.5 ${methodColor[method]}`}>
        {method}
      </span>
      <div className="flex-1 min-w-0">
        <code className="text-sm text-text-primary font-mono">{path}</code>
        <p className="text-xs text-text-secondary mt-0.5">{description}</p>
      </div>
      {auth && (
        <span className="text-xs text-text-muted flex-shrink-0 mt-0.5">🔒 auth</span>
      )}
    </div>
  );
}

function Group({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <Card title={title}>
      <div>{children}</div>
    </Card>
  );
}

export function ApiDocs() {
  const { t } = useTranslation();

  return (
    <div className="space-y-6 max-w-3xl">
      <div className="flex items-center gap-3">
        <Link to="/docs" className="text-text-muted hover:text-bitcoin-orange transition-colors">
          <ArrowLeft size={20} />
        </Link>
        <div>
          <h1 className="text-3xl font-bold text-text-primary">
            {t('docs.api.title', 'API Reference')}
          </h1>
          <p className="text-text-secondary">
            {t('docs.api.subtitle', 'All REST endpoints, request/response formats and authentication.')}
          </p>
        </div>
      </div>

      <Card>
        <p className="text-sm text-text-secondary">
          Base URL: <code className="text-bitcoin-orange font-mono">http://localhost:8000</code>
          {' '}(configure via <code className="text-text-primary font-mono">VITE_API_URL</code>)
        </p>
        <p className="text-sm text-text-secondary mt-2">
          All protected endpoints require a Bearer token in the <code className="text-text-primary">Authorization</code> header.
          Obtain a token via <code className="text-text-primary font-mono">POST /api/auth/login</code>.
        </p>
      </Card>

      <Group title="Authentication">
        <EndpointRow method="POST" path="/api/auth/login" description="Login — returns JWT access token" auth={false} />
        <EndpointRow method="POST" path="/api/auth/logout" description="Invalidate current session" />
        <EndpointRow method="GET" path="/api/auth/me" description="Get current authenticated user info" />
      </Group>

      <Group title="Trading">
        <EndpointRow method="GET" path="/api/trading/status" description="Bot running state, current position, unrealized P&L" />
        <EndpointRow method="POST" path="/api/trading/start" description="Start the trading bot" />
        <EndpointRow method="POST" path="/api/trading/stop" description="Stop the trading bot" />
        <EndpointRow method="GET" path="/api/trading/signal" description="Current agent signal (BUY / SELL / HOLD) with confidence" />
        <EndpointRow method="GET" path="/api/trading/orders" description="List all open and recent orders" />
        <EndpointRow method="POST" path="/api/trading/order" description="Place a manual order {symbol, side, order_type, quantity, price?}" />
        <EndpointRow method="DELETE" path="/api/trading/orders/{id}" description="Cancel an open order by ID" />
        <EndpointRow method="GET" path="/api/trading/balance" description="Current USDT and BTC balances" />
        <EndpointRow method="GET" path="/api/trading/config" description="Active trading configuration (leverage, symbol, etc.)" />
        <EndpointRow method="PUT" path="/api/trading/config" description="Update trading configuration" />
      </Group>

      <Group title="Analytics">
        <EndpointRow method="GET" path="/api/analytics/metrics" description="Performance metrics: win_rate, sharpe_ratio, total_pnl, max_drawdown, etc." />
        <EndpointRow method="GET" path="/api/analytics/equity-curve" description="Historical equity curve [{timestamp, value}]" />
        <EndpointRow method="GET" path="/api/analytics/monthly-returns" description="Monthly return breakdown [{month, return}]" />
        <EndpointRow method="GET" path="/api/analytics/trade-distribution" description="Trade P&L distribution histogram" />
      </Group>

      <Group title="Models">
        <EndpointRow method="GET" path="/api/models/" description="List all available trained models" />
        <EndpointRow method="GET" path="/api/models/{id}" description="Get details of a specific model" />
        <EndpointRow method="POST" path="/api/models/train" description="Trigger a new training run" />
        <EndpointRow method="DELETE" path="/api/models/{id}" description="Delete a model by ID" />
        <EndpointRow method="GET" path="/api/models/training/history" description="List all past training jobs with metrics" />
        <EndpointRow method="GET" path="/api/models/train/{jobId}/status" description="Training job status and progress" />
      </Group>

      <Group title="Configuration">
        <EndpointRow method="GET" path="/api/config/" description="Full bot + risk + data configuration" />
        <EndpointRow method="GET" path="/api/config/bot" description="Bot-specific settings (symbol, timeframe, agent type)" />
        <EndpointRow method="PUT" path="/api/config/bot" description="Update bot settings" />
        <EndpointRow method="GET" path="/api/config/risk" description="Risk parameters (maxDrawdown, stopLoss, takeProfit)" />
        <EndpointRow method="PUT" path="/api/config/risk" description="Update risk parameters" />
        <EndpointRow method="GET" path="/api/config/data" description="Data source configuration" />
        <EndpointRow method="PUT" path="/api/config/data" description="Update data source configuration" />
      </Group>

      <Group title="System">
        <EndpointRow method="GET" path="/api/system/metrics" description="CPU, memory, disk usage, uptime, requests/sec" />
        <EndpointRow method="GET" path="/api/system/logs" description="Recent system log entries" />
        <EndpointRow method="GET" path="/api/system/endpoints" description="List all registered API routes" />
        <EndpointRow method="GET" path="/api/system/env" description="Active environment variables (values may be masked)" />
      </Group>

      <Group title="Order Book">
        <EndpointRow method="GET" path="/api/orderbook/{symbol}?depth=N" description="Live order book bids and asks for a symbol (default depth=10)" />
      </Group>

      <Group title="Status">
        <EndpointRow method="GET" path="/api/status" description="Health check — returns {status, timestamp, version}" auth={false} />
      </Group>

      <div className="flex gap-4">
        <Link to="/docs/trading-guide" className="text-sm text-text-muted hover:text-bitcoin-orange">
          ← Trading Guide
        </Link>
        <Link to="/docs/glossary" className="text-sm text-bitcoin-orange hover:underline">
          Next: Glossary →
        </Link>
      </div>
    </div>
  );
}
