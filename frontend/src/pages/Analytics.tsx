import { useState, useEffect } from 'react';
import { Card } from '../components/ui';
import { api } from '../api/client';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  BarChart,
  Bar,
} from 'recharts';
import { TrendingUp, TrendingDown, Award, Target, Clock, Percent, AlertCircle } from 'lucide-react';

// API response type — all fields are snake_case as returned by the backend
interface ApiMetrics {
  total_trades: number;
  win_rate: number;
  profit_factor: number;
  sharpe_ratio: number;
  max_drawdown: number;
  avg_trade_duration: number;
  total_pnl: number;
}

export function Analytics() {
  const [metrics, setMetrics] = useState<ApiMetrics | null>(null);
  const [equityCurve, setEquityCurve] = useState<{ timestamp: string; value: number }[]>([]);
  const [monthlyReturns, setMonthlyReturns] = useState<{ month: string; return: number }[]>([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchData = async () => {
      setError(null);
      try {
        const [metricsData, equityData, returnsData] = await Promise.all([
          api.analytics.getMetrics(),
          api.analytics.getEquityCurve(),
          api.analytics.getMonthlyReturns(),
        ]);
        setMetrics(metricsData as unknown as ApiMetrics);
        setEquityCurve(equityData);
        setMonthlyReturns(returnsData);
      } catch (e) {
        const msg = (e as Error).message || 'Failed to load analytics';
        console.error('[Analytics] fetch failed:', e);
        setError(msg);
      }
    };
    fetchData();
  }, []);

  // Compute derived values from available API fields
  const winningTrades = metrics ? Math.round(metrics.total_trades * metrics.win_rate) : 0;
  const losingTrades = metrics ? metrics.total_trades - winningTrades : 0;

  const stats = [
    {
      label: 'Total Return',
      value: metrics ? `${(metrics.total_pnl * 100).toFixed(1)}%` : '0%',
      icon: TrendingUp,
      positive: (metrics?.total_pnl ?? 0) > 0,
    },
    {
      label: 'Sharpe Ratio',
      value: metrics?.sharpe_ratio?.toFixed(2) ?? '0',
      icon: Award,
      positive: true,
    },
    {
      label: 'Max Drawdown',
      value: metrics ? `${(metrics.max_drawdown * 100).toFixed(1)}%` : '0%',
      icon: TrendingDown,
      positive: false,
    },
    {
      label: 'Win Rate',
      value: metrics ? `${(metrics.win_rate * 100).toFixed(1)}%` : '0%',
      icon: Percent,
      positive: (metrics?.win_rate ?? 0) > 0.5,
    },
    {
      label: 'Profit Factor',
      value: metrics?.profit_factor?.toFixed(2) ?? '0',
      icon: Target,
      positive: (metrics?.profit_factor ?? 0) > 1,
    },
    {
      label: 'Total Trades',
      value: metrics?.total_trades?.toString() ?? '0',
      icon: Clock,
      positive: true,
    },
  ];

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-text-primary">Analytics</h1>
        <p className="text-text-secondary">Performance metrics and statistics</p>
      </div>

      {/* Error banner — visible when API fetch fails */}
      {error && (
        <div className="flex items-center gap-2 p-3 rounded-lg bg-red-500/10 border border-red-500/30 text-red-400 text-sm">
          <AlertCircle size={16} className="shrink-0" />
          <span>Failed to load analytics data: {error}</span>
        </div>
      )}

      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4">
        {stats.map((stat) => {
          const Icon = stat.icon;
          return (
            <Card key={stat.label} className="text-center">
              <Icon
                size={24}
                className={`mx-auto mb-2 ${stat.positive ? 'text-green-400' : 'text-red-400'}`}
              />
              <p className="text-2xl font-bold text-text-primary">{stat.value}</p>
              <p className="text-sm text-text-secondary">{stat.label}</p>
            </Card>
          );
        })}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card title="Equity Curve">
          <div className="h-80">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={equityCurve}>
                <CartesianGrid strokeDasharray="3 3" stroke="#333" />
                <XAxis dataKey="timestamp" stroke="#666" fontSize={12} />
                <YAxis stroke="#666" fontSize={12} domain={['auto', 'auto']} />
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#1a1a1a',
                    border: '1px solid #333',
                    borderRadius: '8px',
                  }}
                />
                <Line
                  type="monotone"
                  dataKey="value"
                  stroke="#F7931A"
                  strokeWidth={2}
                  dot={false}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </Card>

        <Card title="Monthly Returns">
          <div className="h-80">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={monthlyReturns}>
                <CartesianGrid strokeDasharray="3 3" stroke="#333" />
                <XAxis dataKey="month" stroke="#666" fontSize={12} />
                <YAxis stroke="#666" fontSize={12} />
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#1a1a1a',
                    border: '1px solid #333',
                    borderRadius: '8px',
                  }}
                />
                <Bar dataKey="return" fill="#F7931A" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </Card>
      </div>

      <Card title="Trade Distribution">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div className="text-center p-4 bg-background rounded-lg">
            <p className="text-3xl font-bold text-green-400">{winningTrades}</p>
            <p className="text-text-secondary">Winning Trades</p>
          </div>
          <div className="text-center p-4 bg-background rounded-lg">
            <p className="text-3xl font-bold text-red-400">{losingTrades}</p>
            <p className="text-text-secondary">Losing Trades</p>
          </div>
          <div className="text-center p-4 bg-background rounded-lg">
            {/* Net profit estimated from total_pnl (already in absolute terms) */}
            <p className="text-3xl font-bold text-bitcoin-orange">
              {metrics ? `${(metrics.total_pnl * 100).toFixed(1)}%` : '0%'}
            </p>
            <p className="text-text-secondary">Net Return</p>
          </div>
        </div>
      </Card>

      <Card title="Risk Metrics">
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="border-b border-border">
                <th className="text-left py-3 px-4 text-sm font-medium text-text-secondary">Metric</th>
                <th className="text-left py-3 px-4 text-sm font-medium text-text-secondary">Value</th>
                <th className="text-left py-3 px-4 text-sm font-medium text-text-secondary">Benchmark</th>
                <th className="text-left py-3 px-4 text-sm font-medium text-text-secondary">Status</th>
              </tr>
            </thead>
            <tbody>
              {[
                { metric: 'Volatility', value: '—', benchmark: '<15%', status: 'info' },
                { metric: 'VaR (95%)', value: '—', benchmark: '<$5,000', status: 'info' },
                {
                  metric: 'Sharpe Ratio',
                  value: metrics?.sharpe_ratio?.toFixed(2) ?? '0',
                  benchmark: '>1.0',
                  status: (metrics?.sharpe_ratio ?? 0) > 1.0 ? 'good' : 'warning',
                },
                {
                  metric: 'Max Drawdown',
                  value: metrics ? `${(metrics.max_drawdown * 100).toFixed(1)}%` : '0%',
                  benchmark: '<20%',
                  status: (metrics?.max_drawdown ?? 1) < 0.2 ? 'good' : 'warning',
                },
                {
                  metric: 'Profit Factor',
                  value: metrics?.profit_factor?.toFixed(2) ?? '0',
                  benchmark: '>1.5',
                  status: (metrics?.profit_factor ?? 0) > 1.5 ? 'good' : 'warning',
                },
              ].map((row, i) => (
                <tr key={i} className="border-b border-border/50">
                  <td className="py-3 px-4 text-sm text-text-primary">{row.metric}</td>
                  <td className="py-3 px-4 text-sm text-text-primary">{row.value}</td>
                  <td className="py-3 px-4 text-sm text-text-secondary">{row.benchmark}</td>
                  <td className="py-3 px-4">
                    <span
                      className={`px-2 py-1 text-xs font-medium rounded ${
                        row.status === 'good'
                          ? 'bg-green-500/10 text-green-400'
                          : row.status === 'warning'
                          ? 'bg-yellow-500/10 text-yellow-400'
                          : 'bg-gray-500/10 text-gray-400'
                      }`}
                    >
                      {row.status === 'info' ? 'n/a' : row.status}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  );
}
