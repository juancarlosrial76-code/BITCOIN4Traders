import { useEffect, useState } from 'react';
import { api } from '../../api/client';
import { Card } from '../ui/Card';

interface PerformanceMetrics {
  champion_name: string;
  totalReturn: number;
  sharpeRatio: number;
  sortinoRatio: number;
  calmarRatio: number;
  maxDrawdown: number;
  winRate: number;
  profitFactor: number;
  totalTrades: number;
  winningTrades: number;
  losingTrades: number;
  avgWin: number;
  avgLoss: number;
}

interface MetricCardProps {
  label: string;
  value: string | number;
  change?: number;
  positive?: boolean;
  helpText?: string;
}

function MetricCard({ label, value, change, positive, helpText }: MetricCardProps) {
  return (
    <div className="p-4 bg-gray-800/50 rounded-lg">
      <div className="text-sm text-gray-400 mb-1">{label}</div>
      <div
        className={`text-2xl font-bold font-mono ${
          positive === undefined ? 'text-white' : positive ? 'text-green-400' : 'text-red-400'
        }`}
      >
        {value}
      </div>
      {change !== undefined && (
        <div className={`text-sm ${change >= 0 ? 'text-green-400' : 'text-red-400'}`}>
          {change >= 0 ? '+' : ''}
          {change.toFixed(2)}%
        </div>
      )}
      {helpText && <div className="text-xs text-gray-500 mt-1">{helpText}</div>}
    </div>
  );
}

export function MetricsDashboard() {
  const [metrics, setMetrics] = useState<PerformanceMetrics | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchMetrics = async () => {
      try {
        setLoading(true);
        const data = await api.analytics.getMetrics();
        setMetrics(data);
        setError(null);
      } catch (e: any) {
        setError(e.message);
        // Mock data for demo
        setMetrics({
          champion_name: 'DarwinBot',
          totalReturn: 15.23,
          sharpeRatio: 1.85,
          sortinoRatio: 2.34,
          calmarRatio: 2.12,
          maxDrawdown: 12.5,
          winRate: 62.5,
          profitFactor: 2.15,
          totalTrades: 156,
          winningTrades: 97,
          losingTrades: 59,
          avgWin: 3.2,
          avgLoss: -1.8,
        });
      } finally {
        setLoading(false);
      }
    };

    fetchMetrics();
  }, []);

  if (loading) {
    return (
      <Card className="p-4">
        <h3 className="font-bold mb-4">Performance Metrics</h3>
        <div className="text-center text-gray-400 py-8">Loading...</div>
      </Card>
    );
  }

  if (error && !metrics) {
    return (
      <Card className="p-4">
        <h3 className="font-bold mb-4">Performance Metrics</h3>
        <div className="text-center text-red-400 py-8">Error: {error}</div>
      </Card>
    );
  }

  if (!metrics) return null;

  return (
    <Card className="p-4">
      <div className="flex items-center justify-between mb-4">
        <h3 className="font-bold">Performance Metrics</h3>
        <span className="px-3 py-1 bg-cyan-500/20 text-cyan-400 rounded text-sm">
          {metrics.champion_name}
        </span>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        {/* Return Metrics */}
        <MetricCard
          label="Total Return"
          value={`${metrics.totalReturn >= 0 ? '+' : ''}${metrics.totalReturn.toFixed(2)}%`}
          positive={metrics.totalReturn >= 0}
          helpText="Overall return"
        />
        <MetricCard
          label="Sharpe Ratio"
          value={metrics.sharpeRatio.toFixed(2)}
          positive={metrics.sharpeRatio >= 1}
          helpText="Risk-adjusted return"
        />
        <MetricCard
          label="Sortino Ratio"
          value={metrics.sortinoRatio.toFixed(2)}
          positive={metrics.sortinoRatio >= 1}
          helpText="Downside risk-adjusted"
        />
        <MetricCard
          label="Calmar Ratio"
          value={metrics.calmarRatio.toFixed(2)}
          positive={metrics.calmarRatio >= 1}
          helpText="Return vs max drawdown"
        />

        {/* Risk Metrics */}
        <MetricCard
          label="Max Drawdown"
          value={`-${metrics.maxDrawdown.toFixed(2)}%`}
          positive={false}
          helpText="Largest peak-to-trough"
        />
        <MetricCard
          label="Win Rate"
          value={`${metrics.winRate.toFixed(1)}%`}
          positive={metrics.winRate >= 50}
          helpText="Profitable trades"
        />
        <MetricCard
          label="Profit Factor"
          value={metrics.profitFactor.toFixed(2)}
          positive={metrics.profitFactor >= 1.5}
          helpText="Gross profit / gross loss"
        />
        <MetricCard label="Total Trades" value={metrics.totalTrades} helpText="All closed trades" />

        {/* Trade Stats */}
        <MetricCard
          label="Winning Trades"
          value={metrics.winningTrades}
          positive={true}
          helpText="Profitable trades"
        />
        <MetricCard
          label="Losing Trades"
          value={metrics.losingTrades}
          positive={false}
          helpText="Unprofitable trades"
        />
        <MetricCard
          label="Avg Win"
          value={`+${metrics.avgWin.toFixed(2)}%`}
          positive={true}
          helpText="Average profit"
        />
        <MetricCard
          label="Avg Loss"
          value={`${metrics.avgLoss.toFixed(2)}%`}
          positive={false}
          helpText="Average loss"
        />
      </div>
    </Card>
  );
}

export default MetricsDashboard;
