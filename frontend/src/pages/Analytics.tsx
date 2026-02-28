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
import { TrendingUp, TrendingDown, Award, Target, Clock, Percent } from 'lucide-react';

export function Analytics() {
  const [metrics, setMetrics] = useState<any>(null);
  const [equityCurve, setEquityCurve] = useState<any[]>([]);
  const [monthlyReturns, setMonthlyReturns] = useState<any[]>([]);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const [metricsData, equityData, returnsData] = await Promise.all([
          api.analytics.getMetrics(),
          api.analytics.getEquityCurve(),
          api.analytics.getMonthlyReturns(),
        ]);
        setMetrics(metricsData);
        setEquityCurve(equityData);
        setMonthlyReturns(returnsData);
      } catch (e) {
        console.error('Failed to fetch analytics:', e);
      }
    };
    fetchData();
  }, []);

  const stats = [
    { label: 'Total Return', value: metrics ? `${(metrics.totalReturn * 100).toFixed(1)}%` : '0%', icon: TrendingUp, positive: metrics?.totalReturn > 0 },
    { label: 'Sharpe Ratio', value: metrics?.sharpeRatio?.toFixed(2) || '0', icon: Award, positive: true },
    { label: 'Max Drawdown', value: metrics ? `${(metrics.maxDrawdown * 100).toFixed(1)}%` : '0%', icon: TrendingDown, positive: false },
    { label: 'Win Rate', value: metrics ? `${(metrics.winRate * 100).toFixed(1)}%` : '0%', icon: Percent, positive: metrics?.winRate > 0.5 },
    { label: 'Profit Factor', value: metrics?.profitFactor?.toFixed(2) || '0', icon: Target, positive: metrics?.profitFactor > 1 },
    { label: 'Total Trades', value: metrics?.totalTrades?.toString() || '0', icon: Clock, positive: true },
  ];

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-text-primary">Analytics</h1>
        <p className="text-text-secondary">Performance metrics and statistics</p>
      </div>

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
            <p className="text-3xl font-bold text-green-400">{metrics?.winningTrades || 0}</p>
            <p className="text-text-secondary">Winning Trades</p>
          </div>
          <div className="text-center p-4 bg-background rounded-lg">
            <p className="text-3xl font-bold text-red-400">{metrics?.losingTrades || 0}</p>
            <p className="text-text-secondary">Losing Trades</p>
          </div>
          <div className="text-center p-4 bg-background rounded-lg">
            <p className="text-3xl font-bold text-bitcoin-orange">
              ${((metrics?.avgWin || 0) * (metrics?.winningTrades || 0) - (metrics?.avgLoss || 0) * (metrics?.losingTrades || 0)).toFixed(0)}
            </p>
            <p className="text-text-secondary">Net Profit</p>
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
                { metric: 'Volatility', value: '12.5%', benchmark: '<15%', status: 'good' },
                { metric: 'Beta', value: '0.85', benchmark: '<1.0', status: 'good' },
                { metric: 'VaR (95%)', value: '$2,340', benchmark: '<$5,000', status: 'good' },
                { metric: 'Sortino Ratio', value: metrics?.sortinoRatio?.toFixed(2) || '0', benchmark: '>1.5', status: (metrics?.sortinoRatio || 0) > 1.5 ? 'good' : 'warning' },
                { metric: 'Calmar Ratio', value: metrics?.calmarRatio?.toFixed(2) || '0', benchmark: '>1.0', status: (metrics?.calmarRatio || 0) > 1.0 ? 'good' : 'warning' },
              ].map((row, i) => (
                <tr key={i} className="border-b border-border/50">
                  <td className="py-3 px-4 text-sm text-text-primary">{row.metric}</td>
                  <td className="py-3 px-4 text-sm text-text-primary">{row.value}</td>
                  <td className="py-3 px-4 text-sm text-text-secondary">{row.benchmark}</td>
                  <td className="py-3 px-4">
                    <span className={`px-2 py-1 text-xs font-medium rounded ${
                      row.status === 'good' ? 'bg-green-500/10 text-green-400' : 'bg-yellow-500/10 text-yellow-400'
                    }`}>
                      {row.status}
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
