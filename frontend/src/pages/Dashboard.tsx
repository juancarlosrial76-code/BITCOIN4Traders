import { useEffect, useState } from 'react';
import { Card } from '../components/ui/Card';
import { useTradingStore } from '../stores';
import { useWebSocket } from '../hooks/useWebSocket';
import { api } from '../api/client';
import {
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  AreaChart,
  Area,
} from 'recharts';
import { TrendingUp, TrendingDown, DollarSign, Activity, Wallet, Percent } from 'lucide-react';

interface PricePoint {
  time: string;
  price: number;
}

export function Dashboard() {
  const { isRunning, setIsRunning, setCurrentPrice, addPricePoint } = useTradingStore();
  const { isConnected, lastPrice } = useWebSocket();
  const [priceHistory, setPriceHistory] = useState<PricePoint[]>([]);
  const [portfolioValue] = useState('$42,847.32');
  const [dailyPnL] = useState('$1,024.56');
  const [winRate, setWinRate] = useState('68.5%');
  const [activeTrades, setActiveTrades] = useState('3');

  useEffect(() => {
    const fetchData = async () => {
      try {
        const status = await api.trading.getStatus();
        setIsRunning(status.is_running);
        const metrics = await api.analytics.getMetrics();
        setWinRate(`${(metrics.winRate * 100).toFixed(1)}%`);
        setActiveTrades(metrics.totalTrades.toString());
      } catch (e) {
        console.error('Failed to fetch data:', e);
      }
    };
    fetchData();
  }, [setIsRunning]);

  useEffect(() => {
    if (lastPrice) {
      setCurrentPrice(lastPrice);
      const now = new Date();
      const time = now.toLocaleTimeString();
      setPriceHistory((prev) => {
        const newHistory = [...prev, { time, price: lastPrice }].slice(-50);
        return newHistory;
      });
      addPricePoint(Date.now(), lastPrice);
    }
  }, [lastPrice, setCurrentPrice, addPricePoint]);

  const price = lastPrice || 43250.87;
  const chartData = priceHistory.length > 0 ? priceHistory : Array.from({ length: 50 }, (_, i) => ({
    time: new Date(Date.now() - (50 - i) * 60000).toLocaleTimeString(),
    price: 42000 + Math.random() * 2000 - 1000,
  }));

  const metrics = [
    { label: 'Portfolio Value', value: portfolioValue, icon: Wallet, change: '+2.4%', positive: true },
    { label: '24h P&L', value: dailyPnL, icon: DollarSign, change: '+5.1%', positive: true },
    { label: 'Win Rate', value: winRate, icon: Percent, change: '+2.1%', positive: true },
    { label: 'Active Trades', value: activeTrades, icon: Activity, change: '0%', positive: true },
  ];

  return (
    <div className="space-y-6">
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold text-text-primary">Dashboard</h1>
          <p className="text-text-secondary">Overview of your trading bot</p>
        </div>
        <div className="flex items-center gap-3">
          <div className={`flex items-center gap-2 px-3 py-1.5 rounded-lg ${isConnected ? 'bg-green-500/10' : 'bg-gray-500/10'}`}>
            <div className={`w-2 h-2 rounded-full ${isConnected ? 'bg-green-500 animate-pulse' : 'bg-gray-500'}`}></div>
            <span className="text-sm text-text-secondary">{isConnected ? 'Live' : 'Disconnected'}</span>
          </div>
          <div className="flex items-center gap-2">
            <div className={`w-3 h-3 rounded-full ${isRunning ? 'bg-green-500 animate-pulse' : 'bg-gray-500'}`}></div>
            <span className="text-sm text-text-secondary">{isRunning ? 'Bot Running' : 'Bot Stopped'}</span>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        {metrics.map((metric) => {
          const Icon = metric.icon;
          return (
            <Card key={metric.label} className="hover:border-bitcoin-orange/30 transition-colors">
              <div className="flex items-start justify-between">
                <div>
                  <p className="text-sm text-text-secondary">{metric.label}</p>
                  <p className="text-2xl font-bold text-text-primary mt-1">{metric.value}</p>
                  <p className={`text-sm mt-1 ${metric.positive ? 'text-green-400' : 'text-red-400'}`}>
                    {metric.change}
                  </p>
                </div>
                <div className="p-2 bg-bitcoin-orange/10 rounded-lg">
                  <Icon className="text-bitcoin-orange" size={20} />
                </div>
              </div>
            </Card>
          );
        })}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <Card className="lg:col-span-2" title="BTC/USDT Price Chart">
          <div className="h-80">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={chartData}>
                <defs>
                  <linearGradient id="priceGradient" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#F7931A" stopOpacity={0.3} />
                    <stop offset="95%" stopColor="#F7931A" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="#333" />
                <XAxis dataKey="time" stroke="#666" fontSize={12} />
                <YAxis stroke="#666" fontSize={12} domain={['auto', 'auto']} />
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#1a1a1a',
                    border: '1px solid #333',
                    borderRadius: '8px',
                  }}
                  labelStyle={{ color: '#fff' }}
                />
                <Area
                  type="monotone"
                  dataKey="price"
                  stroke="#F7931A"
                  fill="url(#priceGradient)"
                  strokeWidth={2}
                />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </Card>

        <Card title="Quick Actions">
          <div className="space-y-3">
            <button className="w-full flex items-center gap-3 p-3 bg-green-500/10 hover:bg-green-500/20 border border-green-500/20 rounded-lg transition-colors">
              <TrendingUp className="text-green-400" size={20} />
              <span className="text-green-400 font-medium">Place Buy Order</span>
            </button>
            <button className="w-full flex items-center gap-3 p-3 bg-red-500/10 hover:bg-red-500/20 border border-red-500/20 rounded-lg transition-colors">
              <TrendingDown className="text-red-400" size={20} />
              <span className="text-red-400 font-medium">Place Sell Order</span>
            </button>
            <div className="pt-3 border-t border-border">
              <div className="flex justify-between text-sm mb-2">
                <span className="text-text-secondary">Current Price</span>
                <span className="text-text-primary font-medium">${price.toLocaleString()}</span>
              </div>
              <div className="flex justify-between text-sm mb-2">
                <span className="text-text-secondary">24h High</span>
                <span className="text-text-primary">${(price * 1.023).toLocaleString()}</span>
              </div>
              <div className="flex justify-between text-sm">
                <span className="text-text-secondary">24h Low</span>
                <span className="text-text-primary">${(price * 0.985).toLocaleString()}</span>
              </div>
            </div>
          </div>
        </Card>
      </div>

      <Card title="Recent Trades">
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="border-b border-border">
                <th className="text-left py-3 px-4 text-sm font-medium text-text-secondary">Time</th>
                <th className="text-left py-3 px-4 text-sm font-medium text-text-secondary">Type</th>
                <th className="text-left py-3 px-4 text-sm font-medium text-text-secondary">Price</th>
                <th className="text-left py-3 px-4 text-sm font-medium text-text-secondary">Amount</th>
                <th className="text-left py-3 px-4 text-sm font-medium text-text-secondary">P&L</th>
              </tr>
            </thead>
            <tbody>
              {[
                { time: '14:32:15', type: 'BUY', price: 43120.50, amount: 0.05, pnl: null },
                { time: '14:28:42', type: 'SELL', price: 43250.00, amount: 0.03, pnl: 45.20 },
                { time: '14:15:33', type: 'BUY', price: 42980.25, amount: 0.08, pnl: null },
                { time: '13:52:18', type: 'SELL', price: 42850.00, amount: 0.05, pnl: -12.50 },
                { time: '13:41:05', type: 'BUY', price: 42750.80, amount: 0.1, pnl: null },
              ].map((trade, i) => (
                <tr key={i} className="border-b border-border/50 hover:bg-background/50">
                  <td className="py-3 px-4 text-sm text-text-secondary">{trade.time}</td>
                  <td className={`py-3 px-4 text-sm font-medium ${trade.type === 'BUY' ? 'text-green-400' : 'text-red-400'}`}>
                    {trade.type}
                  </td>
                  <td className="py-3 px-4 text-sm text-text-primary">${trade.price.toLocaleString()}</td>
                  <td className="py-3 px-4 text-sm text-text-secondary">{trade.amount} BTC</td>
                  <td className={`py-3 px-4 text-sm font-medium ${trade.pnl && trade.pnl >= 0 ? 'text-green-400' : trade.pnl ? 'text-red-400' : 'text-text-muted'}`}>
                    {trade.pnl ? `$${trade.pnl.toFixed(2)}` : '-'}
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
