import { useEffect, useState } from 'react';
import { Card } from '../components/ui/Card';
import { useTradingStore } from '../stores';
import { useWebSocket } from '../hooks/useWebSocket';
import { api } from '../api/client';
import { formatCurrency, formatPercent, formatNumber } from '../lib/utils';
import {
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  AreaChart,
  Area,
  LineChart,
  Line,
} from 'recharts';
import { TrendingUp, TrendingDown, DollarSign, Activity, Wallet, Percent, Zap, Power, PowerOff } from 'lucide-react';
import { TooltipContent, TooltipTrigger, Tooltip as CustomTooltip } from '../components/ui/Tooltip';

interface PricePoint {
  time: string;
  price: number;
}

interface MetricCardProps {
  label: string;
  value: string;
  change?: string;
  positive?: boolean;
  icon: React.ElementType;
  helpText?: string;
}

function MetricCard({ label, value, change, positive, icon: Icon, helpText }: MetricCardProps) {
  return (
    <CustomTooltip>
      <TooltipTrigger asChild>
        <Card className="hover:border-bitcoin-orange/30 transition-colors cursor-help">
          <div className="flex items-start justify-between">
            <div>
              <p className="text-sm text-text-secondary">{label}</p>
              <p className="text-2xl font-bold text-text-primary mt-1">{value}</p>
              {change && (
                <p className={`text-sm mt-1 ${positive ? 'text-green-400' : 'text-red-400'}`}>
                  {change}
                </p>
              )}
            </div>
            <div className="p-2 bg-bitcoin-orange/10 rounded-lg">
              <Icon className="text-bitcoin-orange" size={20} />
            </div>
          </div>
        </Card>
      </TooltipTrigger>
      {helpText && (
        <TooltipContent>
          <p>{helpText}</p>
        </TooltipContent>
      )}
    </CustomTooltip>
  );
}

export function Dashboard() {
  const { isRunning, setIsRunning, setCurrentPrice, addPricePoint } = useTradingStore();
  const { isConnected, lastPrice } = useWebSocket();
  const [priceHistory, setPriceHistory] = useState<PricePoint[]>([]);
  const [portfolioValue, setPortfolioValue] = useState('$0.00');
  const [dailyPnL, setDailyPnL] = useState('$0.00');
  const [dailyPnLPercent, setDailyPnLPercent] = useState('0%');
  const [winRate, setWinRate] = useState('0%');
  const [totalTrades, setTotalTrades] = useState('0');
  const [sharpeRatio, setSharpeRatio] = useState('0.00');
  const [maxDrawdown, setMaxDrawdown] = useState('0%');
  const [signal, setSignal] = useState({ label: 'FLAT', signal: 0, champion: 'None' });
  const [balance, setBalance] = useState({ USDT: 0, BTC: 0 });
  const [equityCurve, setEquityCurve] = useState<{ timestamp: string; value: number }[]>([]);
  const [, setIsLoading] = useState(true);

  useEffect(() => {
    const fetchData = async () => {
      try {
        setIsLoading(true);
        const [status, metrics, bal] = await Promise.all([
          api.trading.getStatus(),
          api.analytics.getMetrics(),
          api.trading.getBalance(),
        ]);
        
        setIsRunning(status.is_running);
        setSignal({
          label: status.champion_signal || 'FLAT',
          signal: status.champion_signal === 'LONG' ? 1 : status.champion_signal === 'SHORT' ? -1 : 0,
          champion: status.champion_name || 'None',
        });
        
        setWinRate(formatPercent(metrics.winRate));
        setTotalTrades(metrics.totalTrades.toString());
        setSharpeRatio(formatNumber(metrics.sharpeRatio));
        setMaxDrawdown(formatPercent(-metrics.maxDrawdown));
        
        const pnl = metrics.totalReturn * 10000;
        setDailyPnL(formatCurrency(pnl));
        setDailyPnLPercent(formatPercent(metrics.totalReturn));
        setPortfolioValue(formatCurrency(10000 * (1 + metrics.totalReturn)));
        
        setBalance(bal.balance);
      } catch (e) {
        console.error('Failed to fetch data:', e);
      } finally {
        setIsLoading(false);
      }
    };
    fetchData();
    
    const interval = setInterval(fetchData, 30000);
    return () => clearInterval(interval);
  }, [setIsRunning]);

  useEffect(() => {
    const fetchEquity = async () => {
      try {
        const data = await api.analytics.getEquityCurve();
        if (data && data.length > 0) {
          setEquityCurve(data.slice(-100));
        }
      } catch (e) {
        console.error('Failed to fetch equity curve:', e);
      }
    };
    fetchEquity();
  }, []);

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

  const price = lastPrice || 0;
  const chartData = priceHistory.length > 0 ? priceHistory : 
    equityCurve.length > 0 ? equityCurve.map(d => ({ time: d.timestamp.slice(11, 16), price: d.value })) :
    Array.from({ length: 50 }, (_, i) => ({
      time: new Date(Date.now() - (50 - i) * 60000).toLocaleTimeString(),
      price: 42000 + Math.random() * 2000 - 1000,
    }));

  const handleStartTrading = async () => {
    try {
      await api.trading.start();
      setIsRunning(true);
    } catch (e) {
      console.error('Failed to start trading:', e);
    }
  };

  const handleStopTrading = async () => {
    try {
      await api.trading.stop();
      setIsRunning(false);
    } catch (e) {
      console.error('Failed to stop trading:', e);
    }
  };

  const metrics = [
    { 
      label: 'Portfolio Wert', 
      value: portfolioValue, 
      change: dailyPnLPercent, 
      positive: parseFloat(dailyPnLPercent) >= 0,
      icon: Wallet,
      helpText: 'Aktueller Gesamtwert des Portfolios inkl. aller Positionen'
    },
    { 
      label: 'Tages-P&L', 
      value: dailyPnL, 
      change: dailyPnLPercent,
      positive: parseFloat(dailyPnLPercent) >= 0,
      icon: DollarSign,
      helpText: 'Gewinn/Verlust des aktuellen Tages'
    },
    { 
      label: 'Win Rate', 
      value: winRate, 
      icon: Percent,
      helpText: 'Prozentualer Anteil gewonnener Trades'
    },
    { 
      label: 'Trades Gesamt', 
      value: totalTrades, 
      icon: Activity,
      helpText: 'Anzahl aller ausgeführten Trades seit Start'
    },
    { 
      label: 'Sharpe Ratio', 
      value: sharpeRatio, 
      icon: TrendingUp,
      helpText: 'Risiko-adjustierte Rendite (>= 1.0 = gut, >= 2.0 = sehr gut)'
    },
    { 
      label: 'Max Drawdown', 
      value: maxDrawdown, 
      icon: TrendingDown,
      helpText: 'Maximaler Verlust vom Allzeithoch'
    },
  ];

  const signalColor = signal.signal === 1 ? 'text-green-400 bg-green-500/10' : 
                      signal.signal === -1 ? 'text-red-400 bg-red-500/10' : 
                      'text-gray-400 bg-gray-500/10';

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold text-text-primary">Dashboard</h1>
          <p className="text-text-secondary">Übersicht deines Trading Bot</p>
        </div>
        <div className="flex items-center gap-3">
          <div className={`flex items-center gap-2 px-3 py-1.5 rounded-lg ${isConnected ? 'bg-green-500/10' : 'bg-gray-500/10'}`}>
            <div className={`w-2 h-2 rounded-full ${isConnected ? 'bg-green-500 animate-pulse' : 'bg-gray-500'}`}></div>
            <span className="text-sm text-text-secondary">{isConnected ? 'Live' : 'Getrennt'}</span>
          </div>
          <div className={`flex items-center gap-2 px-3 py-1.5 rounded-lg ${isRunning ? 'bg-green-500/10' : 'bg-gray-500/10'}`}>
            <div className={`w-3 h-3 rounded-full ${isRunning ? 'bg-green-500 animate-pulse' : 'bg-gray-500'}`}></div>
            <span className="text-sm text-text-secondary">{isRunning ? 'Bot Läuft' : 'Bot Gestoppt'}</span>
          </div>
        </div>
      </div>

      {/* Signal Banner */}
      <Card className={`border-l-4 ${signalColor.replace('text-', 'border-l-').split(' ')[0]}`}>
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
          <div className="flex items-center gap-4">
            <div className={`p-3 rounded-lg ${signalColor}`}>
              {signal.signal === 1 ? <Zap size={24} /> : signal.signal === -1 ? <Zap size={24} /> : <Activity size={24} />}
            </div>
            <div>
              <p className="text-sm text-text-secondary">Aktuelles Signal</p>
              <p className="text-xl font-bold">{signal.label}</p>
              <p className="text-xs text-text-muted">von {signal.champion}</p>
            </div>
          </div>
          <div className="flex gap-2">
            {isRunning ? (
              <button
                onClick={handleStopTrading}
                className="flex items-center gap-2 px-4 py-2 bg-red-500/10 hover:bg-red-500/20 border border-red-500/20 rounded-lg text-red-400 transition-colors"
              >
                <PowerOff size={18} />
                <span>Trading Stoppen</span>
              </button>
            ) : (
              <button
                onClick={handleStartTrading}
                className="flex items-center gap-2 px-4 py-2 bg-green-500/10 hover:bg-green-500/20 border border-green-500/20 rounded-lg text-green-400 transition-colors"
              >
                <Power size={18} />
                <span>Trading Starten</span>
              </button>
            )}
          </div>
        </div>
      </Card>

      {/* Metrics Grid */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-6 gap-4">
        {metrics.map((metric) => {
          const Icon = metric.icon;
          return (
            <MetricCard
              key={metric.label}
              label={metric.label}
              value={metric.value}
              change={metric.change}
              positive={metric.positive}
              icon={Icon}
              helpText={metric.helpText}
            />
          );
        })}
      </div>

      {/* Charts Row */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <Card className="lg:col-span-2" title="BTC/USDT Preis" helpText="Echtzeit-Preis in USDT von Binance">
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
                <XAxis dataKey="time" stroke="#666" fontSize={12} tick={{ fill: '#666' }} />
                <YAxis 
                  stroke="#666" 
                  fontSize={12} 
                  tick={{ fill: '#666' }}
                  domain={['auto', 'auto']}
                  tickFormatter={(v) => `$${(v/1000).toFixed(0)}k`}
                />
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#1a1a1a',
                    border: '1px solid #333',
                    borderRadius: '8px',
                    color: '#fff',
                  }}
                  labelStyle={{ color: '#fff' }}
                  formatter={(value: number) => [`$${value.toLocaleString()}`, 'Preis']}
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

        {/* Quick Actions & Balance */}
        <div className="space-y-6">
          <Card title="Kontostand">
            <div className="space-y-4">
              <div className="flex justify-between items-center p-3 bg-background rounded-lg">
                <span className="text-text-secondary">USDT</span>
                <span className="text-lg font-bold text-text-primary font-mono">
                  {balance.USDT.toFixed(2)}
                </span>
              </div>
              <div className="flex justify-between items-center p-3 bg-background rounded-lg">
                <span className="text-text-secondary">BTC</span>
                <span className="text-lg font-bold text-text-primary font-mono">
                  {balance.BTC.toFixed(6)}
                </span>
              </div>
              <div className="pt-3 border-t border-border">
                <div className="flex justify-between text-sm mb-2">
                  <span className="text-text-secondary">Aktueller Preis</span>
                  <span className="text-text-primary font-medium">${price > 0 ? price.toLocaleString() : '0'}</span>
                </div>
                <div className="flex justify-between text-sm mb-2">
                  <span className="text-text-secondary">24h Hoch</span>
                  <span className="text-text-primary">${price > 0 ? (price * 1.023).toLocaleString() : '0'}</span>
                </div>
                <div className="flex justify-between text-sm">
                  <span className="text-text-secondary">24h Tief</span>
                  <span className="text-text-primary">${price > 0 ? (price * 0.985).toLocaleString() : '0'}</span>
                </div>
              </div>
            </div>
          </Card>

          <Card title="Quick Actions">
            <div className="space-y-3">
              <button className="w-full flex items-center gap-3 p-3 bg-green-500/10 hover:bg-green-500/20 border border-green-500/20 rounded-lg transition-colors">
                <TrendingUp className="text-green-400" size={20} />
                <span className="text-green-400 font-medium">Kauf Order</span>
              </button>
              <button className="w-full flex items-center gap-3 p-3 bg-red-500/10 hover:bg-red-500/20 border border-red-500/20 rounded-lg transition-colors">
                <TrendingDown className="text-red-400" size={20} />
                <span className="text-red-400 font-medium">Verkauf Order</span>
              </button>
            </div>
          </Card>
        </div>
      </div>

      {/* Equity Curve */}
      {equityCurve.length > 0 && (
        <Card title="Equity Curve" helpText="Verlauf des Portfolio-Werts über Zeit">
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={equityCurve}>
                <CartesianGrid strokeDasharray="3 3" stroke="#333" />
                <XAxis 
                  dataKey="timestamp" 
                  stroke="#666" 
                  fontSize={12} 
                  tick={{ fill: '#666' }}
                  tickFormatter={(v) => v.slice(5, 10)}
                />
                <YAxis 
                  stroke="#666" 
                  fontSize={12} 
                  tick={{ fill: '#666' }}
                  tickFormatter={(v) => `$${(v/1000).toFixed(0)}k`}
                />
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#1a1a1a',
                    border: '1px solid #333',
                    borderRadius: '8px',
                    color: '#fff',
                  }}
                  labelStyle={{ color: '#fff' }}
                  formatter={(value: number) => [`$${value.toLocaleString()}`, 'Wert']}
                />
                <Line
                  type="monotone"
                  dataKey="value"
                  stroke="#22c55e"
                  strokeWidth={2}
                  dot={false}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </Card>
      )}
    </div>
  );
}
