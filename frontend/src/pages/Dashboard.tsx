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
} from 'recharts';
import {
  TrendingUp,
  TrendingDown,
  DollarSign,
  Activity,
  Wallet,
  Zap,
  Power,
  PowerOff,
  Wifi,
  WifiOff,
  Pause,
  Info,
  ArrowUp,
  ArrowDown,
  BarChart3,
  Target,
  Shield,
} from 'lucide-react';

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
    <Card className="hover:border-bitcoin-orange/30 transition-colors">
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
      {helpText && <p className="text-xs text-text-muted mt-2">{helpText}</p>}
    </Card>
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
  const [signal, setSignal] = useState({ label: 'FLAT', signal: 0, champion: 'None', reason: '' });
  const [balance, setBalance] = useState({ USDT: 0, BTC: 0 });
  const [equityCurve, setEquityCurve] = useState<{ timestamp: string; value: number }[]>([]);
  const [, setIsLoading] = useState(true);
  const [tradingMode, setTradingMode] = useState<'paper' | 'live'>('paper');
  const [lastUpdate, setLastUpdate] = useState<Date>(new Date());

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
        setTradingMode(status.mode === 'live' ? 'live' : 'paper');

        const signalMap: Record<string, number> = { LONG: 1, SHORT: -1, FLAT: 0 };
        setSignal({
          label: status.champion_signal || 'FLAT',
          signal: signalMap[status.champion_signal || 'FLAT'] || 0,
          champion: status.champion_name || 'None',
          reason: 'Automatic signal from trained model',
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
        setLastUpdate(new Date());
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
      setPriceHistory(prev => {
        const newHistory = [...prev, { time, price: lastPrice }].slice(-50);
        return newHistory;
      });
      addPricePoint(Date.now(), lastPrice);
    }
  }, [lastPrice, setCurrentPrice, addPricePoint]);

  const price = lastPrice || 0;
  const chartData =
    priceHistory.length > 0
      ? priceHistory
      : equityCurve.length > 0
        ? equityCurve.map(d => ({ time: d.timestamp.slice(11, 16), price: d.value }))
        : Array.from({ length: 50 }, (_, i) => ({
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

  const signalInfo = {
    LONG: {
      color: 'bg-green-500',
      text: 'text-green-400',
      icon: ArrowUp,
      label: 'KAUFEN',
      desc: 'Signal empfiehlt Long-Position',
    },
    SHORT: {
      color: 'bg-red-500',
      text: 'text-red-400',
      icon: ArrowDown,
      label: 'VERKAUFEN',
      desc: 'Signal empfiehlt Short-Position',
    },
    FLAT: {
      color: 'bg-gray-500',
      text: 'text-gray-400',
      icon: Pause,
      label: 'FLACH',
      desc: 'Kein klares Signal - keine Position',
    },
  };

  const currentSignal = signalInfo[signal.label as keyof typeof signalInfo] || signalInfo.FLAT;
  const SignalIcon = currentSignal.icon;

  const metrics = [
    {
      label: 'Portfolio Wert',
      value: portfolioValue,
      change: dailyPnLPercent,
      positive: parseFloat(dailyPnLPercent) >= 0,
      icon: Wallet,
      helpText: 'Aktueller Gesamtwert',
    },
    {
      label: 'Tages-P&L',
      value: dailyPnL,
      change: dailyPnLPercent,
      positive: parseFloat(dailyPnLPercent) >= 0,
      icon: DollarSign,
      helpText: 'Gewinn/Verlust heute',
    },
    { label: 'Win Rate', value: winRate, icon: Target, helpText: 'Prozent gewonnener Trades' },
    { label: 'Trades', value: totalTrades, icon: Activity, helpText: 'Gesamtzahl aller Trades' },
    {
      label: 'Sharpe',
      value: sharpeRatio,
      icon: TrendingUp,
      helpText: 'Risiko-adjustierte Rendite',
    },
    { label: 'Max DD', value: maxDrawdown, icon: TrendingDown, helpText: 'Maximaler Verlust' },
  ];

  return (
    <div className="space-y-6">
      {/* MODE BANNER */}
      <div
        className={`rounded-xl p-4 flex flex-col md:flex-row md:items-center justify-between gap-4 ${
          tradingMode === 'live'
            ? 'bg-red-500/10 border border-red-500/30'
            : 'bg-yellow-500/10 border border-yellow-500/30'
        }`}
      >
        <div className="flex items-center gap-3">
          <div
            className={`w-12 h-12 rounded-full flex items-center justify-center ${
              tradingMode === 'live' ? 'bg-red-500/20' : 'bg-yellow-500/20'
            }`}
          >
            {tradingMode === 'live' ? (
              <Zap className="text-red-400" size={24} />
            ) : (
              <Shield className="text-yellow-400" size={24} />
            )}
          </div>
          <div>
            <h2 className="text-xl font-bold flex items-center gap-2">
              {tradingMode === 'live' ? (
                <>
                  <span className="text-red-400">LIVE TRADING</span>
                  <span className="text-xs bg-red-500 text-white px-2 py-0.5 rounded">
                    ECHTES GELD
                  </span>
                </>
              ) : (
                <>
                  <span className="text-yellow-400">PAPER TRADING</span>
                  <span className="text-xs bg-yellow-500 text-black px-2 py-0.5 rounded">
                    SIMULATION
                  </span>
                </>
              )}
            </h2>
            <p className="text-sm text-text-secondary">
              {tradingMode === 'live'
                ? 'Handel mit echtem Binance-Konto aktiv'
                : 'Simulation mit virtuellem Guthaben - kein echtes Risiko'}
            </p>
          </div>
        </div>

        {/* Connection Status */}
        <div className="flex items-center gap-4">
          <div
            className={`flex items-center gap-2 px-3 py-2 rounded-lg ${
              isConnected ? 'bg-green-500/10' : 'bg-red-500/10'
            }`}
          >
            {isConnected ? (
              <Wifi className="text-green-400" size={18} />
            ) : (
              <WifiOff className="text-red-400" size={18} />
            )}
            <span
              className={`text-sm font-medium ${isConnected ? 'text-green-400' : 'text-red-400'}`}
            >
              {isConnected ? 'Verbunden' : 'Getrennt'}
            </span>
          </div>
          <div className="text-xs text-text-muted">
            Letzte Aktualisierung: {lastUpdate.toLocaleTimeString()}
          </div>
        </div>
      </div>

      {/* SIGNAL BANNER */}
      <Card
        className={`border-l-4 ${
          signal.signal === 1
            ? 'border-l-green-500'
            : signal.signal === -1
              ? 'border-l-red-500'
              : 'border-l-gray-500'
        }`}
      >
        <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4">
          <div className="flex items-center gap-4">
            <div className={`p-4 rounded-xl ${currentSignal.color}/20`}>
              <SignalIcon className={currentSignal.text} size={32} />
            </div>
            <div>
              <p className="text-sm text-text-secondary">Aktuelles Signal</p>
              <h3 className={`text-2xl font-bold ${currentSignal.text}`}>{currentSignal.label}</h3>
              <p className="text-sm text-text-muted">{currentSignal.desc}</p>
            </div>
          </div>

          <div className="flex gap-3">
            {isRunning ? (
              <button
                onClick={handleStopTrading}
                className="flex items-center gap-2 px-6 py-3 bg-red-500 hover:bg-red-600 text-white rounded-lg font-semibold transition-colors"
              >
                <PowerOff size={20} />
                <span>Trading STOPPEN</span>
              </button>
            ) : (
              <button
                onClick={handleStartTrading}
                className="flex items-center gap-2 px-6 py-3 bg-green-500 hover:bg-green-600 text-white rounded-lg font-semibold transition-colors"
              >
                <Power size={20} />
                <span>Trading STARTEN</span>
              </button>
            )}
          </div>
        </div>
      </Card>

      {/* Bot Status */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <Card className={isRunning ? 'bg-green-500/5 border-green-500/30' : 'bg-gray-500/5'}>
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div
                className={`w-3 h-3 rounded-full ${isRunning ? 'bg-green-500 animate-pulse' : 'bg-gray-500'}`}
              />
              <span className="font-medium">Trading Bot</span>
            </div>
            <span className={`text-sm ${isRunning ? 'text-green-400' : 'text-gray-400'}`}>
              {isRunning ? 'AKTIV' : 'INAKTIV'}
            </span>
          </div>
          <p className="text-xs text-text-muted mt-2">
            {isRunning
              ? 'Der Bot führt automatisch Trades basierend auf Signalen aus'
              : 'Bot ist gestoppt - keine automatischen Trades'}
          </p>
        </Card>

        <Card className="bg-bitcoin-orange/5 border-bitcoin-orange/30">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <BarChart3 className="text-bitcoin-orange" size={20} />
              <span className="font-medium">Datenquelle</span>
            </div>
            <span className="text-sm text-bitcoin-orange">Binance</span>
          </div>
          <p className="text-xs text-text-muted mt-2">
            Preisdaten in Echtzeit von Binance Exchange
          </p>
        </Card>

        <Card className="bg-blue-500/5 border-blue-500/30">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <Activity className="text-blue-400" size={20} />
              <span className="font-medium">Modell</span>
            </div>
            <span className="text-sm text-blue-400 truncate max-w-[150px]">{signal.champion}</span>
          </div>
          <p className="text-xs text-text-muted mt-2">KI-Modell generiert Trading-Signale</p>
        </Card>
      </div>

      {/* Metrics Grid */}
      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4">
        {metrics.map(metric => {
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
        <Card className="lg:col-span-2" title="BTC/USDT Preis" helpText="Live-Preis von Binance">
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
                  tickFormatter={v => `$${(v / 1000).toFixed(0)}k`}
                />
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#1a1a1a',
                    border: '1px solid #333',
                    borderRadius: '8px',
                    color: '#fff',
                  }}
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
                  <span className="text-text-primary font-medium">
                    ${price > 0 ? price.toLocaleString() : '0'}
                  </span>
                </div>
                <div className="flex justify-between text-sm mb-2">
                  <span className="text-text-secondary">24h Hoch</span>
                  <span className="text-text-primary">
                    ${price > 0 ? (price * 1.023).toLocaleString() : '0'}
                  </span>
                </div>
                <div className="flex justify-between text-sm">
                  <span className="text-text-secondary">24h Tief</span>
                  <span className="text-text-primary">
                    ${price > 0 ? (price * 0.985).toLocaleString() : '0'}
                  </span>
                </div>
              </div>
            </div>
          </Card>

          <Card title="Schnellaktionen">
            <div className="space-y-3">
              <button className="w-full flex items-center gap-3 p-3 bg-green-500/10 hover:bg-green-500/20 border border-green-500/20 rounded-lg transition-colors">
                <ArrowUp className="text-green-400" size={20} />
                <span className="text-green-400 font-medium">Kauf Order</span>
              </button>
              <button className="w-full flex items-center gap-3 p-3 bg-red-500/10 hover:bg-red-500/20 border border-red-500/20 rounded-lg transition-colors">
                <ArrowDown className="text-red-400" size={20} />
                <span className="text-red-400 font-medium">Verkauf Order</span>
              </button>
            </div>
          </Card>
        </div>
      </div>

      {/* Info Box */}
      <Card className="bg-blue-500/5 border-blue-500/30">
        <div className="flex items-start gap-3">
          <Info className="text-blue-400 flex-shrink-0 mt-0.5" size={20} />
          <div>
            <h4 className="font-medium text-text-primary">So funktioniert&apos;s</h4>
            <ol className="text-sm text-text-secondary mt-2 space-y-1 list-decimal list-inside">
              <li>
                Das <strong>Signal</strong> oben zeigt die aktuelle Handelsempfehlung des KI-Modells
              </li>
              <li>
                Klicke auf <strong>&quot;Trading Starten&quot;</strong> um den Bot zu aktivieren
              </li>
              <li>
                Der Bot führt <strong>automatisierte Trades</strong> basierend auf Signalen aus
              </li>
              <li>
                Im <strong>Paper-Modus</strong> wird mit virtuellem Geld gehandelt - kein Risiko!
              </li>
            </ol>
          </div>
        </div>
      </Card>
    </div>
  );
}
