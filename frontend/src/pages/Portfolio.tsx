import { useState, useEffect } from 'react';
import { Card, Button } from '../components/ui';
import { api } from '../api/client';
import { formatCurrency } from '../lib/utils';
import { Wallet, TrendingUp, TrendingDown, PieChart, DollarSign, RefreshCw, ExternalLink } from 'lucide-react';

interface Position {
  symbol: string;
  side: 'long' | 'short';
  quantity: number;
  entryPrice: number;
  currentPrice: number;
  pnl: number;
  pnlPercent: number;
}

interface PortfolioSummary {
  totalValue: number;
  totalPnL: number;
  totalPnLPercent: number;
  availableBalance: number;
  positions: Position[];
}

export function Portfolio() {
  const [portfolio, setPortfolio] = useState<PortfolioSummary | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);

  const fetchPortfolio = async () => {
    try {
      setRefreshing(true);
      const [status, balance] = await Promise.all([
        api.trading.getStatus(),
        api.trading.getBalance(),
      ]);

      const totalValue = balance.balance.USDT + (balance.balance.BTC * (status.current_position || 43000));
      const availableBalance = balance.balance.USDT;
      
      const positions: Position[] = status.current_position > 0 ? [{
        symbol: 'BTC/USDT',
        side: status.current_position > 0 ? 'long' : 'short',
        quantity: Math.abs(status.current_position),
        entryPrice: 43000,
        currentPrice: 43000,
        pnl: status.unrealized_pnl || 0,
        pnlPercent: ((status.unrealized_pnl || 0) / (status.current_position * 43000)) * 100,
      }] : [];

      setPortfolio({
        totalValue,
        totalPnL: status.unrealized_pnl || 0,
        totalPnLPercent: ((status.unrealized_pnl || 0) / totalValue) * 100,
        availableBalance,
        positions,
      });
    } catch (e) {
      console.error('Failed to fetch portfolio:', e);
    } finally {
      setIsLoading(false);
      setRefreshing(false);
    }
  };

  useEffect(() => {
    fetchPortfolio();
    const interval = setInterval(fetchPortfolio, 10000);
    return () => clearInterval(interval);
  }, []);

  const summaryCards = [
    { 
      label: 'Total Value', 
      value: portfolio ? formatCurrency(portfolio.totalValue) : '$0.00', 
      icon: Wallet,
      color: 'text-bitcoin-orange'
    },
    { 
      label: 'Available Balance', 
      value: portfolio ? formatCurrency(portfolio.availableBalance) : '$0.00', 
      icon: DollarSign,
      color: 'text-blue-400'
    },
    { 
      label: 'Unrealized P&L', 
      value: portfolio ? formatCurrency(portfolio.totalPnL) : '$0.00', 
      icon: portfolio && portfolio.totalPnL >= 0 ? TrendingUp : TrendingDown,
      color: portfolio && portfolio.totalPnL >= 0 ? 'text-green-400' : 'text-red-400'
    },
    { 
      label: 'P&L %', 
      value: portfolio ? `${portfolio.totalPnLPercent.toFixed(2)}%` : '0%', 
      icon: PieChart,
      color: portfolio && portfolio.totalPnLPercent >= 0 ? 'text-green-400' : 'text-red-400'
    },
  ];

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold text-text-primary">Portfolio</h1>
          <p className="text-text-secondary">Manage your positions and balances</p>
        </div>
        <div className="flex items-center gap-3">
          <Button 
            variant="secondary" 
            size="sm" 
            onClick={fetchPortfolio}
            disabled={refreshing}
          >
            <RefreshCw size={16} className={`mr-2 ${refreshing ? 'animate-spin' : ''}`} />
            Refresh
          </Button>
          <Button 
            variant="secondary" 
            size="sm"
            onClick={() => window.open('https://www.binance.com/en/balance', '_blank')}
          >
            <ExternalLink size={16} className="mr-2" />
            Exchange
          </Button>
        </div>
      </div>

      {/* Summary Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        {summaryCards.map((card) => {
          const Icon = card.icon;
          return (
            <Card key={card.label} className="hover:border-bitcoin-orange/30 transition-colors">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-text-secondary">{card.label}</p>
                  <p className={`text-2xl font-bold mt-1 ${card.color}`}>{card.value}</p>
                </div>
                <div className={`p-3 bg-bitcoin-orange/10 rounded-lg`}>
                  <Icon className={card.color} size={24} />
                </div>
              </div>
            </Card>
          );
        })}
      </div>

      {/* Positions */}
      <Card title="Open Positions" helpText="Your current trading positions">
        {isLoading ? (
          <div className="animate-pulse space-y-3">
            {[1, 2, 3].map((i) => (
              <div key={i} className="h-20 bg-background rounded-lg" />
            ))}
          </div>
        ) : portfolio && portfolio.positions.length > 0 ? (
          <div className="space-y-3">
            {portfolio.positions.map((position, index) => (
              <div 
                key={index}
                className="flex items-center justify-between p-4 bg-background rounded-lg hover:bg-background/80 transition-colors"
              >
                <div className="flex items-center gap-4">
                  <div className={`w-10 h-10 rounded-lg flex items-center justify-center ${
                    position.side === 'long' ? 'bg-green-500/10' : 'bg-red-500/10'
                  }`}>
                    {position.side === 'long' ? (
                      <TrendingUp className="text-green-400" size={20} />
                    ) : (
                      <TrendingDown className="text-red-400" size={20} />
                    )}
                  </div>
                  <div>
                    <p className="font-medium text-text-primary">{position.symbol}</p>
                    <p className="text-sm text-text-secondary">
                      {position.quantity} BTC @ {formatCurrency(position.entryPrice)}
                    </p>
                  </div>
                </div>
                <div className="text-right">
                  <p className="font-medium text-text-primary">{formatCurrency(position.currentPrice * position.quantity)}</p>
                  <p className={`text-sm ${position.pnl >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                    {position.pnl >= 0 ? '+' : ''}{formatCurrency(position.pnl)} ({position.pnlPercent.toFixed(2)}%)
                  </p>
                </div>
              </div>
            ))}
          </div>
        ) : (
          <div className="text-center py-12">
            <Wallet size={48} className="mx-auto text-text-muted mb-4" />
            <p className="text-text-secondary">No open positions</p>
            <p className="text-sm text-text-muted mt-1">Start trading to open positions</p>
          </div>
        )}
      </Card>

      {/* Asset Allocation */}
      <Card title="Asset Allocation" helpText="Distribution of your portfolio">
        <div className="space-y-4">
          {[
            { name: 'USDT', value: portfolio?.availableBalance || 0, percent: ((portfolio?.availableBalance || 0) / (portfolio?.totalValue || 1)) * 100, color: 'bg-green-500' },
            { name: 'BTC', value: (portfolio?.positions[0]?.quantity || 0) * 43000, percent: ((portfolio?.positions[0]?.quantity || 0) * 43000 / (portfolio?.totalValue || 1)) * 100, color: 'bg-bitcoin-orange' },
          ].map((asset) => (
            <div key={asset.name} className="space-y-2">
              <div className="flex justify-between text-sm">
                <span className="text-text-secondary">{asset.name}</span>
                <span className="text-text-primary font-medium">{formatCurrency(asset.value)} ({asset.percent.toFixed(1)}%)</span>
              </div>
              <div className="h-2 bg-background rounded-full overflow-hidden">
                <div 
                  className={`h-full ${asset.color} transition-all duration-500`} 
                  style={{ width: `${asset.percent}%` }}
                />
              </div>
            </div>
          ))}
        </div>
      </Card>
    </div>
  );
}
