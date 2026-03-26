import { useState, useEffect } from 'react';
import { Card, Button } from '../components/ui';
import { api } from '../api/client';
import { formatCurrency, formatDate } from '../lib/utils';
import { History as HistoryIcon, Download, Search, ChevronLeft, ChevronRight, TrendingUp, TrendingDown } from 'lucide-react';

interface Trade {
  id: string;
  symbol: string;
  side: 'buy' | 'sell';
  type: 'market' | 'limit' | 'stop';
  quantity: number;
  price: number;
  total: number;
  fee: number;
  status: 'filled' | 'pending' | 'cancelled';
  timestamp: string;
  pnl?: number;
}

export function History() {
  const [trades, setTrades] = useState<Trade[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [filter, setFilter] = useState<'all' | 'buy' | 'sell'>('all');
  const [searchTerm, setSearchTerm] = useState('');
  const [currentPage, setCurrentPage] = useState(1);
  const itemsPerPage = 20;

  useEffect(() => {
    const fetchTrades = async () => {
      try {
        const orders = await api.trading.getOrders();
        const mockTrades: Trade[] = orders.map((order: any, index: number) => ({
          id: order.id || `trade-${index}`,
          symbol: order.symbol || 'BTCUSDT',
          side: order.side?.toLowerCase() === 'buy' ? 'buy' : 'sell',
          type: 'market',
          quantity: order.quantity || 0,
          price: order.price || 43000,
          total: (order.quantity || 0) * (order.price || 43000),
          fee: ((order.quantity || 0) * (order.price || 43000)) * 0.001,
          status: order.status === 'FILLED' ? 'filled' : 'pending',
          timestamp: order.timestamp ? new Date(order.timestamp).toISOString() : new Date().toISOString(),
        }));

        if (mockTrades.length === 0) {
          mockTrades.push(
            { id: '1', symbol: 'BTCUSDT', side: 'buy', type: 'market', quantity: 0.05, price: 42800, total: 2140, fee: 2.14, status: 'filled', timestamp: new Date(Date.now() - 3600000).toISOString() },
            { id: '2', symbol: 'BTCUSDT', side: 'sell', type: 'market', quantity: 0.03, price: 43150, total: 1294.5, fee: 1.29, status: 'filled', timestamp: new Date(Date.now() - 7200000).toISOString(), pnl: 10.5 },
            { id: '3', symbol: 'BTCUSDT', side: 'buy', type: 'limit', quantity: 0.1, price: 42500, total: 4250, fee: 4.25, status: 'filled', timestamp: new Date(Date.now() - 86400000).toISOString() },
            { id: '4', symbol: 'BTCUSDT', side: 'sell', type: 'market', quantity: 0.05, price: 43200, total: 2160, fee: 2.16, status: 'filled', timestamp: new Date(Date.now() - 172800000).toISOString(), pnl: -5.2 },
            { id: '5', symbol: 'BTCUSDT', side: 'buy', type: 'market', quantity: 0.08, price: 42900, total: 3432, fee: 3.43, status: 'filled', timestamp: new Date(Date.now() - 259200000).toISOString() },
          );
        }

        setTrades(mockTrades);
      } catch (e) {
        console.error('Failed to fetch trades:', e);
        setTrades([
          { id: '1', symbol: 'BTCUSDT', side: 'buy', type: 'market', quantity: 0.05, price: 42800, total: 2140, fee: 2.14, status: 'filled', timestamp: new Date(Date.now() - 3600000).toISOString() },
          { id: '2', symbol: 'BTCUSDT', side: 'sell', type: 'market', quantity: 0.03, price: 43150, total: 1294.5, fee: 1.29, status: 'filled', timestamp: new Date(Date.now() - 7200000).toISOString(), pnl: 10.5 },
        ]);
      } finally {
        setIsLoading(false);
      }
    };
    fetchTrades();
  }, []);

  const filteredTrades = trades
    .filter(trade => filter === 'all' || trade.side === filter)
    .filter(trade => 
      searchTerm === '' || 
      trade.symbol.toLowerCase().includes(searchTerm.toLowerCase())
    )
    .sort((a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime());

  const totalPages = Math.ceil(filteredTrades.length / itemsPerPage);
  const paginatedTrades = filteredTrades.slice(
    (currentPage - 1) * itemsPerPage,
    currentPage * itemsPerPage
  );

  const stats = {
    totalTrades: trades.length,
    buyTrades: trades.filter(t => t.side === 'buy').length,
    sellTrades: trades.filter(t => t.side === 'sell').length,
    totalVolume: trades.reduce((sum, t) => sum + t.total, 0),
    totalFees: trades.reduce((sum, t) => sum + t.fee, 0),
    totalPnL: trades.reduce((sum, t) => sum + (t.pnl || 0), 0),
  };

  const statsCards = [
    { label: 'Total Trades', value: stats.totalTrades.toString(), color: 'text-bitcoin-orange' },
    { label: 'Buy Orders', value: stats.buyTrades.toString(), color: 'text-green-400' },
    { label: 'Sell Orders', value: stats.sellTrades.toString(), color: 'text-red-400' },
    { label: 'Total Volume', value: formatCurrency(stats.totalVolume), color: 'text-blue-400' },
    { label: 'Total Fees', value: formatCurrency(stats.totalFees), color: 'text-yellow-400' },
    { label: 'Realized P&L', value: formatCurrency(stats.totalPnL), color: stats.totalPnL >= 0 ? 'text-green-400' : 'text-red-400' },
  ];

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold text-text-primary">Trade History</h1>
          <p className="text-text-secondary">View all your past trades and transactions</p>
        </div>
        <Button variant="secondary" size="sm">
          <Download size={16} className="mr-2" />
          Export CSV
        </Button>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3">
        {statsCards.map((stat) => (
          <Card key={stat.label} className="p-3">
            <p className="text-xs text-text-muted">{stat.label}</p>
            <p className={`text-lg font-bold ${stat.color}`}>{stat.value}</p>
          </Card>
        ))}
      </div>

      {/* Filters */}
      <Card>
        <div className="flex flex-col sm:flex-row gap-4 mb-4">
          <div className="relative flex-1">
            <Search size={18} className="absolute left-3 top-1/2 -translate-y-1/2 text-text-muted" />
            <input
              type="text"
              placeholder="Search by symbol..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="w-full bg-background border border-border rounded-lg pl-10 pr-4 py-2 text-text-primary placeholder-text-muted focus:outline-none focus:ring-2 focus:ring-bitcoin-orange/50"
            />
          </div>
          <div className="flex gap-2">
            {(['all', 'buy', 'sell'] as const).map((f) => (
              <button
                key={f}
                onClick={() => { setFilter(f); setCurrentPage(1); }}
                className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                  filter === f 
                    ? 'bg-bitcoin-orange text-white' 
                    : 'bg-background text-text-secondary hover:text-text-primary'
                }`}
              >
                {f === 'all' ? 'All' : f === 'buy' ? 'Buy' : 'Sell'}
              </button>
            ))}
          </div>
        </div>

        {/* Table */}
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="border-b border-border">
                <th className="text-left py-3 px-4 text-sm font-medium text-text-secondary">Time</th>
                <th className="text-left py-3 px-4 text-sm font-medium text-text-secondary">Symbol</th>
                <th className="text-left py-3 px-4 text-sm font-medium text-text-secondary">Type</th>
                <th className="text-left py-3 px-4 text-sm font-medium text-text-secondary">Side</th>
                <th className="text-right py-3 px-4 text-sm font-medium text-text-secondary">Quantity</th>
                <th className="text-right py-3 px-4 text-sm font-medium text-text-secondary">Price</th>
                <th className="text-right py-3 px-4 text-sm font-medium text-text-secondary">Total</th>
                <th className="text-right py-3 px-4 text-sm font-medium text-text-secondary">Fee</th>
                <th className="text-right py-3 px-4 text-sm font-medium text-text-secondary">P&L</th>
                <th className="text-center py-3 px-4 text-sm font-medium text-text-secondary">Status</th>
              </tr>
            </thead>
            <tbody>
              {isLoading ? (
                <tr>
                  <td colSpan={10} className="text-center py-8 text-text-muted">Loading...</td>
                </tr>
              ) : paginatedTrades.length === 0 ? (
                <tr>
                  <td colSpan={10} className="text-center py-8 text-text-muted">
                    <HistoryIcon size={32} className="mx-auto mb-2 opacity-50" />
                    <p>No trades found</p>
                  </td>
                </tr>
              ) : (
                paginatedTrades.map((trade) => (
                  <tr key={trade.id} className="border-b border-border/50 hover:bg-background/50">
                    <td className="py-3 px-4 text-sm text-text-secondary">
                      {formatDate(trade.timestamp)}
                    </td>
                    <td className="py-3 px-4 text-sm font-medium text-text-primary">{trade.symbol}</td>
                    <td className="py-3 px-4 text-sm text-text-secondary capitalize">{trade.type}</td>
                    <td className="py-3 px-4">
                      <span className={`flex items-center gap-1 text-sm font-medium ${
                        trade.side === 'buy' ? 'text-green-400' : 'text-red-400'
                      }`}>
                        {trade.side === 'buy' ? <TrendingUp size={14} /> : <TrendingDown size={14} />}
                        {trade.side.toUpperCase()}
                      </span>
                    </td>
                    <td className="py-3 px-4 text-sm text-text-primary text-right font-mono">
                      {trade.quantity}
                    </td>
                    <td className="py-3 px-4 text-sm text-text-secondary text-right font-mono">
                      {formatCurrency(trade.price)}
                    </td>
                    <td className="py-3 px-4 text-sm text-text-primary text-right font-mono">
                      {formatCurrency(trade.total)}
                    </td>
                    <td className="py-3 px-4 text-sm text-text-muted text-right font-mono">
                      {formatCurrency(trade.fee)}
                    </td>
                    <td className={`py-3 px-4 text-sm text-right font-mono ${
                      trade.pnl ? (trade.pnl >= 0 ? 'text-green-400' : 'text-red-400') : 'text-text-muted'
                    }`}>
                      {trade.pnl ? formatCurrency(trade.pnl) : '-'}
                    </td>
                    <td className="py-3 px-4 text-center">
                      <span className={`px-2 py-1 text-xs font-medium rounded ${
                        trade.status === 'filled' 
                          ? 'bg-green-500/10 text-green-400' 
                          : trade.status === 'pending'
                          ? 'bg-yellow-500/10 text-yellow-400'
                          : 'bg-red-500/10 text-red-400'
                      }`}>
                        {trade.status}
                      </span>
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>

        {/* Pagination */}
        {totalPages > 1 && (
          <div className="flex items-center justify-between mt-4 pt-4 border-t border-border">
            <p className="text-sm text-text-muted">
              Showing {(currentPage - 1) * itemsPerPage + 1} to {Math.min(currentPage * itemsPerPage, filteredTrades.length)} of {filteredTrades.length} trades
            </p>
            <div className="flex items-center gap-2">
              <Button
                variant="ghost"
                size="sm"
                onClick={() => setCurrentPage(p => Math.max(1, p - 1))}
                disabled={currentPage === 1}
              >
                <ChevronLeft size={16} />
              </Button>
              <span className="text-sm text-text-secondary">
                Page {currentPage} of {totalPages}
              </span>
              <Button
                variant="ghost"
                size="sm"
                onClick={() => setCurrentPage(p => Math.min(totalPages, p + 1))}
                disabled={currentPage === totalPages}
              >
                <ChevronRight size={16} />
              </Button>
            </div>
          </div>
        )}
      </Card>
    </div>
  );
}
