import { useEffect, useState } from 'react';
import { api } from '../../api/client';
import { Card } from '../ui/Card';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '../ui/table';

interface Position {
  id: string;
  symbol: string;
  side: 'LONG' | 'SHORT';
  quantity: number;
  entryPrice: number;
  currentPrice: number;
  unrealizedPnl: number;
  unrealizedPnlPercent: number;
  leverage: number;
  liquidationPrice?: number;
  openedAt: string;
}

interface PositionsTableProps {
  onClosePosition?: (id: string) => void;
}

export function PositionsTable({ onClosePosition }: PositionsTableProps) {
  const [positions, setPositions] = useState<Position[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchPositions = async () => {
      try {
        setLoading(true);
        const orders = await api.trading.getOrders();
        // Map open orders to Position objects.
        // Orders represent pending/open Binance orders; we show them as positions
        // until a dedicated /api/trading/positions endpoint is available.
        const mapped: Position[] = orders
          .filter(o => o.status === 'open')
          .map(o => ({
            id: o.id,
            symbol: o.symbol,
            side: (o.side.toUpperCase() === 'BUY' ? 'LONG' : 'SHORT') as 'LONG' | 'SHORT',
            quantity: o.quantity,
            entryPrice: o.price ?? 0,
            currentPrice: o.price ?? 0,  // live price unavailable from orders endpoint
            unrealizedPnl: 0,
            unrealizedPnlPercent: 0,
            leverage: 1,
            liquidationPrice: undefined,
            openedAt: o.created_at,
          }));
        setPositions(mapped);
        setError(null);
      } catch (e: unknown) {
        const message = e instanceof Error ? e.message : 'Failed to fetch positions';
        setError(message);
      } finally {
        setLoading(false);
      }
    };

    fetchPositions();

    // Refresh every 10 seconds
    const interval = setInterval(fetchPositions, 10000);
    return () => clearInterval(interval);
  }, []);

  if (loading && positions.length === 0) {
    return (
      <Card className="p-4">
        <h3 className="font-bold mb-4">Open Positions</h3>
        <div className="text-center text-gray-400 py-8">Loading positions...</div>
      </Card>
    );
  }

  if (error) {
    return (
      <Card className="p-4">
        <h3 className="font-bold mb-4">Open Positions</h3>
        <div className="text-center text-red-400 py-8">Error: {error}</div>
      </Card>
    );
  }

  if (positions.length === 0) {
    return (
      <Card className="p-4">
        <h3 className="font-bold mb-4">Open Positions</h3>
        <div className="text-center text-gray-400 py-8">No open positions</div>
      </Card>
    );
  }

  const totalPnl = positions.reduce((sum, p) => sum + p.unrealizedPnl, 0);
  const totalValue = positions.reduce((sum, p) => sum + p.quantity * p.currentPrice, 0);

  return (
    <Card className="p-4">
      <div className="flex items-center justify-between mb-4">
        <h3 className="font-bold">Open Positions</h3>
        <div className="flex items-center gap-4">
          <div className="text-sm">
            <span className="text-gray-400">Total Value: </span>
            <span className="font-mono">${totalValue.toLocaleString()}</span>
          </div>
          <div className="text-sm">
            <span className="text-gray-400">Unrealized P&L: </span>
            <span className={`font-mono ${totalPnl >= 0 ? 'text-green-400' : 'text-red-400'}`}>
              {totalPnl >= 0 ? '+' : ''}
              {totalPnl.toFixed(2)} USDT
            </span>
          </div>
        </div>
      </div>

      <div className="overflow-x-auto">
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Symbol</TableHead>
              <TableHead>Side</TableHead>
              <TableHead>Quantity</TableHead>
              <TableHead>Entry</TableHead>
              <TableHead>Current</TableHead>
              <TableHead>P&L</TableHead>
              <TableHead>P&L %</TableHead>
              <TableHead>Liq. Price</TableHead>
              <TableHead>Actions</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {positions.map(position => (
              <TableRow key={position.id}>
                <TableCell className="font-medium">{position.symbol}</TableCell>
                <TableCell>
                  <span
                    className={`px-2 py-1 rounded text-xs ${
                      position.side === 'LONG'
                        ? 'bg-green-500/20 text-green-400'
                        : 'bg-red-500/20 text-red-400'
                    }`}
                  >
                    {position.side}
                  </span>
                </TableCell>
                <TableCell className="font-mono">{position.quantity.toFixed(4)}</TableCell>
                <TableCell className="font-mono">${position.entryPrice.toLocaleString()}</TableCell>
                <TableCell className="font-mono">
                  ${position.currentPrice.toLocaleString()}
                </TableCell>
                <TableCell
                  className={`font-mono ${position.unrealizedPnl >= 0 ? 'text-green-400' : 'text-red-400'}`}
                >
                  {position.unrealizedPnl >= 0 ? '+' : ''}
                  {position.unrealizedPnl.toFixed(2)}
                </TableCell>
                <TableCell
                  className={`font-mono ${position.unrealizedPnlPercent >= 0 ? 'text-green-400' : 'text-red-400'}`}
                >
                  {position.unrealizedPnlPercent >= 0 ? '+' : ''}
                  {position.unrealizedPnlPercent.toFixed(2)}%
                </TableCell>
                <TableCell className="font-mono text-gray-400">
                  ${position.liquidationPrice?.toLocaleString() || '—'}
                </TableCell>
                <TableCell>
                  <button
                    onClick={() => onClosePosition?.(position.id)}
                    className="px-3 py-1 bg-red-500/20 hover:bg-red-500/40 text-red-400 rounded text-sm transition-colors"
                  >
                    Close
                  </button>
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </div>
    </Card>
  );
}

export default PositionsTable;
