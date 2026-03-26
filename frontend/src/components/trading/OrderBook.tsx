import { useState, useEffect } from 'react';
import { Card } from '../ui/Card';

interface OrderBookEntry {
  price: number;
  quantity: number;
  total: number;
}

interface OrderBookProps {
  symbol?: string;
  depth?: number;
}

export function OrderBook({ symbol = 'BTCUSDT', depth = 10 }: OrderBookProps) {
  const [bids, setBids] = useState<OrderBookEntry[]>([]);
  const [asks, setAsks] = useState<OrderBookEntry[]>([]);
  const [loading, setLoading] = useState(true);

  // Mock data for demonstration - in production, this would come from WebSocket or API
  useEffect(() => {
    const generateMockOrderBook = () => {
      const basePrice = 67000;

      // Generate bids (buy orders)
      const newBids: OrderBookEntry[] = [];
      let bidTotal = 0;
      for (let i = 0; i < depth; i++) {
        const price = basePrice - i * 5;
        const quantity = Math.random() * 2 + 0.1;
        bidTotal += quantity * price;
        newBids.push({ price, quantity, total: bidTotal });
      }
      setBids(newBids);

      // Generate asks (sell orders)
      const newAsks: OrderBookEntry[] = [];
      let askTotal = 0;
      for (let i = 0; i < depth; i++) {
        const price = basePrice + i * 5 + 5;
        const quantity = Math.random() * 2 + 0.1;
        askTotal += quantity * price;
        newAsks.push({ price, quantity, total: askTotal });
      }
      setAsks(newAsks);
      setLoading(false);
    };

    generateMockOrderBook();

    // Refresh every 2 seconds
    const interval = setInterval(generateMockOrderBook, 2000);
    return () => clearInterval(interval);
  }, [symbol, depth]);

  const maxTotal = Math.max(
    bids.length > 0 ? bids[bids.length - 1].total : 0,
    asks.length > 0 ? asks[asks.length - 1].total : 0
  );

  if (loading) {
    return (
      <Card className="p-4">
        <h3 className="font-bold mb-4">Order Book</h3>
        <div className="text-center text-gray-400 py-8">Loading...</div>
      </Card>
    );
  }

  return (
    <Card className="p-4">
      <h3 className="font-bold mb-4">Order Book - {symbol}</h3>

      {/* Header */}
      <div className="grid grid-cols-3 gap-2 text-xs text-gray-400 mb-2 px-2">
        <div>Price (USDT)</div>
        <div className="text-right">Amount</div>
        <div className="text-right">Total</div>
      </div>

      {/* Asks (Sell Orders) - Red */}
      <div className="space-y-px mb-2">
        {[...asks].reverse().map((ask, index) => (
          <div key={`ask-${index}`} className="relative">
            <div
              className="absolute right-0 top-0 bottom-0 bg-red-500/20"
              style={{ width: `${(ask.total / maxTotal) * 100}%` }}
            />
            <div className="relative grid grid-cols-3 gap-2 text-xs px-2 py-1">
              <div className="text-red-400 font-mono">{ask.price.toLocaleString()}</div>
              <div className="text-right font-mono">{ask.quantity.toFixed(4)}</div>
              <div className="text-right font-mono text-gray-400">{ask.total.toFixed(2)}</div>
            </div>
          </div>
        ))}
      </div>

      {/* Spread */}
      <div className="py-2 text-center text-sm border-y border-gray-700 my-2">
        <span className="text-gray-400">Spread: </span>
        <span className="font-mono">
          {asks.length > 0 && bids.length > 0 ? (asks[0].price - bids[0].price).toFixed(2) : '—'}{' '}
          USDT
        </span>
        <span className="text-gray-400 ml-2">
          (
          {asks.length > 0 && bids.length > 0
            ? (((asks[0].price - bids[0].price) / asks[0].price) * 100).toFixed(3)
            : '0'}
          %)
        </span>
      </div>

      {/* Bids (Buy Orders) - Green */}
      <div className="space-y-px">
        {bids.map((bid, index) => (
          <div key={`bid-${index}`} className="relative">
            <div
              className="absolute right-0 top-0 bottom-0 bg-green-500/20"
              style={{ width: `${(bid.total / maxTotal) * 100}%` }}
            />
            <div className="relative grid grid-cols-3 gap-2 text-xs px-2 py-1">
              <div className="text-green-400 font-mono">{bid.price.toLocaleString()}</div>
              <div className="text-right font-mono">{bid.quantity.toFixed(4)}</div>
              <div className="text-right font-mono text-gray-400">{bid.total.toFixed(2)}</div>
            </div>
          </div>
        ))}
      </div>
    </Card>
  );
}

export default OrderBook;
