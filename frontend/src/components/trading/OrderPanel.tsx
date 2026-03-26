import { useState, useMemo } from 'react';
import { api } from '../../api/client';
import { useTradingStore } from '../../stores/tradingStore';
import { Button } from '../ui/Button';
import { Input } from '../ui/Input';
import { Card } from '../ui/Card';

export function OrderPanel() {
  const { currentPrice, selectedSymbol, fetchOrders, fetchBalance } = useTradingStore();

  const [side, setSide] = useState<'buy' | 'sell'>('buy');
  const [orderType, setOrderType] = useState<'market' | 'limit'>('market');
  const [quantity, setQuantity] = useState('');
  const [price, setPrice] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);

  const balance = useMemo(() => {
    // Mock balance - in real app, get from store
    return { total: 10000, available: 10000 };
  }, []);

  const estimatedTotal = useMemo(() => {
    const qty = parseFloat(quantity) || 0;
    const prc = orderType === 'market' ? currentPrice : parseFloat(price) || 0;
    return qty * prc;
  }, [quantity, price, orderType, currentPrice]);

  const validationError = useMemo(() => {
    const qty = parseFloat(quantity);
    if (!qty || qty <= 0) return 'Invalid quantity';
    if (qty > 100) return 'Max quantity is 100';
    if (estimatedTotal > balance.available) return 'Insufficient balance';
    if (orderType === 'limit' && (!price || parseFloat(price) <= 0)) return 'Invalid limit price';
    return null;
  }, [quantity, price, orderType, estimatedTotal, balance.available]);

  const handleSubmit = async () => {
    if (validationError) return;

    setIsLoading(true);
    setError(null);
    setSuccess(null);

    try {
      await api.trading.placeOrder({
        symbol: selectedSymbol,
        side,
        order_type: orderType,
        quantity: parseFloat(quantity),
        price: orderType === 'limit' ? parseFloat(price) : undefined,
      });

      // Reset
      setQuantity('');
      setPrice('');
      setSuccess('Order placed successfully!');

      // Refresh data
      await Promise.all([fetchOrders(), fetchBalance()]);

      setTimeout(() => setSuccess(null), 3000);
    } catch (e: any) {
      setError(e.message || 'Order failed');
    } finally {
      setIsLoading(false);
    }
  };

  const setMaxQuantity = () => {
    if (currentPrice > 0) {
      const max = balance.available / currentPrice;
      setQuantity(Math.min(max, 100).toFixed(6));
    }
  };

  return (
    <Card className="p-4">
      <h3 className="font-bold mb-4">Place Order</h3>

      {/* Side Selection */}
      <div className="grid grid-cols-2 gap-2 mb-4">
        <Button
          variant={side === 'buy' ? 'default' : 'outline'}
          onClick={() => setSide('buy')}
          className={side === 'buy' ? 'bg-green-500 hover:bg-green-600' : ''}
        >
          BUY
        </Button>
        <Button
          variant={side === 'sell' ? 'default' : 'outline'}
          onClick={() => setSide('sell')}
          className={side === 'sell' ? 'bg-red-500 hover:bg-red-600' : ''}
        >
          SELL
        </Button>
      </div>

      {/* Order Type */}
      <div className="mb-4">
        <label className="block text-sm text-gray-400 mb-1">Order Type</label>
        <select
          value={orderType}
          onChange={e => setOrderType(e.target.value as any)}
          className="w-full p-2 bg-gray-800 border border-gray-700 rounded-lg text-white"
        >
          <option value="market">Market Order</option>
          <option value="limit">Limit Order</option>
        </select>
      </div>

      {/* Price (for limit orders) */}
      {orderType === 'limit' && (
        <div className="mb-4">
          <label className="block text-sm text-gray-400 mb-1">Limit Price</label>
          <Input
            type="number"
            value={price}
            onChange={e => setPrice(e.target.value)}
            placeholder={currentPrice.toString()}
          />
        </div>
      )}

      {/* Quantity */}
      <div className="mb-4">
        <label className="block text-sm text-gray-400 mb-1">Quantity</label>
        <div className="flex gap-2">
          <Input
            type="number"
            value={quantity}
            onChange={e => setQuantity(e.target.value)}
            placeholder="0.00"
            className="flex-1"
          />
          <Button variant="outline" onClick={setMaxQuantity}>
            MAX
          </Button>
        </div>
      </div>

      {/* Summary */}
      <div className="bg-gray-800/50 rounded p-3 mb-4 text-sm">
        <div className="flex justify-between">
          <span className="text-gray-400">Current Price</span>
          <span>${currentPrice.toLocaleString()}</span>
        </div>
        <div className="flex justify-between">
          <span className="text-gray-400">Est. Total</span>
          <span>${estimatedTotal.toLocaleString(undefined, { minimumFractionDigits: 2 })}</span>
        </div>
        <div className="flex justify-between">
          <span className="text-gray-400">Available</span>
          <span>${balance.available.toLocaleString(undefined, { minimumFractionDigits: 2 })}</span>
        </div>
      </div>

      {/* Error */}
      {(error || validationError) && (
        <div className="mb-4 p-2 bg-red-500/20 text-red-400 rounded text-sm">
          {error || validationError}
        </div>
      )}

      {/* Success */}
      {success && (
        <div className="mb-4 p-2 bg-green-500/20 text-green-400 rounded text-sm">{success}</div>
      )}

      {/* Submit */}
      <Button
        className={`w-full ${
          side === 'buy' ? 'bg-green-500 hover:bg-green-600' : 'bg-red-500 hover:bg-red-600'
        }`}
        onClick={handleSubmit}
        disabled={!!validationError || isLoading}
      >
        {isLoading ? 'Processing...' : `${side.toUpperCase()} ${selectedSymbol}`}
      </Button>
    </Card>
  );
}

export default OrderPanel;
