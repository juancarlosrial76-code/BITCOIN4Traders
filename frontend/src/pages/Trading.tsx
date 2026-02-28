import { useState, useEffect } from 'react';
import { Card, Button, Input, Select } from '../components/ui';
import { useTradingStore } from '../stores';
import { api } from '../api/client';
import { Play, Square, Plus, Minus, Settings } from 'lucide-react';

export function Trading() {
  const { isRunning, config, startTrading, stopTrading } = useTradingStore();
  const [orderType, setOrderType] = useState('market');
  const [orderSide, setOrderSide] = useState('buy');
  const [amount, setAmount] = useState('0.01');
  const [price, setPrice] = useState('');
  const [currentPosition, setCurrentPosition] = useState('0.05 BTC');
  const [unrealizedPnL, setUnrealizedPnL] = useState('+$124.50');
  const [activeOrders, setActiveOrders] = useState<any[]>([]);
  const [isLoading, setIsLoading] = useState(false);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const status = await api.trading.getStatus();
        const orders = await api.trading.getOrders();
        setActiveOrders(orders);
        setCurrentPosition(`${status.current_position} BTC`);
        setUnrealizedPnL(`$${status.unrealized_pnl.toFixed(2)}`);
      } catch (e) {
        console.error('Failed to fetch trading data:', e);
      }
    };
    fetchData();
    const interval = setInterval(fetchData, 5000);
    return () => clearInterval(interval);
  }, []);

  const handleToggleBot = async () => {
    setIsLoading(true);
    try {
      if (isRunning) {
        await stopTrading();
      } else {
        await startTrading();
      }
    } catch (e) {
      console.error('Failed to toggle bot:', e);
    }
    setIsLoading(false);
  };

  const handlePlaceOrder = async () => {
    setIsLoading(true);
    try {
      await api.trading.placeOrder({
        symbol: 'BTCUSDT',
        side: orderSide.toUpperCase(),
        order_type: orderType,
        quantity: parseFloat(amount),
        price: price ? parseFloat(price) : undefined,
      });
      const orders = await api.trading.getOrders();
      setActiveOrders(orders);
    } catch (e) {
      console.error('Failed to place order:', e);
    }
    setIsLoading(false);
  };

  return (
    <div className="space-y-6">
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold text-text-primary">Trading</h1>
          <p className="text-text-secondary">Control your trading bot and place orders</p>
        </div>
        <div className="flex items-center gap-3">
          <Button
            variant={isRunning ? 'danger' : 'primary'}
            onClick={handleToggleBot}
            disabled={isLoading}
          >
            {isRunning ? (
              <>
                <Square size={18} />
                Stop Bot
              </>
            ) : (
              <>
                <Play size={18} />
                Start Bot
              </>
            )}
          </Button>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <Card className="lg:col-span-2" title="Place Order">
          <div className="space-y-4">
            <div className="flex gap-2">
              <button
                onClick={() => setOrderSide('buy')}
                className={`flex-1 py-3 rounded-lg font-medium transition-colors ${
                  orderSide === 'buy'
                    ? 'bg-green-500/20 text-green-400 border border-green-500/30'
                    : 'bg-background text-text-secondary hover:text-text-primary'
                }`}
              >
                <Plus size={18} className="inline mr-2" />
                Buy
              </button>
              <button
                onClick={() => setOrderSide('sell')}
                className={`flex-1 py-3 rounded-lg font-medium transition-colors ${
                  orderSide === 'sell'
                    ? 'bg-red-500/20 text-red-400 border border-red-500/30'
                    : 'bg-background text-text-secondary hover:text-text-primary'
                }`}
              >
                <Minus size={18} className="inline mr-2" />
                Sell
              </button>
            </div>

            <Select
              label="Order Type"
              value={orderType}
              onChange={(e) => setOrderType(e.target.value)}
              options={[
                { value: 'market', label: 'Market Order' },
                { value: 'limit', label: 'Limit Order' },
                { value: 'stop', label: 'Stop Loss' },
              ]}
            />

            {orderType !== 'market' && (
              <Input
                label="Price (USDT)"
                type="number"
                value={price}
                onChange={(e) => setPrice(e.target.value)}
                placeholder="0.00"
              />
            )}

            <Input
              label="Amount (BTC)"
              type="number"
              value={amount}
              onChange={(e) => setAmount(e.target.value)}
              placeholder="0.00"
            />

            <div className="flex gap-2 pt-2">
              <Button variant="secondary" className="flex-1" onClick={() => setAmount(String(parseFloat(amount) * 0.25))}>
                25%
              </Button>
              <Button variant="secondary" className="flex-1" onClick={() => setAmount(String(parseFloat(amount) * 0.5))}>
                50%
              </Button>
              <Button variant="secondary" className="flex-1" onClick={() => setAmount(String(parseFloat(amount) * 0.75))}>
                75%
              </Button>
              <Button variant="secondary" className="flex-1" onClick={() => setAmount(String(parseFloat(amount) || 0))}>
                100%
              </Button>
            </div>

            <div className="pt-4 border-t border-border">
              <div className="flex justify-between mb-2">
                <span className="text-text-secondary">Estimated Total</span>
                <span className="text-text-primary font-medium">$0.00</span>
              </div>
              <div className="flex justify-between mb-2">
                <span className="text-text-secondary">Fee (0.1%)</span>
                <span className="text-text-primary">$0.00</span>
              </div>
            </div>

            <Button
              variant={orderSide === 'buy' ? 'primary' : 'danger'}
              className="w-full"
              onClick={handlePlaceOrder}
              disabled={isLoading}
            >
              {orderSide === 'buy' ? 'Buy BTC' : 'Sell BTC'}
            </Button>
          </div>
        </Card>

        <div className="space-y-6">
          <Card title="Bot Status">
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <span className="text-text-secondary">Status</span>
                <div className="flex items-center gap-2">
                  <div className={`w-2 h-2 rounded-full ${isRunning ? 'bg-green-500 animate-pulse' : 'bg-gray-500'}`}></div>
                  <span className={isRunning ? 'text-green-400' : 'text-gray-400'}>
                    {isRunning ? 'Running' : 'Stopped'}
                  </span>
                </div>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-text-secondary">Current Position</span>
                <span className="text-text-primary">{currentPosition}</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-text-secondary">Unrealized P&L</span>
                <span className="text-green-400">{unrealizedPnL}</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-text-secondary">Leverage</span>
                <span className="text-text-primary">{config.leverage}x</span>
              </div>
            </div>
          </Card>

          <Card title="Trading Parameters">
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <span className="text-text-secondary">Max Position</span>
                <span className="text-text-primary">{config.maxPositionSize * 100}%</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-text-secondary">Stop Loss</span>
                <span className="text-red-400">-{config.stopLoss * 100}%</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-text-secondary">Take Profit</span>
                <span className="text-green-400">+{config.takeProfit * 100}%</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-text-secondary">Risk/Trade</span>
                <span className="text-text-primary">{config.riskPerTrade * 100}%</span>
              </div>
              <Button variant="ghost" size="sm" className="w-full mt-2">
                <Settings size={16} className="mr-2" />
                Configure
              </Button>
            </div>
          </Card>

          <Card title="Active Orders">
            <div className="space-y-3">
              {activeOrders.length > 0 ? activeOrders.slice(0, 3).map((order, i) => (
                <div key={i} className="p-3 bg-background rounded-lg">
                  <div className="flex items-center justify-between mb-1">
                    <span className={`text-sm font-medium ${order.side === 'BUY' ? 'text-green-400' : 'text-red-400'}`}>
                      {order.side}
                    </span>
                    <span className="text-text-secondary text-xs">{order.status}</span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-text-primary">{order.quantity} BTC @ {order.price}</span>
                  </div>
                </div>
              )) : (
                <p className="text-center text-text-muted text-sm py-4">No active orders</p>
              )}
            </div>
          </Card>
        </div>
      </div>
    </div>
  );
}
