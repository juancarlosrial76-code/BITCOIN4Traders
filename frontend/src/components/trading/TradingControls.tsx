import { useState, memo } from 'react';
import { useTradingStore } from '../../stores/tradingStore';
import { api } from '../../api/client';
import { Button } from '../ui/Button';
import { Card } from '../ui/Card';

export const TradingControls = memo(function TradingControls() {
  const { isRunning, mode, setIsRunning, setMode, error, setError, fetchStatus } =
    useTradingStore();

  const [actionLoading, setActionLoading] = useState(false);

  const handleStart = async (selectedMode: 'live' | 'paper') => {
    setActionLoading(true);
    setError(null);

    try {
      await api.trading.start();
      setIsRunning(true);
      setMode(selectedMode);
      await fetchStatus();
    } catch (e: unknown) {
      const message = e instanceof Error ? e.message : 'Start failed';
      setError(message);
    } finally {
      setActionLoading(false);
    }
  };

  const handleStop = async () => {
    setActionLoading(true);
    setError(null);

    try {
      await api.trading.stop();
      setIsRunning(false);
      await fetchStatus();
    } catch (e: unknown) {
      const message = e instanceof Error ? e.message : 'Stop failed';
      setError(message);
    } finally {
      setActionLoading(false);
    }
  };

  return (
    <Card className="p-4">
      <div className="flex items-center justify-between">
        {/* Status */}
        <div className="flex items-center gap-4">
          <div
            className={`w-3 h-3 rounded-full ${
              isRunning ? 'bg-green-500 animate-pulse' : 'bg-gray-500'
            }`}
          />

          <div>
            <p className="font-medium">{isRunning ? 'Trading Active' : 'Trading Stopped'}</p>
            <p className="text-sm text-gray-400">Mode: {(mode || 'paper').toUpperCase()}</p>
          </div>
        </div>

        {/* Controls */}
        <div className="flex gap-2">
          {!isRunning ? (
            <>
              <Button
                variant="secondary"
                onClick={() => handleStart('paper')}
                disabled={actionLoading}
              >
                Paper Trade
              </Button>
              <Button
                variant="primary"
                onClick={() => handleStart('live')}
                disabled={actionLoading}
                className="bg-red-500 hover:bg-red-600"
              >
                Go Live
              </Button>
            </>
          ) : (
            <Button variant="danger" onClick={handleStop} disabled={actionLoading}>
              Stop Trading
            </Button>
          )}
        </div>
      </div>

      {error && <div className="mt-3 p-2 bg-red-500/20 text-red-400 rounded text-sm">{error}</div>}
    </Card>
  );
});

export default TradingControls;
