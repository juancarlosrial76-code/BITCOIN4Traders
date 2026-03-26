import { useEffect, useState, memo } from 'react';
import { api } from '../../api/client';
import { Card } from '../ui/Card';

interface Signal {
  label: 'LONG' | 'FLAT' | 'SHORT';
  confidence: number;
  champion?: string;
  reason?: string;
}

export const SignalDisplay = memo(function SignalDisplay() {
  const [signal, setSignal] = useState<Signal | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchSignal = async () => {
      try {
        setLoading(true);
        const data = await api.trading.getSignal();
        setSignal({
          label: (data.champion_signal as Signal['label']) || 'FLAT',
          confidence: 75, // Backend doesn't return confidence, mock for now
          champion: data.champion_name,
        });
        setError(null);
      } catch (e: unknown) {
        const message = e instanceof Error ? e.message : 'Failed to fetch signal';
        setError(message);
      } finally {
        setLoading(false);
      }
    };

    fetchSignal();

    // Refresh every 60 seconds
    const interval = setInterval(fetchSignal, 60000);
    return () => clearInterval(interval);
  }, []);

  if (loading) {
    return <Card className="p-4 text-center text-gray-400">Loading signal...</Card>;
  }

  if (error || !signal) {
    return <Card className="p-4 text-center text-gray-400">No Signal Available</Card>;
  }

  const signalConfig = {
    LONG: {
      color: 'bg-green-500/20 border-green-500/50 text-green-400',
      icon: '▲',
      pulse: true,
    },
    SHORT: {
      color: 'bg-red-500/20 border-red-500/50 text-red-400',
      icon: '▼',
      pulse: true,
    },
    FLAT: {
      color: 'bg-gray-500/20 border-gray-500/50 text-gray-400',
      icon: '●',
      pulse: false,
    },
  };

  const config = signalConfig[signal.label];

  return (
    <Card className={`p-4 border ${config.color}`}>
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <span className={`text-2xl ${config.pulse ? 'animate-pulse' : ''}`}>{config.icon}</span>

          <div>
            <h3 className="text-lg font-bold">{signal.label}</h3>
            <p className="text-sm opacity-75">Confidence: {signal.confidence}%</p>
          </div>
        </div>

        {signal.champion && (
          <div className="text-right">
            <span className="px-3 py-1 bg-gray-700/50 rounded text-sm">{signal.champion}</span>
            <p className="text-xs text-gray-400 mt-1">Active Model</p>
          </div>
        )}
      </div>

      {signal.reason && <p className="text-xs text-gray-500 mt-2">{signal.reason}</p>}
    </Card>
  );
});
