import { useEffect, useState } from 'react';
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
} from 'recharts';
import { api } from '../../api/client';
import { Card } from '../ui/Card';

interface EquityPoint {
  timestamp: string;
  value: number;
}

interface EquityCurveProps {
  height?: number;
  showBenchmark?: boolean;
}

export function EquityCurve({
  height = 300,
  showBenchmark: _showBenchmark = false,
}: EquityCurveProps) {
  const [data, setData] = useState<EquityPoint[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchData = async () => {
      try {
        setLoading(true);
        const equityData = await api.analytics.getEquityCurve();
        setData(equityData);
        setError(null);
      } catch (e: unknown) {
        const message = e instanceof Error ? e.message : 'Failed to fetch equity data';
        setError(message);
        // Mock data for demo
        setData(generateMockEquityData());
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, []);

  const formatValue = (value: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USDT',
      minimumFractionDigits: 0,
      maximumFractionDigits: 0,
    }).format(value);
  };

  const formatDate = (timestamp: string) => {
    return new Date(timestamp).toLocaleDateString('en-US', {
      month: 'short',
      day: 'numeric',
    });
  };

  if (loading) {
    return (
      <Card className="p-4">
        <h3 className="font-bold mb-4">Equity Curve</h3>
        <div className="flex items-center justify-center text-gray-400" style={{ height }}>
          Loading...
        </div>
      </Card>
    );
  }

  if (error && data.length === 0) {
    return (
      <Card className="p-4">
        <h3 className="font-bold mb-4">Equity Curve</h3>
        <div className="flex items-center justify-center text-red-400" style={{ height }}>
          Error: {error}
        </div>
      </Card>
    );
  }

  const startValue = data.length > 0 ? data[0].value : 0;
  const endValue = data.length > 0 ? data[data.length - 1].value : 0;
  const totalReturn = startValue > 0 ? ((endValue - startValue) / startValue) * 100 : 0;

  return (
    <Card className="p-4">
      <div className="flex items-center justify-between mb-4">
        <h3 className="font-bold">Equity Curve</h3>
        <div className="flex items-center gap-4 text-sm">
          <div>
            <span className="text-gray-400">Current: </span>
            <span className="font-mono font-bold">{formatValue(endValue)}</span>
          </div>
          <div>
            <span className="text-gray-400">Return: </span>
            <span
              className={`font-mono font-bold ${totalReturn >= 0 ? 'text-green-400' : 'text-red-400'}`}
            >
              {totalReturn >= 0 ? '+' : ''}
              {totalReturn.toFixed(2)}%
            </span>
          </div>
        </div>
      </div>

      <ResponsiveContainer width="100%" height={height}>
        <AreaChart data={data} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
          <defs>
            <linearGradient id="equityGradient" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#10b981" stopOpacity={0.3} />
              <stop offset="95%" stopColor="#10b981" stopOpacity={0} />
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" stroke="#1a1a25" />
          <XAxis dataKey="timestamp" tickFormatter={formatDate} stroke="#71717a" fontSize={12} />
          <YAxis tickFormatter={formatValue} stroke="#71717a" fontSize={12} width={80} />
          <Tooltip
            formatter={(value: number) => [formatValue(value), 'Equity']}
            labelFormatter={formatDate}
            contentStyle={{
              backgroundColor: '#1a1a25',
              border: '1px solid #2a2a3a',
              borderRadius: '8px',
            }}
          />
          <ReferenceLine y={startValue} stroke="#71717a" strokeDasharray="3 3" />
          <Area
            type="monotone"
            dataKey="value"
            stroke="#10b981"
            strokeWidth={2}
            fill="url(#equityGradient)"
          />
        </AreaChart>
      </ResponsiveContainer>
    </Card>
  );
}

// Mock data generator for demo
function generateMockEquityData(): EquityPoint[] {
  const data: EquityPoint[] = [];
  let value = 10000;
  const startDate = new Date();
  startDate.setMonth(startDate.getMonth() - 6);

  for (let i = 0; i < 180; i++) {
    const date = new Date(startDate);
    date.setDate(date.getDate() + i);

    // Random walk with upward trend
    value = value * (1 + (Math.random() - 0.45) * 0.02);

    data.push({
      timestamp: date.toISOString(),
      value: Math.max(0, value),
    });
  }

  return data;
}

export default EquityCurve;
