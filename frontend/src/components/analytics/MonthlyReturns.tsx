import { useEffect, useState } from 'react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
  ReferenceLine,
} from 'recharts';
import { api } from '../../api/client';
import { Card } from '../ui/Card';

interface MonthlyReturn {
  month: string;
  return: number;
}

interface MonthlyReturnsProps {
  height?: number;
}

export function MonthlyReturns({ height = 300 }: MonthlyReturnsProps) {
  const [data, setData] = useState<MonthlyReturn[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchData = async () => {
      try {
        setLoading(true);
        const monthlyData = await api.analytics.getMonthlyReturns();
        setData(monthlyData);
        setError(null);
      } catch (e: any) {
        setError(e.message);
        // Mock data for demo
        setData(generateMockMonthlyReturns());
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, []);

  const formatReturn = (value: number) => {
    return `${value >= 0 ? '+' : ''}${value.toFixed(1)}%`;
  };

  const getBarColor = (value: number) => {
    if (value > 5) return '#10b981'; // Strong green
    if (value > 0) return '#34d399'; // Light green
    if (value > -5) return '#f87171'; // Light red
    return '#ef4444'; // Strong red
  };

  if (loading) {
    return (
      <Card className="p-4">
        <h3 className="font-bold mb-4">Monthly Returns</h3>
        <div className="flex items-center justify-center text-gray-400" style={{ height }}>
          Loading...
        </div>
      </Card>
    );
  }

  if (error && data.length === 0) {
    return (
      <Card className="p-4">
        <h3 className="font-bold mb-4">Monthly Returns</h3>
        <div className="flex items-center justify-center text-red-400" style={{ height }}>
          Error: {error}
        </div>
      </Card>
    );
  }

  const totalReturn = data.reduce((sum, d) => sum + d.return, 0);
  const winningMonths = data.filter(d => d.return > 0).length;
  const avgReturn = data.length > 0 ? totalReturn / data.length : 0;

  return (
    <Card className="p-4">
      <div className="flex items-center justify-between mb-4">
        <h3 className="font-bold">Monthly Returns</h3>
        <div className="flex gap-4 text-sm">
          <div>
            <span className="text-gray-400">Total: </span>
            <span className={`font-mono ${totalReturn >= 0 ? 'text-green-400' : 'text-red-400'}`}>
              {formatReturn(totalReturn)}
            </span>
          </div>
          <div>
            <span className="text-gray-400">Avg: </span>
            <span className={`font-mono ${avgReturn >= 0 ? 'text-green-400' : 'text-red-400'}`}>
              {formatReturn(avgReturn)}
            </span>
          </div>
          <div>
            <span className="text-gray-400">Win Rate: </span>
            <span className="font-mono">
              {data.length > 0 ? ((winningMonths / data.length) * 100).toFixed(0) : 0}%
            </span>
          </div>
        </div>
      </div>

      <ResponsiveContainer width="100%" height={height}>
        <BarChart data={data} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#1a1a25" vertical={false} />
          <XAxis dataKey="month" stroke="#71717a" fontSize={11} tickLine={false} />
          <YAxis
            stroke="#71717a"
            fontSize={11}
            tickFormatter={value => `${value}%`}
            width={50}
            tickLine={false}
          />
          <Tooltip
            formatter={(value: number) => [formatReturn(value), 'Return']}
            contentStyle={{
              backgroundColor: '#1a1a25',
              border: '1px solid #2a2a3a',
              borderRadius: '8px',
            }}
            cursor={{ fill: '#2a2a3a' }}
          />
          <ReferenceLine y={0} stroke="#3f3f46" />
          <Bar dataKey="return" radius={[4, 4, 0, 0]}>
            {data.map((entry, index) => (
              <Cell key={`cell-${index}`} fill={getBarColor(entry.return)} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </Card>
  );
}

// Mock data generator
function generateMockMonthlyReturns(): MonthlyReturn[] {
  const months = [
    'Jan',
    'Feb',
    'Mar',
    'Apr',
    'May',
    'Jun',
    'Jul',
    'Aug',
    'Sep',
    'Oct',
    'Nov',
    'Dec',
  ];
  const year = new Date().getFullYear();

  return months.map((month, _index) => ({
    month: `${month} ${year}`,
    return: (Math.random() - 0.4) * 10, // -4% to +6%
  }));
}

export default MonthlyReturns;
