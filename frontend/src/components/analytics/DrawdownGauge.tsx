import { useEffect, useState } from 'react';
import { Card } from '../ui/Card';

interface DrawdownGaugeProps {
  height?: number;
  maxDrawdownThreshold?: number;
}

export function DrawdownGauge({ height = 200, maxDrawdownThreshold = 20 }: DrawdownGaugeProps) {
  const [currentDrawdown, setCurrentDrawdown] = useState(0);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // Simulate drawdown calculation
    const calculateDrawdown = () => {
      // In production, this would calculate from equity curve
      const mockDrawdown = Math.random() * 15; // 0-15% drawdown
      setCurrentDrawdown(mockDrawdown);
      setLoading(false);
    };

    calculateDrawdown();
  }, []);

  const percentage = Math.min((currentDrawdown / maxDrawdownThreshold) * 100, 100);

  // Color based on severity
  const getColor = (pct: number) => {
    if (pct < 40) return '#10b981'; // Green
    if (pct < 70) return '#f59e0b'; // Yellow
    return '#ef4444'; // Red
  };

  const color = getColor(percentage);
  const isWarning = percentage >= 60;
  const isCritical = percentage >= 80;

  if (loading) {
    return (
      <Card className="p-4">
        <h3 className="font-bold mb-4">Drawdown</h3>
        <div className="flex items-center justify-center text-gray-400" style={{ height }}>
          Loading...
        </div>
      </Card>
    );
  }

  // SVG arc calculation
  const radius = 70;
  const strokeWidth = 12;
  const circumference = 2 * Math.PI * radius;
  const arcLength = circumference * 0.75; // 270 degrees
  const offset = arcLength - (arcLength * percentage) / 100;

  return (
    <Card
      className={`p-4 ${isCritical ? 'border-red-500/50' : isWarning ? 'border-yellow-500/50' : ''}`}
    >
      <h3 className="font-bold mb-4">Drawdown</h3>

      <div className="flex flex-col items-center" style={{ height }}>
        <svg width="180" height={height - 40} viewBox="0 0 180 140">
          {/* Background arc */}
          <circle
            cx="90"
            cy="80"
            r={radius}
            fill="none"
            stroke="#1a1a25"
            strokeWidth={strokeWidth}
            strokeDasharray={`${arcLength} ${circumference}`}
            strokeLinecap="round"
            transform="rotate(135 90 80)"
          />

          {/* Value arc */}
          <circle
            cx="90"
            cy="80"
            r={radius}
            fill="none"
            stroke={color}
            strokeWidth={strokeWidth}
            strokeDasharray={`${arcLength} ${circumference}`}
            strokeLinecap="round"
            transform="rotate(135 90 80)"
            style={{
              strokeDashoffset: offset,
              transition: 'stroke-dashoffset 0.5s ease',
              filter: `drop-shadow(0 0 8px ${color}40)`,
            }}
          />

          {/* Center text */}
          <text
            x="90"
            y="70"
            textAnchor="middle"
            fill="white"
            fontSize="28"
            fontWeight="bold"
            className="font-mono"
          >
            {currentDrawdown.toFixed(1)}%
          </text>
          <text x="90" y="95" textAnchor="middle" fill="#71717a" fontSize="12">
            Current Drawdown
          </text>
        </svg>

        {/* Warning indicator */}
        {isWarning && (
          <div
            className={`mt-2 px-3 py-1 rounded text-sm ${
              isCritical ? 'bg-red-500/20 text-red-400' : 'bg-yellow-500/20 text-yellow-400'
            }`}
          >
            {isCritical ? '⚠️ Critical Drawdown!' : '⚠️ Approaching Limit'}
          </div>
        )}

        {/* Max drawdown indicator */}
        <div className="mt-2 text-sm text-gray-400">Max Threshold: {maxDrawdownThreshold}%</div>
      </div>
    </Card>
  );
}

export default DrawdownGauge;
