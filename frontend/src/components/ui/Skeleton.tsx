import { memo, HTMLAttributes } from 'react';
import { clsx } from 'clsx';

interface SkeletonProps extends HTMLAttributes<HTMLDivElement> {
  variant?: 'text' | 'circular' | 'rectangular';
  width?: string | number;
  height?: string | number;
  animation?: 'pulse' | 'wave' | 'none';
}

export const Skeleton = memo(function Skeleton({
  variant = 'text',
  width,
  height,
  animation = 'pulse',
  className,
  ...props
}: SkeletonProps) {
  const baseStyles = 'bg-gray-700/50';

  const variantStyles = {
    text: 'rounded',
    circular: 'rounded-full',
    rectangular: 'rounded-lg',
  };

  const animationStyles = {
    pulse: 'animate-pulse',
    wave: 'animate-shimmer',
    none: '',
  };

  return (
    <div
      className={clsx(baseStyles, variantStyles[variant], animationStyles[animation], className)}
      style={{
        width: width,
        height: height || (variant === 'text' ? '1em' : undefined),
      }}
      aria-hidden="true"
      {...props}
    />
  );
});

interface SkeletonCardProps {
  lines?: number;
}

export const SkeletonCard = memo(function SkeletonCard({ lines = 3 }: SkeletonCardProps) {
  return (
    <div className="bg-card border border-border rounded-lg p-4 space-y-3">
      <Skeleton variant="text" width="40%" height={20} />
      {Array.from({ length: lines }).map((_, i) => (
        <Skeleton key={i} variant="text" width={i === lines - 1 ? '60%' : '100%'} height={16} />
      ))}
    </div>
  );
});

interface SkeletonTableProps {
  rows?: number;
  columns?: number;
}

export const SkeletonTable = memo(function SkeletonTable({
  rows = 5,
  columns = 4,
}: SkeletonTableProps) {
  return (
    <div className="bg-card border border-border rounded-lg overflow-hidden">
      <div className="bg-background p-3 border-b border-border">
        <div className="flex gap-4">
          {Array.from({ length: columns }).map((_, i) => (
            <Skeleton key={i} variant="text" width={80} height={16} />
          ))}
        </div>
      </div>
      <div className="divide-y divide-border">
        {Array.from({ length: rows }).map((_, rowIndex) => (
          <div key={rowIndex} className="p-3 flex gap-4">
            {Array.from({ length: columns }).map((_, colIndex) => (
              <Skeleton
                key={colIndex}
                variant="text"
                width={colIndex === 0 ? 100 : 60}
                height={16}
              />
            ))}
          </div>
        ))}
      </div>
    </div>
  );
});

interface SkeletonChartProps {
  height?: number;
}

export const SkeletonChart = memo(function SkeletonChart({ height = 300 }: SkeletonChartProps) {
  return (
    <div className="bg-card border border-border rounded-lg p-4" style={{ height }}>
      <Skeleton variant="text" width={120} height={20} className="mb-4" />
      <Skeleton variant="rectangular" width="100%" height={height - 60} />
    </div>
  );
});

interface SkeletonGridProps {
  count?: number;
  columns?: number;
}

export const SkeletonGrid = memo(function SkeletonGrid({
  count = 4,
  columns = 2,
}: SkeletonGridProps) {
  return (
    <div
      className="grid gap-4"
      style={{ gridTemplateColumns: `repeat(${columns}, minmax(0, 1fr))` }}
    >
      {Array.from({ length: count }).map((_, i) => (
        <SkeletonCard key={i} />
      ))}
    </div>
  );
});

export default Skeleton;
