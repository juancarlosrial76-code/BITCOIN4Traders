import { ReactNode } from 'react';

interface CardProps {
  children: ReactNode;
  className?: string;
  title?: string;
  action?: ReactNode;
}

export function Card({ children, className = '', title, action }: CardProps) {
  return (
    <div className={`bg-card border border-border rounded-lg ${className}`}>
      {(title || action) && (
        <div className="flex items-center justify-between px-4 py-3 border-b border-border">
          {title && <h3 className="font-semibold text-text-primary">{title}</h3>}
          {action && <div>{action}</div>}
        </div>
      )}
      <div className="p-4">{children}</div>
    </div>
  );
}
