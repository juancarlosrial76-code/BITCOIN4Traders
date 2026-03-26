import { ReactNode, forwardRef } from 'react';
import { Info } from 'lucide-react';

interface CardProps {
  children: ReactNode;
  className?: string;
  title?: string;
  action?: ReactNode;
  helpText?: string;
}

export const Card = forwardRef<HTMLDivElement, CardProps>(
  ({ children, className = '', title, action, helpText }, ref) => {
    return (
      <div ref={ref} className={`bg-card border border-border rounded-lg ${className}`}>
        {(title || action || helpText) && (
          <div className="flex items-center justify-between px-4 py-3 border-b border-border">
            <div className="flex items-center gap-2">
              {title && <h3 className="font-semibold text-text-primary">{title}</h3>}
              {helpText && (
                <div className="group relative">
                  <Info size={16} className="text-text-muted cursor-help" />
                  <div className="absolute bottom-full left-0 mb-2 hidden group-hover:block z-50">
                    <div className="bg-popover border border-border rounded-lg p-3 shadow-xl max-w-xs">
                      <p className="text-sm text-text-primary">{helpText}</p>
                    </div>
                  </div>
                </div>
              )}
            </div>
            {action && <div>{action}</div>}
          </div>
        )}
        <div className="p-4">{children}</div>
      </div>
    );
  }
);

Card.displayName = 'Card';
