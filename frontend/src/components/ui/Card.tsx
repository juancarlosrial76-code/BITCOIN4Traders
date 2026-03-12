import { ReactNode } from 'react';
import { Info } from 'lucide-react';
import { TooltipContent, TooltipTrigger, Tooltip } from './Tooltip';

interface CardProps {
  children: ReactNode;
  className?: string;
  title?: string;
  action?: ReactNode;
  helpText?: string;
}

export function Card({ children, className = '', title, action, helpText }: CardProps) {
  return (
    <div className={`bg-card border border-border rounded-lg ${className}`}>
      {(title || action || helpText) && (
        <div className="flex items-center justify-between px-4 py-3 border-b border-border">
          <div className="flex items-center gap-2">
            {title && <h3 className="font-semibold text-text-primary">{title}</h3>}
            {helpText && (
              <Tooltip>
                <TooltipTrigger asChild>
                  <Info size={16} className="text-text-muted cursor-help" />
                </TooltipTrigger>
                <TooltipContent>
                  <p className="max-w-xs">{helpText}</p>
                </TooltipContent>
              </Tooltip>
            )}
          </div>
          {action && <div>{action}</div>}
        </div>
      )}
      <div className="p-4">{children}</div>
    </div>
  );
}
