import { SelectHTMLAttributes, forwardRef } from 'react';

interface SelectProps extends SelectHTMLAttributes<HTMLSelectElement> {
  label?: string;
  error?: string;
  options: { value: string; label: string }[];
}

export const Select = forwardRef<HTMLSelectElement, SelectProps>(
  ({ className = '', label, error, options, ...props }, ref) => {
    return (
      <div className="w-full">
        {label && (
          <label className="block text-sm font-medium text-text-secondary mb-1.5">
            {label}
          </label>
        )}
        <select
          ref={ref}
          className={`w-full h-10 px-3 bg-background border border-border rounded-lg text-text-primary focus:outline-none focus:ring-2 focus:ring-bitcoin-orange/50 focus:border-bitcoin-orange transition-colors appearance-none cursor-pointer ${
            error ? 'border-red-500' : ''
          } ${className}`}
          {...props}
        >
          {options.map((option) => (
            <option key={option.value} value={option.value}>
              {option.label}
            </option>
          ))}
        </select>
        {error && <p className="mt-1 text-sm text-red-400">{error}</p>}
      </div>
    );
  }
);

Select.displayName = 'Select';
