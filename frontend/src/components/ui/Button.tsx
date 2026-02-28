import { ButtonHTMLAttributes, forwardRef } from 'react';

interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: 'primary' | 'secondary' | 'ghost' | 'danger';
  size?: 'sm' | 'md' | 'lg';
}

export const Button = forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className = '', variant = 'primary', size = 'md', children, ...props }, ref) => {
    const baseStyles =
      'inline-flex items-center justify-center font-medium rounded-lg transition-colors focus:outline-none focus:ring-2 focus:ring-bitcoin-orange/50 disabled:opacity-50 disabled:cursor-not-allowed';

    const variants = {
      primary: 'bg-bitcoin-orange hover:bg-bitcoin-orange/90 text-white',
      secondary: 'bg-background hover:bg-border text-text-primary border border-border',
      ghost: 'hover:bg-background text-text-secondary hover:text-text-primary',
      danger: 'bg-red-500/10 hover:bg-red-500/20 text-red-400 border border-red-500/20',
    };

    const sizes = {
      sm: 'h-8 px-3 text-sm gap-1.5',
      md: 'h-10 px-4 text-sm gap-2',
      lg: 'h-12 px-6 text-base gap-2',
    };

    return (
      <button
        ref={ref}
        className={`${baseStyles} ${variants[variant]} ${sizes[size]} ${className}`}
        {...props}
      >
        {children}
      </button>
    );
  }
);

Button.displayName = 'Button';
