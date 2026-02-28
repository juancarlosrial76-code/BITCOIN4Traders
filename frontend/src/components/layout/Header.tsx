import { useLocation, Link } from 'react-router-dom';
import { useAuth } from '../../hooks/useAuth';
import { Button } from '../ui/Button';
import {
  LayoutDashboard,
  LineChart,
  Settings,
  BarChart3,
  Brain,
  Monitor,
  LogOut,
  Menu,
  X,
} from 'lucide-react';
import { useState } from 'react';

const navItems = [
  { path: '/dashboard', label: 'Dashboard', icon: LayoutDashboard },
  { path: '/trading', label: 'Trading', icon: LineChart },
  { path: '/configuration', label: 'Configuration', icon: Settings },
  { path: '/analytics', label: 'Analytics', icon: BarChart3 },
  { path: '/models', label: 'Models', icon: Brain },
  { path: '/system', label: 'System', icon: Monitor },
];

export function Header() {
  const { username, logout } = useAuth();
  const location = useLocation();
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  return (
    <header className="bg-card border-b border-border sticky top-0 z-50">
      <div className="flex items-center justify-between h-16 px-4">
        <div className="flex items-center gap-3">
          <button
            className="lg:hidden p-2 hover:bg-background rounded-lg"
            onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
          >
            {mobileMenuOpen ? <X size={20} /> : <Menu size={20} />}
          </button>
          <Link to="/dashboard" className="flex items-center gap-2">
            <div className="w-8 h-8 bg-bitcoin-orange rounded-full flex items-center justify-center">
              <span className="text-white font-bold text-sm">₿</span>
            </div>
            <span className="font-bold text-lg text-text-primary hidden sm:block">
              BITCOIN4Traders
            </span>
          </Link>
        </div>

        <nav className="hidden lg:flex items-center gap-1">
          {navItems.map((item) => {
            const Icon = item.icon;
            const isActive = location.pathname === item.path;
            return (
              <Link
                key={item.path}
                to={item.path}
                className={`flex items-center gap-2 px-3 py-2 rounded-lg text-sm font-medium transition-colors ${
                  isActive
                    ? 'bg-bitcoin-orange/10 text-bitcoin-orange'
                    : 'text-text-secondary hover:text-text-primary hover:bg-background'
                }`}
              >
                <Icon size={18} />
                {item.label}
              </Link>
            );
          })}
        </nav>

        <div className="flex items-center gap-3">
          <div className="hidden sm:flex items-center gap-2 px-3 py-1.5 bg-background rounded-lg">
            <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
            <span className="text-sm text-text-secondary">Connected</span>
          </div>
          {username && (
            <span className="text-sm text-text-secondary hidden md:block font-medium">
              {username}
            </span>
          )}
          <Button
            variant="ghost"
            size="sm"
            onClick={logout}
            className="text-text-secondary hover:text-red-400"
          >
            <LogOut size={18} />
          </Button>
        </div>
      </div>

      {mobileMenuOpen && (
        <nav className="lg:hidden border-t border-border p-2">
          {navItems.map((item) => {
            const Icon = item.icon;
            const isActive = location.pathname === item.path;
            return (
              <Link
                key={item.path}
                to={item.path}
                onClick={() => setMobileMenuOpen(false)}
                className={`flex items-center gap-3 px-3 py-2 rounded-lg text-sm font-medium transition-colors ${
                  isActive
                    ? 'bg-bitcoin-orange/10 text-bitcoin-orange'
                    : 'text-text-secondary hover:text-text-primary hover:bg-background'
                }`}
              >
                <Icon size={18} />
                {item.label}
              </Link>
            );
          })}
        </nav>
      )}
    </header>
  );
}
