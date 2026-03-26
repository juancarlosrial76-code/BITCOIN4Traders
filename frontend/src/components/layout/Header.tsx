import { useLocation, Link } from 'react-router-dom';
import { useAuth } from '../../hooks/useAuth';
import { Button, LanguageSelector } from '../ui';
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
  HelpCircle,
  Wallet,
  History,
} from 'lucide-react';
import { useState } from 'react';
import * as DropdownMenu from '@radix-ui/react-dropdown-menu';

const navItems = [
  { path: '/dashboard', label: 'Dashboard', icon: LayoutDashboard },
  { path: '/trading', label: 'Trading', icon: LineChart },
  { path: '/portfolio', label: 'Portfolio', icon: Wallet },
  { path: '/configuration', label: 'Settings', icon: Settings },
  { path: '/analytics', label: 'Analytics', icon: BarChart3 },
  { path: '/models', label: 'Models', icon: Brain },
  { path: '/system', label: 'System', icon: Monitor },
  { path: '/history', label: 'History', icon: History },
];

const helpItems = [
  { label: '📖 Documentation', href: '/manual/index.html', external: true },
  { label: '🚀 Quick Start', href: '/manual/quickstart.html', external: true },
  { label: '❓ FAQ', href: '/manual/troubleshooting.html', external: true },
  { label: '🔧 API Reference', href: '/manual/api.html', external: true },
  { label: '📊 Glossary', href: '/manual/glossary.html', external: true },
];

export function Header() {
  const { username, logout } = useAuth();
  const location = useLocation();
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  return (
    <header className="bg-card border-b border-border sticky top-0 z-50">
      <div className="flex items-center justify-between h-16 px-4">
        {/* Logo & Mobile Menu */}
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
            <span className="font-bold text-lg text-text-primary hidden md:block">
              BITCOIN4Traders
            </span>
          </Link>
        </div>

        {/* Desktop Navigation */}
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
                <Icon size={16} />
                <span className="hidden xl:inline">{item.label}</span>
              </Link>
            );
          })}
        </nav>

        {/* Right Section */}
        <div className="flex items-center gap-2">
          {/* Language Selector */}
          <LanguageSelector />

          {/* Help Dropdown */}
          <DropdownMenu.Root>
            <DropdownMenu.Trigger asChild>
              <Button variant="ghost" size="sm" className="text-text-secondary">
                <HelpCircle size={18} />
              </Button>
            </DropdownMenu.Trigger>
            <DropdownMenu.Portal>
              <DropdownMenu.Content 
                className="min-w-[200px] bg-card border border-border rounded-lg p-1 shadow-xl z-50"
                sideOffset={5}
                align="end"
              >
                {helpItems.map((item, index) => (
                  <DropdownMenu.Item
                    key={index}
                    className="flex items-center gap-2 px-3 py-2 text-sm text-text-secondary hover:text-text-primary hover:bg-background rounded-md cursor-pointer outline-none"
                    onSelect={() => {
                      if (item.external) {
                        window.open(item.href, '_blank');
                      }
                    }}
                  >
                    {item.label}
                  </DropdownMenu.Item>
                ))}
              </DropdownMenu.Content>
            </DropdownMenu.Portal>
          </DropdownMenu.Root>

          {/* User & Logout */}
          {username && (
            <div className="hidden sm:flex items-center gap-2 px-3 py-1.5 bg-background rounded-lg">
              <div className="w-6 h-6 bg-bitcoin-orange/20 rounded-full flex items-center justify-center">
                <span className="text-bitcoin-orange text-xs font-bold">
                  {username.charAt(0).toUpperCase()}
                </span>
              </div>
              <span className="text-sm text-text-secondary">{username}</span>
            </div>
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

      {/* Mobile Menu */}
      {mobileMenuOpen && (
        <nav className="lg:hidden border-t border-border p-2 max-h-[80vh] overflow-y-auto">
          {navItems.map((item) => {
            const Icon = item.icon;
            const isActive = location.pathname === item.path;
            return (
              <Link
                key={item.path}
                to={item.path}
                onClick={() => setMobileMenuOpen(false)}
                className={`flex items-center gap-3 px-3 py-3 rounded-lg text-sm font-medium transition-colors ${
                  isActive
                    ? 'bg-bitcoin-orange/10 text-bitcoin-orange'
                    : 'text-text-secondary hover:text-text-primary hover:bg-background'
                }`}
              >
                <Icon size={20} />
                {item.label}
              </Link>
            );
          })}
          <div className="border-t border-border mt-2 pt-2">
            <p className="px-3 py-1 text-xs text-text-muted uppercase">Help</p>
            {helpItems.map((item, index) => (
              <button
                key={index}
                onClick={() => {
                  window.open(item.href, '_blank');
                  setMobileMenuOpen(false);
                }}
                className="w-full flex items-center gap-3 px-3 py-2 text-sm text-text-secondary hover:text-text-primary hover:bg-background rounded-lg"
              >
                {item.label}
              </button>
            ))}
          </div>
        </nav>
      )}
    </header>
  );
}
