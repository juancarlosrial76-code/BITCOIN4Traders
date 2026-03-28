import { useState, useEffect, useMemo } from 'react';
import { Outlet, NavLink, useLocation, useNavigate } from 'react-router-dom';
import { useAuth } from '../../hooks/useAuth';
import { useTranslation } from 'react-i18next';
import { Button } from '../ui/Button';
import { LanguageSwitcher } from '../ui/LanguageSwitcher';
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '../ui/Tooltip';
import {
  LayoutDashboard,
  LineChart,
  Wallet,
  History,
  Settings,
  BarChart3,
  Brain,
  Monitor,
  LogOut,
  Menu,
  X,
  HelpCircle,
  Book,
  FileQuestion,
  Code,
  BookOpen,
  ChevronRight,
} from 'lucide-react';
import * as DropdownMenu from '@radix-ui/react-dropdown-menu';

export function Layout() {
  const { t } = useTranslation();
  const { username, logout } = useAuth();
  const location = useLocation();
  const navigate = useNavigate();
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  const navItems = useMemo(
    () => [
      {
        path: '/dashboard',
        label: t('nav.dashboard', 'Dashboard'),
        icon: LayoutDashboard,
        description: t('nav.dashboardDesc', 'Overview and quick stats'),
      },
      {
        path: '/trading',
        label: t('nav.trading', 'Trading'),
        icon: LineChart,
        description: t('nav.tradingDesc', 'Live trading and positions'),
      },
      {
        path: '/portfolio',
        label: t('nav.portfolio', 'Portfolio'),
        icon: Wallet,
        description: t('nav.portfolioDesc', 'Assets and allocation'),
      },
      {
        path: '/history',
        label: t('nav.history', 'History'),
        icon: History,
        description: t('nav.historyDesc', 'Trade history and logs'),
      },
      {
        path: '/configuration',
        label: t('nav.configuration', 'Configuration'),
        icon: Settings,
        description: t('nav.configurationDesc', 'Bot parameters and settings'),
      },
      {
        path: '/analytics',
        label: t('nav.analytics', 'Analytics'),
        icon: BarChart3,
        description: t('nav.analyticsDesc', 'Performance charts and metrics'),
      },
      {
        path: '/models',
        label: t('nav.models', 'Models'),
        icon: Brain,
        description: t('nav.modelsDesc', 'ML models and training'),
      },
      {
        path: '/system',
        label: t('nav.system', 'System'),
        icon: Monitor,
        description: t('nav.systemDesc', 'Server status and logs'),
      },
    ],
    [t]
  );

  // All docs are now React routes — use navigate() instead of window.open()
  const helpItems = useMemo(
    () => [
      {
        label: t('help.faq', '❓ FAQ'),
        icon: FileQuestion,
        description: t('help.faqDesc', 'Frequently asked questions'),
        action: () => navigate('/faq'),
      },
      {
        label: t('help.documentation', '📖 Documentation'),
        icon: Book,
        description: t('help.documentationDesc', 'Read full documentation'),
        action: () => navigate('/docs'),
      },
      {
        label: t('help.quickstart', '🚀 Quickstart'),
        icon: ChevronRight,
        description: t('help.quickstartDesc', 'Get started in 5 minutes'),
        action: () => navigate('/docs/quickstart'),
      },
      {
        label: t('help.api', '🔧 API Reference'),
        icon: Code,
        description: t('help.apiDesc', 'API endpoints documentation'),
        action: () => navigate('/docs/api'),
      },
      {
        label: t('help.glossary', '📊 Glossary'),
        icon: BookOpen,
        description: t('help.glossaryDesc', 'Trading terms explained'),
        action: () => navigate('/docs/glossary'),
      },
    ],
    [t, navigate]
  );

  useEffect(() => {
    const handleResize = () => {
      if (window.innerWidth < 1024) {
        setSidebarOpen(false);
      } else {
        setSidebarOpen(true);
      }
    };
    handleResize();
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  const currentPage = navItems.find(item => item.path === location.pathname);

  return (
    <TooltipProvider delayDuration={300}>
      <div className="min-h-screen bg-background flex">
        {/* Sidebar - Desktop */}
        <aside
          className={`fixed inset-y-0 left-0 z-40 bg-card border-r border-border transition-all duration-300 ${
            sidebarOpen ? 'w-64' : 'w-16'
          } hidden lg:flex flex-col`}
        >
          {/* Logo */}
          <div className="h-16 flex items-center justify-between px-4 border-b border-border">
            <NavLink to="/dashboard" className="flex items-center gap-2">
              <div className="w-8 h-8 bg-bitcoin-orange rounded-full flex items-center justify-center flex-shrink-0">
                <span className="text-white font-bold text-sm">₿</span>
              </div>
              {sidebarOpen && (
                <span className="font-bold text-sm text-text-primary whitespace-nowrap">
                  BITCOIN4Traders
                </span>
              )}
            </NavLink>
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setSidebarOpen(!sidebarOpen)}
              className="text-text-secondary hover:text-text-primary"
            >
              <ChevronRight
                className={`transition-transform ${sidebarOpen ? 'rotate-180' : ''}`}
                size={16}
              />
            </Button>
          </div>

          {/* Navigation */}
          <nav className="flex-1 py-4 px-2 space-y-1 overflow-y-auto">
            {navItems.map(item => {
              const Icon = item.icon;
              const isActive = location.pathname === item.path;
              return (
                <Tooltip key={item.path}>
                  <TooltipTrigger asChild>
                    <NavLink
                      to={item.path}
                      className={`flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-all ${
                        isActive
                          ? 'bg-bitcoin-orange/10 text-bitcoin-orange'
                          : 'text-text-secondary hover:text-text-primary hover:bg-background'
                      }`}
                    >
                      <Icon size={20} className="flex-shrink-0" />
                      {sidebarOpen && (
                        <>
                          <span className="flex-1">{item.label}</span>
                          {isActive && (
                            <div className="w-1.5 h-1.5 bg-bitcoin-orange rounded-full" />
                          )}
                        </>
                      )}
                    </NavLink>
                  </TooltipTrigger>
                  {!sidebarOpen && (
                    <TooltipContent side="right" className="max-w-xs">
                      <p className="font-medium">{item.label}</p>
                      <p className="text-sm text-text-muted mt-1">{item.description}</p>
                    </TooltipContent>
                  )}
                </Tooltip>
              );
            })}
          </nav>

          {/* Help Dropdown */}
          <div className="p-2 border-t border-border">
            <DropdownMenu.Root>
              <DropdownMenu.Trigger asChild>
                <button
                  className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium text-text-secondary hover:text-text-primary hover:bg-background transition-colors ${
                    !sidebarOpen ? 'justify-center' : ''
                  }`}
                >
                  <HelpCircle size={20} />
                  {sidebarOpen && <span>{t('nav.help', 'Help & Documentation')}</span>}
                </button>
              </DropdownMenu.Trigger>
              <DropdownMenu.Portal>
                <DropdownMenu.Content
                  className="min-w-[220px] bg-card border border-border rounded-lg p-1 shadow-xl z-50"
                  sideOffset={5}
                  align="start"
                >
                  {helpItems.map((item, index) => {
                    return (
                      <DropdownMenu.Item
                        key={index}
                        className="flex items-center gap-3 px-3 py-2 text-sm text-text-secondary hover:text-text-primary hover:bg-background rounded-md cursor-pointer outline-none"
                        onSelect={item.action}
                      >
                        <item.icon size={16} />
                        <div className="flex flex-col">
                          <span>{item.label}</span>
                          <span className="text-xs text-text-muted">{item.description}</span>
                        </div>
                      </DropdownMenu.Item>
                    );
                  })}
                  <DropdownMenu.Separator className="h-px bg-border my-1" />
                  <DropdownMenu.Item
                    className="flex items-center gap-3 px-3 py-2 text-sm text-bitcoin-orange hover:bg-background rounded-md cursor-pointer outline-none"
                    onSelect={() => navigate('/docs/troubleshooting')}
                  >
                    <HelpCircle size={16} />
                    <span>Troubleshooting</span>
                  </DropdownMenu.Item>
                </DropdownMenu.Content>
              </DropdownMenu.Portal>
            </DropdownMenu.Root>
          </div>

          {/* User Section */}
          <div className="p-2 border-t border-border">
            <div
              className={`flex items-center gap-3 px-3 py-2 ${!sidebarOpen ? 'justify-center' : ''}`}
            >
              <div className="w-8 h-8 bg-bitcoin-orange/20 rounded-full flex items-center justify-center flex-shrink-0">
                <span className="text-bitcoin-orange text-sm font-bold">
                  {username?.charAt(0).toUpperCase() || 'U'}
                </span>
              </div>
              {sidebarOpen && (
                <div className="flex-1 min-w-0">
                  <p className="text-sm font-medium text-text-primary truncate">{username}</p>
                  <p className="text-xs text-text-muted">Online</p>
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
            {/* Language Switcher */}
            <div className={`px-3 py-2 ${!sidebarOpen ? 'hidden' : ''}`}>
              <LanguageSwitcher variant="dropdown" showName size="sm" />
            </div>
          </div>
        </aside>

        {/* Mobile Header */}
        <header className="lg:hidden fixed top-0 left-0 right-0 h-16 bg-card border-b border-border z-50">
          <div className="flex items-center justify-between h-full px-4">
            <button
              className="p-2 hover:bg-background rounded-lg"
              onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
            >
              {mobileMenuOpen ? <X size={20} /> : <Menu size={20} />}
            </button>
            <NavLink to="/dashboard" className="flex items-center gap-2">
              <div className="w-8 h-8 bg-bitcoin-orange rounded-full flex items-center justify-center">
                <span className="text-white font-bold text-sm">₿</span>
              </div>
              <span className="font-bold text-sm text-text-primary">BITCOIN4Traders</span>
            </NavLink>
            <div className="w-10" />
          </div>

          {/* Mobile Menu */}
          {mobileMenuOpen && (
            <div className="absolute top-16 left-0 right-0 bg-card border-b border-border p-2 shadow-xl">
              <nav className="space-y-1">
                {navItems.map(item => {
                  const Icon = item.icon;
                  const isActive = location.pathname === item.path;
                  return (
                    <NavLink
                      key={item.path}
                      to={item.path}
                      onClick={() => setMobileMenuOpen(false)}
                      className={`flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-colors ${
                        isActive
                          ? 'bg-bitcoin-orange/10 text-bitcoin-orange'
                          : 'text-text-secondary hover:text-text-primary hover:bg-background'
                      }`}
                    >
                      <Icon size={20} />
                      <div className="flex flex-col">
                        <span>{item.label}</span>
                        <span className="text-xs text-text-muted">{item.description}</span>
                      </div>
                    </NavLink>
                  );
                })}
              </nav>
              <div className="border-t border-border mt-2 pt-2">
                <p className="px-3 py-1 text-xs text-text-muted uppercase">
                  {t('nav.help', 'Help')}
                </p>
                {helpItems.slice(0, 3).map((item, index) => (
                  <button
                    key={index}
                    onClick={() => {
                      item.action?.();
                      setMobileMenuOpen(false);
                    }}
                    className="w-full flex items-center gap-3 px-3 py-2 text-sm text-text-secondary hover:text-text-primary hover:bg-background rounded-lg"
                  >
                    <item.icon size={18} />
                    <span>{item.label}</span>
                  </button>
                ))}
              </div>
            </div>
          )}
        </header>

        {/* Main Content */}
        <main
          className={`flex-1 transition-all duration-300 ${
            sidebarOpen ? 'lg:ml-64' : 'lg:ml-16'
          } pt-16 lg:pt-0`}
        >
          {/* Breadcrumb / Page Title */}
          <div className="bg-card/50 border-b border-border px-4 py-3 lg:hidden">
            <div className="flex items-center gap-2 text-sm">
              <NavLink to="/dashboard" className="text-text-muted hover:text-text-primary">
                Home
              </NavLink>
              <ChevronRight size={14} className="text-text-muted" />
              <span className="text-text-primary font-medium">
                {currentPage?.label || 'Dashboard'}
              </span>
            </div>
          </div>

          {/* Page Content */}
          <div className="p-4 md:p-6 min-h-screen">
            <Outlet />
          </div>
        </main>
      </div>
    </TooltipProvider>
  );
}
