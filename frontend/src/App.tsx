import { useState, FormEvent } from 'react';
import { BrowserRouter, Routes, Route, Navigate, useNavigate } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { Layout } from './components/layout/Layout';
import { Dashboard } from './pages/Dashboard';
import { Trading } from './pages/Trading';
import { Configuration } from './pages/Configuration';
import { Analytics } from './pages/Analytics';
import { Models } from './pages/Models';
import { System } from './pages/System';
import { AuthProvider, useAuth } from './hooks/useAuth';

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 5000,
      refetchOnWindowFocus: false,
    },
  },
});

function ProtectedRoute({ children }: { children: React.ReactNode }) {
  const { isAuthenticated, isLoading } = useAuth();

  if (isLoading) {
    return (
      <div className="min-h-screen bg-background flex items-center justify-center">
        <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-bitcoin-orange"></div>
      </div>
    );
  }

  if (!isAuthenticated) {
    return <Navigate to="/login" replace />;
  }

  return <>{children}</>;
}

function Login() {
  const { login, isLoading, error, isAuthenticated } = useAuth();
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const navigate = useNavigate();

  // Bereits eingeloggt -> direkt weiterleiten
  if (isAuthenticated) {
    return <Navigate to="/dashboard" replace />;
  }

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
    const ok = await login(username, password);
    if (ok) {
      navigate('/dashboard', { replace: true });
    }
  };

  return (
    <div className="min-h-screen bg-background flex items-center justify-center p-4">
      <div className="bg-card border border-border rounded-xl p-8 w-full max-w-md shadow-2xl">
        {/* Logo + Title */}
        <div className="text-center mb-8">
          <div className="inline-flex items-center justify-center w-16 h-16 rounded-full bg-bitcoin-orange/10 mb-4">
            <svg viewBox="0 0 24 24" className="w-8 h-8 text-bitcoin-orange fill-current">
              <path d="M23.638 14.904c-1.602 6.425-8.113 10.34-14.542 8.736C2.67 22.04-1.244 15.525.362 9.105 1.962 2.68 8.475-1.243 14.9.358c6.43 1.605 10.342 8.115 8.738 14.546z"/>
              <path fill="#fff" d="M17.16 10.49c.24-1.6-.975-2.46-2.635-3.035l.54-2.156-1.315-.328-.524 2.1c-.346-.086-.7-.167-1.054-.247l.527-2.112-1.315-.327-.54 2.154c-.286-.065-.567-.13-.84-.198l.002-.007-1.815-.453-.35 1.404s.975.224.954.238c.533.133.63.486.613.766L8.79 12.68c.046.012.106.028.172.054l-.175-.044-.86 3.445c-.065.16-.23.402-.602.31.013.02-.954-.238-.954-.238L5.7 17.66l1.71.426c.32.08.633.163.941.242l-.545 2.183 1.314.328.54-2.158c.358.097.705.186 1.046.272l-.538 2.15 1.316.329.545-2.18c2.245.425 3.933.254 4.644-1.777.574-1.636-.028-2.578-1.211-3.192.861-.2 1.51-.765 1.682-1.934zm-3.012 4.224c-.408 1.636-3.168.751-4.063.53l.724-2.903c.896.224 3.766.667 3.34 2.373zm.408-4.248c-.373 1.493-2.672.734-3.418.548l.657-2.634c.746.186 3.147.534 2.762 2.086z"/>
            </svg>
          </div>
          <h1 className="text-2xl font-bold text-text-primary">BITCOIN4Traders</h1>
          <p className="text-text-secondary mt-1">AI Trading Bot Control Center</p>
        </div>

        {/* Login Form */}
        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-text-secondary mb-1">
              Username
            </label>
            <input
              type="text"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              placeholder="admin"
              required
              autoFocus
              className="w-full bg-background border border-border rounded-lg px-4 py-3 text-text-primary placeholder-text-muted focus:outline-none focus:ring-2 focus:ring-bitcoin-orange/50 focus:border-bitcoin-orange transition-colors"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-text-secondary mb-1">
              Password
            </label>
            <input
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              placeholder="••••••••"
              required
              className="w-full bg-background border border-border rounded-lg px-4 py-3 text-text-primary placeholder-text-muted focus:outline-none focus:ring-2 focus:ring-bitcoin-orange/50 focus:border-bitcoin-orange transition-colors"
            />
          </div>

          {/* Error Message */}
          {error && (
            <div className="flex items-center gap-2 p-3 bg-red-500/10 border border-red-500/20 rounded-lg">
              <svg className="w-4 h-4 text-red-400 flex-shrink-0" fill="currentColor" viewBox="0 0 20 20">
                <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7 4a1 1 0 11-2 0 1 1 0 012 0zm-1-9a1 1 0 00-1 1v4a1 1 0 102 0V6a1 1 0 00-1-1z" clipRule="evenodd"/>
              </svg>
              <p className="text-sm text-red-400">{error}</p>
            </div>
          )}

          <button
            type="submit"
            disabled={isLoading || !username || !password}
            className="w-full bg-bitcoin-orange hover:bg-bitcoin-orange/90 disabled:opacity-50 disabled:cursor-not-allowed text-white font-semibold py-3 px-4 rounded-lg transition-colors flex items-center justify-center gap-2"
          >
            {isLoading ? (
              <>
                <div className="animate-spin rounded-full h-4 w-4 border-t-2 border-b-2 border-white"></div>
                Signing in...
              </>
            ) : (
              'Sign In'
            )}
          </button>
        </form>

        {/* Hint */}
        <p className="text-center text-xs text-text-muted mt-6">
          Default: <span className="text-text-secondary font-mono">admin</span> / <span className="text-text-secondary font-mono">admin123</span>
        </p>
      </div>
    </div>
  );
}

function AppRoutes() {
  return (
    <Routes>
      <Route path="/login" element={<Login />} />
      <Route
        path="/"
        element={
          <ProtectedRoute>
            <Layout />
          </ProtectedRoute>
        }
      >
        <Route index element={<Navigate to="/dashboard" replace />} />
        <Route path="dashboard" element={<Dashboard />} />
        <Route path="trading" element={<Trading />} />
        <Route path="configuration" element={<Configuration />} />
        <Route path="analytics" element={<Analytics />} />
        <Route path="models" element={<Models />} />
        <Route path="system" element={<System />} />
      </Route>
    </Routes>
  );
}

export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      {/* BrowserRouter muss außen sein damit useNavigate überall funktioniert */}
      <BrowserRouter future={{ v7_startTransition: true, v7_relativeSplatPath: true }}>
        <AuthProvider>
          <AppRoutes />
        </AuthProvider>
      </BrowserRouter>
    </QueryClientProvider>
  );
}
