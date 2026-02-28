import { createContext, useContext, useState, useCallback, ReactNode } from 'react';

interface AuthContextType {
  isAuthenticated: boolean;
  isLoading: boolean;
  username: string | null;
  error: string | null;
  login: (username: string, password: string) => Promise<boolean>;
  logout: () => void;
}

const AuthContext = createContext<AuthContextType | null>(null);

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export function AuthProvider({ children }: { children: ReactNode }) {
  const [isAuthenticated, setIsAuthenticated] = useState(() => {
    const token = localStorage.getItem('access_token');
    return !!token;
  });
  const [username, setUsername] = useState<string | null>(() => {
    return localStorage.getItem('username');
  });
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const login = useCallback(async (username: string, password: string): Promise<boolean> => {
    setIsLoading(true);
    setError(null);
    try {
      const response = await fetch(`${API_BASE}/api/auth/login`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username, password }),
      });

      if (!response.ok) {
        const data = await response.json();
        setError(data.detail || 'Login fehlgeschlagen');
        setIsLoading(false);
        return false;
      }

      const data = await response.json();
      localStorage.setItem('access_token', data.access_token);
      localStorage.setItem('username', username);
      setUsername(username);
      setIsAuthenticated(true);
      setIsLoading(false);
      return true;
    } catch {
      setError('Verbindung zum Server fehlgeschlagen');
      setIsLoading(false);
      return false;
    }
  }, []);

  const logout = useCallback(() => {
    localStorage.removeItem('access_token');
    localStorage.removeItem('username');
    setUsername(null);
    setIsAuthenticated(false);
    setError(null);
  }, []);

  return (
    <AuthContext.Provider value={{ isAuthenticated, isLoading, username, error, login, logout }}>
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
}
