const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000';

async function fetchApi<T>(endpoint: string, options?: RequestInit): Promise<T> {
  const token = localStorage.getItem('access_token');

  const response = await fetch(`${API_BASE}${endpoint}`, {
    headers: {
      'Content-Type': 'application/json',
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
      ...options?.headers,
    },
    ...options,
  });

  if (response.status === 401) {
    // Token expired or invalid — force re-login
    localStorage.removeItem('access_token');
    localStorage.removeItem('username');
    window.location.href = '/login';
    throw new Error('Session expired. Please log in again.');
  }

  if (!response.ok) {
    throw new Error(`API Error: ${response.status}`);
  }

  return response.json();
}

export const api = {
  status: () => fetchApi<{ status: string; timestamp: string; version: string }>('/api/status'),
  
  trading: {
    getStatus: () => fetchApi<{ is_running: boolean; current_position: number; unrealized_pnl: number; timestamp: string }>('/api/trading/status'),
    start: () => fetchApi<{ status: string; timestamp: string }>('/api/trading/start', { method: 'POST' }),
    stop: () => fetchApi<{ status: string; timestamp: string }>('/api/trading/stop', { method: 'POST' }),
    getOrders: () => fetchApi<any[]>('/api/trading/orders'),
    placeOrder: (order: { symbol: string; side: string; order_type: string; quantity: number; price?: number }) =>
      fetchApi<any>('/api/trading/order', { method: 'POST', body: JSON.stringify(order) }),
    getConfig: () => fetchApi<any>('/api/trading/config'),
    updateConfig: (config: any) => fetchApi<any>('/api/trading/config', { method: 'PUT', body: JSON.stringify(config) }),
  },

  config: {
    get: () => fetchApi<any>('/api/config/'),
    getBot: () => fetchApi<any>('/api/config/bot'),
    updateBot: (config: any) => fetchApi<any>('/api/config/bot', { method: 'PUT', body: JSON.stringify(config) }),
    getRisk: () => fetchApi<any>('/api/config/risk'),
    updateRisk: (config: any) => fetchApi<any>('/api/config/risk', { method: 'PUT', body: JSON.stringify(config) }),
    getData: () => fetchApi<any>('/api/config/data'),
    updateData: (config: any) => fetchApi<any>('/api/config/data', { method: 'PUT', body: JSON.stringify(config) }),
  },

  analytics: {
    getMetrics: () => fetchApi<any>('/api/analytics/metrics'),
    getEquityCurve: () => fetchApi<{ timestamp: string; value: number }[]>('/api/analytics/equity-curve'),
    getMonthlyReturns: () => fetchApi<{ month: string; return: number }[]>('/api/analytics/monthly-returns'),
    getTradeDistribution: () => fetchApi<{ range: string; count: number }[]>('/api/analytics/trade-distribution'),
  },

  models: {
    list: () => fetchApi<any[]>('/api/models/'),
    get: (id: number) => fetchApi<any>(`/api/models/${id}`),
    train: () => fetchApi<any>('/api/models/train', { method: 'POST' }),
    delete: (id: number) => fetchApi<any>(`/api/models/${id}`, { method: 'DELETE' }),
    getTrainingHistory: () => fetchApi<any[]>('/api/models/training/history'),
  },

  system: {
    getMetrics: () => fetchApi<any>('/api/system/metrics'),
    getLogs: () => fetchApi<any[]>('/api/system/logs'),
    getEndpoints: () => fetchApi<any[]>('/api/system/endpoints'),
    getEnv: () => fetchApi<any[]>('/api/system/env'),
  },
};
