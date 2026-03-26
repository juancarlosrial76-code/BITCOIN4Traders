import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { api, apiInterceptors, fetchWithRetry } from '../api/client';

const API_BASE = 'http://localhost:8000';

global.fetch = vi.fn();

function createMockResponse(data: unknown, ok = true, status = 200) {
  return {
    ok,
    status,
    json: () => Promise.resolve(data),
  } as Response;
}

describe('API Client', () => {
  beforeEach(() => {
    vi.spyOn(console, 'error').mockImplementation(() => {});
    localStorage.clear();
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  describe('fetchWithRetry', () => {
    it('makes a successful request', async () => {
      (global.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        createMockResponse({ status: 'ok' })
      );

      const result = await fetchWithRetry<{ status: string }>('/api/test', {});
      expect(result).toEqual({ status: 'ok' });
    });

    it('retries on server error', async () => {
      (global.fetch as ReturnType<typeof vi.fn>)
        .mockResolvedValueOnce(createMockResponse(null, false, 500))
        .mockResolvedValueOnce(createMockResponse({ status: 'ok' }));

      const result = await fetchWithRetry<{ status: string }>(
        '/api/test',
        {},
        {
          maxRetries: 3,
          retryDelay: 10,
          retryCondition: error => error.status === 500,
        }
      );

      expect(result).toEqual({ status: 'ok' });
      expect(global.fetch).toHaveBeenCalledTimes(2);
    });

    it('throws after max retries', async () => {
      (global.fetch as ReturnType<typeof vi.fn>).mockResolvedValue(
        createMockResponse(null, false, 500)
      );

      await expect(
        fetchWithRetry<{ status: string }>(
          '/api/test',
          {},
          {
            maxRetries: 2,
            retryDelay: 10,
            retryCondition: error => error.status === 500,
          }
        )
      ).rejects.toThrow();

      expect(global.fetch).toHaveBeenCalledTimes(3);
    });

    it('throws after max retries', async () => {
      (global.fetch as ReturnType<typeof vi.fn>)
        .mockResolvedValueOnce(createMockResponse(null, false, 500))
        .mockResolvedValueOnce(createMockResponse(null, false, 500))
        .mockResolvedValueOnce(createMockResponse(null, false, 500));

      await expect(
        fetchWithRetry<{ status: string }>(
          '/api/test',
          {},
          {
            maxRetries: 2,
            retryDelay: 10,
            retryCondition: error => error.status === 500,
          }
        )
      ).rejects.toThrow();

      expect(global.fetch).toHaveBeenCalledTimes(3);
    });

    it('does not retry on client error (4xx)', async () => {
      (global.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        createMockResponse({ message: 'Bad request' }, false, 400)
      );

      await expect(
        fetchWithRetry<{ status: string }>(
          '/api/test',
          {},
          {
            maxRetries: 3,
            retryDelay: 10,
          }
        )
      ).rejects.toThrow();

      expect(global.fetch).toHaveBeenCalledTimes(1);
    });

    it('handles 401 unauthorized', async () => {
      (global.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        createMockResponse({ message: 'Unauthorized' }, false, 401)
      );

      await expect(fetchWithRetry<{ status: string }>('/api/test', {})).rejects.toThrow();

      expect(localStorage.getItem('access_token')).toBeNull();
    });

    it('handles 204 No Content', async () => {
      (global.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        createMockResponse(null, true, 204)
      );

      const result = await fetchWithRetry<void>('/api/test', {});
      expect(result).toBeUndefined();
    });
  });

  describe('api.status', () => {
    it('returns status data', async () => {
      const mockStatus = { status: 'running', timestamp: '2024-01-01', version: '1.0.0' };
      (global.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        createMockResponse(mockStatus)
      );

      const result = await api.status();
      expect(result).toEqual(mockStatus);
    });
  });

  describe('api.trading', () => {
    it('getStatus returns trading status', async () => {
      const mockStatus = {
        is_running: true,
        current_position: 0.5,
        unrealized_pnl: 100,
        timestamp: '2024-01-01',
      };
      (global.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        createMockResponse(mockStatus)
      );

      const result = await api.trading.getStatus();
      expect(result).toEqual(mockStatus);
    });

    it('start sends POST request', async () => {
      (global.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        createMockResponse({ status: 'started', timestamp: '2024-01-01' })
      );

      const result = await api.trading.start();
      expect(result).toEqual({ status: 'started', timestamp: '2024-01-01' });
      expect(global.fetch).toHaveBeenCalledWith(
        `${API_BASE}/api/trading/start`,
        expect.objectContaining({ method: 'POST' })
      );
    });

    it('stop sends POST request', async () => {
      (global.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        createMockResponse({ status: 'stopped', timestamp: '2024-01-01' })
      );

      const result = await api.trading.stop();
      expect(result).toEqual({ status: 'stopped', timestamp: '2024-01-01' });
    });

    it('placeOrder sends order data', async () => {
      (global.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        createMockResponse({ id: 'order-123', status: 'filled' })
      );

      const order = {
        symbol: 'BTCUSDT',
        side: 'buy' as const,
        order_type: 'market' as const,
        quantity: 0.1,
      };

      const result = await api.trading.placeOrder(order);
      expect(result).toEqual({ id: 'order-123', status: 'filled' });
      expect(global.fetch).toHaveBeenCalledWith(
        `${API_BASE}/api/trading/order`,
        expect.objectContaining({
          method: 'POST',
          body: JSON.stringify(order),
        })
      );
    });
  });

  describe('apiInterceptors', () => {
    it('adds and removes request interceptor', () => {
      const interceptor = vi.fn(config => config);
      const remove = apiInterceptors.addRequestInterceptor(interceptor);

      expect(apiInterceptors.addRequestInterceptor).toBeDefined();

      remove();
      expect(apiInterceptors.addRequestInterceptor).toBeDefined();
    });

    it('adds and removes response interceptor', () => {
      const interceptor = vi.fn(response => response);
      const remove = apiInterceptors.addResponseInterceptor(interceptor);

      expect(apiInterceptors.addResponseInterceptor).toBeDefined();

      remove();
      expect(apiInterceptors.addResponseInterceptor).toBeDefined();
    });

    it('adds and removes error interceptor', () => {
      const interceptor = vi.fn(error => error);
      const remove = apiInterceptors.addErrorInterceptor(interceptor);

      expect(apiInterceptors.addErrorInterceptor).toBeDefined();

      remove();
      expect(apiInterceptors.addErrorInterceptor).toBeDefined();
    });
  });
});
