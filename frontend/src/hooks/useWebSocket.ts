import { useEffect, useRef, useState, useCallback } from 'react';

const WS_URL = import.meta.env.VITE_WS_URL || 'ws://localhost:8000/ws';

export interface PriceUpdate {
  type: string;
  data: {
    symbol: string;
    price: number;
    timestamp: string;
  };
}

export interface WebSocketStatus {
  status: 'connecting' | 'connected' | 'disconnected' | 'reconnecting';
  reconnectAttempt: number;
}

interface UseWebSocketOptions {
  url?: string;
  reconnect?: boolean;
  maxReconnectAttempts?: number;
  bufferInterval?: number;
  maxBufferSize?: number;
  onMessage?: (data: PriceUpdate['data']) => void;
  onOpen?: () => void;
  onClose?: () => void;
  onError?: (error: Event) => void;
}

interface UseWebSocketReturn {
  isConnected: boolean;
  lastPrice: number | null;
  status: WebSocketStatus;
  sendMessage: (message: object) => void;
  reconnect: () => void;
  disconnect: () => void;
}

export function useWebSocket(options: UseWebSocketOptions = {}): UseWebSocketReturn {
  const {
    reconnect = true,
    maxReconnectAttempts = 10,
    bufferInterval = 100,
    maxBufferSize = 1000,
    onMessage,
    onOpen,
    onClose,
    onError,
  } = options;

  const [isConnected, setIsConnected] = useState(false);
  const [lastPrice, setLastPrice] = useState<number | null>(null);
  const [status, setStatus] = useState<WebSocketStatus>({
    status: 'disconnected',
    reconnectAttempt: 0,
  });

  const wsRef = useRef<WebSocket | null>(null);
  const reconnectAttemptRef = useRef(0);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const bufferRef = useRef<PriceUpdate['data'][]>([]);
  const flushIntervalRef = useRef<NodeJS.Timeout | null>(null);
  const isUnmountedRef = useRef(false);

  // Buffer flush - verhindert Re-Render-Flut
  // KRITISCH: Bei 10 Messages/Sek nur 10 Re-Renders/Sek statt 10/Sek
  useEffect(() => {
    flushIntervalRef.current = setInterval(() => {
      if (bufferRef.current.length === 0 || isUnmountedRef.current) return;

      // Nur DER NEUESTE Wert wird gesetzt
      const latest = bufferRef.current[bufferRef.current.length - 1];
      setLastPrice(latest.price);
      onMessage?.(latest);

      // Buffer leeren
      bufferRef.current = [];
    }, bufferInterval);

    return () => {
      if (flushIntervalRef.current) {
        clearInterval(flushIntervalRef.current);
      }
    };
  }, [bufferInterval, onMessage]);

  const connect = useCallback(() => {
    if (
      wsRef.current?.readyState === WebSocket.OPEN ||
      wsRef.current?.readyState === WebSocket.CONNECTING
    ) {
      return;
    }

    setStatus(prev => ({ ...prev, status: 'connecting' }));

    const token = localStorage.getItem('access_token');
    // Use WebSocket subprotocol to pass auth token — avoids token appearing in
    // server logs, browser history, and proxy access logs (HIGH-004 fix).
    const ws = token
      ? new WebSocket(WS_URL, [`token.${token}`])
      : new WebSocket(WS_URL);

    ws.onopen = () => {
      if (isUnmountedRef.current) {
        ws.close();
        return;
      }

      reconnectAttemptRef.current = 0;
      setIsConnected(true);
      setStatus({ status: 'connected', reconnectAttempt: 0 });
      onOpen?.();
    };

    ws.onmessage = event => {
      if (isUnmountedRef.current) return;

      try {
        const message: PriceUpdate = JSON.parse(event.data);

        if (message.type === 'price_update') {
          // Buffer hinzufügen
          bufferRef.current.push(message.data);

          // Buffer-Size limitieren um Memory Leaks zu vermeiden
          if (bufferRef.current.length > maxBufferSize) {
            bufferRef.current = bufferRef.current.slice(-maxBufferSize);
          }
        }
      } catch (parseError) {
        console.error('WebSocket parse error:', parseError);
      }
    };

    ws.onerror = error => {
      console.error('WebSocket error:', error);
      onError?.(error);
    };

    ws.onclose = event => {
      if (isUnmountedRef.current) return;

      setIsConnected(false);
      onClose?.();

      // Auto-Reconnect wenn gewünscht
      if (reconnect && !event.wasClean && reconnectAttemptRef.current < maxReconnectAttempts) {
        // Exponential Backoff berechnen
        const delay = Math.min(1000 * Math.pow(2, reconnectAttemptRef.current), 30000);

        reconnectAttemptRef.current++;

        setStatus(prev => ({
          ...prev,
          status: 'reconnecting',
          reconnectAttempt: reconnectAttemptRef.current,
        }));

        reconnectTimeoutRef.current = setTimeout(() => {
          connect();
        }, delay);
      } else {
        setStatus(prev => ({ ...prev, status: 'disconnected' }));
      }
    };

    wsRef.current = ws;
  }, [reconnect, maxReconnectAttempts, maxBufferSize, onOpen, onClose, onError]);

  const disconnect = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
    }

    if (wsRef.current) {
      wsRef.current.close(1000, 'Manual disconnect');
      wsRef.current = null;
    }

    setIsConnected(false);
    setStatus({ status: 'disconnected', reconnectAttempt: 0 });
  }, []);

  const sendMessage = useCallback((message: object) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(message));
    }
  }, []);

  const reconnect = useCallback(() => {
    reconnectAttemptRef.current = 0;
    disconnect();
    connect();
  }, [connect, disconnect]);

  useEffect(() => {
    isUnmountedRef.current = false;
    connect();

    return () => {
      isUnmountedRef.current = true;
      disconnect();
    };
  }, [connect, disconnect]);

  return {
    isConnected,
    lastPrice,
    status,
    sendMessage,
    reconnect,
    disconnect,
  };
}

// Specialized hook for price subscriptions
export function usePriceUpdates(symbols: string[] = ['BTCUSDT']) {
  const { isConnected, lastPrice, status, sendMessage, reconnect, disconnect } = useWebSocket();

  // Subscribe to symbols when connected
  useEffect(() => {
    if (isConnected && symbols.length > 0) {
      sendMessage({
        type: 'subscribe',
        symbols,
      });
    }
  }, [isConnected, symbols, sendMessage]);

  return {
    isConnected,
    price: lastPrice,
    status,
    reconnect,
    disconnect,
  };
}

export default useWebSocket;
