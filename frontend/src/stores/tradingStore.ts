import { create } from 'zustand';
import { devtools, persist, subscribeWithSelector } from 'zustand/middleware';
import { api } from '../api/client';

interface Position {
  id: string;
  symbol: string;
  side: 'long' | 'short';
  quantity: number;
  entryPrice: number;
  currentPrice: number;
  pnl: number;
  pnlPercent: number;
  openedAt: string;
}

interface Order {
  id: string;
  symbol: string;
  side: 'buy' | 'sell';
  type: 'market' | 'limit';
  quantity: number;
  price?: number;
  status: 'pending' | 'filled' | 'cancelled';
  createdAt: string;
}

interface TradingConfig {
  maxPositionSize: number;
  stopLoss: number;
  takeProfit: number;
  riskPerTrade: number;
  leverage: number;
}

interface TradingState {
  isRunning: boolean;
  mode: 'live' | 'paper';
  positions: Position[];
  orders: Order[];
  config: TradingConfig;
  currentPrice: number;
  priceHistory: { timestamp: number; price: number }[];
  balance: { USDT: number; BTC: number; totalUSD: number };
  isLoading: boolean;
  error: string | null;
  lastUpdated: number | null;
}

interface TradingActions {
  setIsRunning: (isRunning: boolean) => void;
  setMode: (mode: 'live' | 'paper') => void;
  setPositions: (positions: Position[]) => void;
  setOrders: (orders: Order[]) => void;
  setConfig: (config: Partial<TradingConfig>) => void;
  setCurrentPrice: (price: number) => void;
  setBalance: (balance: { USDT: number; BTC: number; totalUSD: number }) => void;
  addPricePoint: (timestamp: number, price: number) => void;
  setError: (error: string | null) => void;
  clearError: () => void;
  fetchStatus: () => Promise<void>;
  fetchBalance: () => Promise<void>;
  startTrading: (mode?: 'live' | 'paper') => Promise<void>;
  stopTrading: () => Promise<void>;
  placeOrder: (order: PlaceOrderRequest) => Promise<void>;
  fetchOrders: () => Promise<void>;
  fetchConfig: () => Promise<void>;
  reset: () => void;
}

type TradingStore = TradingState & TradingActions;

interface PlaceOrderRequest {
  symbol: string;
  side: 'buy' | 'sell';
  type: 'market' | 'limit';
  quantity: number;
  price?: number;
}

const initialState: TradingState = {
  isRunning: false,
  mode: 'paper',
  positions: [],
  orders: [],
  config: {
    maxPositionSize: 0.1,
    stopLoss: 0.02,
    takeProfit: 0.05,
    riskPerTrade: 0.02,
    leverage: 1,
  },
  currentPrice: 0,
  priceHistory: [],
  balance: { USDT: 10000, BTC: 0, totalUSD: 10000 },
  isLoading: false,
  error: null,
  lastUpdated: null,
};

export const useTradingStore = create<TradingStore>()(
  subscribeWithSelector(
    devtools(
      persist(
        (set, get) => ({
          ...initialState,

          setIsRunning: isRunning => set({ isRunning }, false, 'setIsRunning'),
          setMode: mode => set({ mode }, false, 'setMode'),
          setPositions: positions =>
            set({ positions, lastUpdated: Date.now() }, false, 'setPositions'),
          setOrders: orders => set({ orders, lastUpdated: Date.now() }, false, 'setOrders'),
          setConfig: config =>
            set(
              state => ({
                config: { ...state.config, ...config },
              }),
              false,
              'setConfig'
            ),
          setCurrentPrice: currentPrice => set({ currentPrice }, false, 'setCurrentPrice'),
          setBalance: balance => set({ balance }, false, 'setBalance'),

          addPricePoint: (timestamp, price) =>
            set(
              state => ({
                priceHistory: [...state.priceHistory.slice(-200), { timestamp, price }],
                currentPrice: price,
              }),
              false,
              'addPricePoint'
            ),

          setError: error => set({ error }, false, 'setError'),
          clearError: () => set({ error: null }, false, 'clearError'),

          fetchStatus: async () => {
            set({ isLoading: true, error: null }, false, 'fetchStatus');
            try {
              const status = await api.trading.getStatus();
              set(
                {
                  isRunning: status.is_running,
                  currentPrice: status.current_position * 43000,
                  isLoading: false,
                  lastUpdated: Date.now(),
                },
                false,
                'fetchStatusSuccess'
              );
            } catch (e) {
              set(
                {
                  error: (e as Error).message,
                  isLoading: false,
                },
                false,
                'fetchStatusError'
              );
            }
          },

          fetchBalance: async () => {
            try {
              const balance = await api.trading.getBalance();
              set({ balance }, false, 'fetchBalance');
            } catch (e) {
              console.error('Failed to fetch balance:', e);
            }
          },

          startTrading: async (mode = 'paper') => {
            set({ isLoading: true, error: null }, false, 'startTrading');
            try {
              await api.trading.start();
              set(
                {
                  isRunning: true,
                  mode,
                  isLoading: false,
                  lastUpdated: Date.now(),
                },
                false,
                'startTradingSuccess'
              );
            } catch (e) {
              set(
                {
                  error: (e as Error).message,
                  isLoading: false,
                },
                false,
                'startTradingError'
              );
            }
          },

          stopTrading: async () => {
            set({ isLoading: true, error: null }, false, 'stopTrading');
            try {
              await api.trading.stop();
              set(
                {
                  isRunning: false,
                  isLoading: false,
                  lastUpdated: Date.now(),
                },
                false,
                'stopTradingSuccess'
              );
            } catch (e) {
              set(
                {
                  error: (e as Error).message,
                  isLoading: false,
                },
                false,
                'stopTradingError'
              );
            }
          },

          placeOrder: async order => {
            set({ isLoading: true, error: null }, false, 'placeOrder');
            try {
              await api.trading.placeOrder(order);
              set({ isLoading: false }, false, 'placeOrderSuccess');
              get().fetchOrders();
            } catch (e) {
              set(
                {
                  error: (e as Error).message,
                  isLoading: false,
                },
                false,
                'placeOrderError'
              );
            }
          },

          fetchOrders: async () => {
            try {
              const orders = await api.trading.getOrders();
              set({ orders: orders as Order[], lastUpdated: Date.now() }, false, 'fetchOrders');
            } catch (e) {
              console.error('Failed to fetch orders:', e);
            }
          },

          fetchConfig: async () => {
            try {
              const config = await api.trading.getConfig();
              set({ config: config as TradingConfig }, false, 'fetchConfig');
            } catch (e) {
              console.error('Failed to fetch config:', e);
            }
          },

          reset: () => set(initialState, false, 'reset'),
        }),
        {
          name: 'trading-storage',
          partialize: state => ({
            mode: state.mode,
            config: state.config,
          }),
        }
      ),
      { name: 'TradingStore' }
    )
  )
);

export const selectIsRunning = (state: TradingStore) => state.isRunning;
export const selectMode = (state: TradingStore) => state.mode;
export const selectPositions = (state: TradingStore) => state.positions;
export const selectOrders = (state: TradingStore) => state.orders;
export const selectConfig = (state: TradingStore) => state.config;
export const selectBalance = (state: TradingStore) => state.balance;
export const selectError = (state: TradingStore) => state.error;
export const selectIsLoading = (state: TradingStore) => state.isLoading;
export const selectCurrentPrice = (state: TradingStore) => state.currentPrice;
export const selectPriceHistory = (state: TradingStore) => state.priceHistory;
