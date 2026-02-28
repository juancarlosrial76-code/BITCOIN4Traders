import { create } from 'zustand';
import { Position, Order, TradingConfig } from '../types';
import { api } from '../api/client';

interface TradingStore {
  isRunning: boolean;
  positions: Position[];
  orders: Order[];
  config: TradingConfig;
  currentPrice: number;
  priceHistory: { timestamp: number; price: number }[];
  isLoading: boolean;
  error: string | null;
  setIsRunning: (isRunning: boolean) => void;
  setPositions: (positions: Position[]) => void;
  setOrders: (orders: Order[]) => void;
  setConfig: (config: Partial<TradingConfig>) => void;
  setCurrentPrice: (price: number) => void;
  addPricePoint: (timestamp: number, price: number) => void;
  fetchStatus: () => Promise<void>;
  startTrading: () => Promise<void>;
  stopTrading: () => Promise<void>;
  fetchOrders: () => Promise<void>;
}

export const useTradingStore = create<TradingStore>((set) => ({
  isRunning: false,
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
  isLoading: false,
  error: null,
  setIsRunning: (isRunning) => set({ isRunning }),
  setPositions: (positions) => set({ positions }),
  setOrders: (orders) => set({ orders }),
  setConfig: (config) =>
    set((state) => ({ config: { ...state.config, ...config } })),
  setCurrentPrice: (currentPrice) => set({ currentPrice }),
  addPricePoint: (timestamp, price) =>
    set((state) => ({
      priceHistory: [...state.priceHistory.slice(-100), { timestamp, price }],
    })),
  fetchStatus: async () => {
    set({ isLoading: true, error: null });
    try {
      const status = await api.trading.getStatus();
      set({
        isRunning: status.is_running,
        currentPrice: status.current_position * 43000,
        isLoading: false,
      });
    } catch (e) {
      set({ error: (e as Error).message, isLoading: false });
    }
  },
  startTrading: async () => {
    set({ isLoading: true, error: null });
    try {
      await api.trading.start();
      set({ isRunning: true, isLoading: false });
    } catch (e) {
      set({ error: (e as Error).message, isLoading: false });
    }
  },
  stopTrading: async () => {
    set({ isLoading: true, error: null });
    try {
      await api.trading.stop();
      set({ isRunning: false, isLoading: false });
    } catch (e) {
      set({ error: (e as Error).message, isLoading: false });
    }
  },
  fetchOrders: async () => {
    try {
      const orders = await api.trading.getOrders();
      set({ orders: orders as Order[] });
    } catch (e) {
      set({ error: (e as Error).message });
    }
  },
}));
