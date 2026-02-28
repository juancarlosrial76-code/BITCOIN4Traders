import { create } from 'zustand';
import { PerformanceMetrics, EquityCurvePoint } from '../types';

interface AnalyticsStore {
  metrics: PerformanceMetrics | null;
  equityCurve: EquityCurvePoint[];
  isLoading: boolean;
  setMetrics: (metrics: PerformanceMetrics) => void;
  setEquityCurve: (curve: EquityCurvePoint[]) => void;
  setIsLoading: (isLoading: boolean) => void;
}

const defaultMetrics: PerformanceMetrics = {
  totalReturn: 0,
  sharpeRatio: 0,
  sortinoRatio: 0,
  calmarRatio: 0,
  maxDrawdown: 0,
  maxDrawdownDuration: 0,
  winRate: 0,
  profitFactor: 0,
  totalTrades: 0,
  winningTrades: 0,
  losingTrades: 0,
  avgWin: 0,
  avgLoss: 0,
  largestWin: 0,
  largestLoss: 0,
  avgHoldingPeriod: 0,
};

export const useAnalyticsStore = create<AnalyticsStore>((set) => ({
  metrics: defaultMetrics,
  equityCurve: [],
  isLoading: false,
  setMetrics: (metrics) => set({ metrics }),
  setEquityCurve: (equityCurve) => set({ equityCurve }),
  setIsLoading: (isLoading) => set({ isLoading }),
}));
