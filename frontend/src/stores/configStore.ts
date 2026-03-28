import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import { BotConfig, RiskConfig, DataConfig } from '../types';
import { api } from '../api/client';

interface ConfigStore {
  botConfig: BotConfig;
  riskConfig: RiskConfig;
  dataConfig: DataConfig;
  isSaving: boolean;
  // saveError is set when saveConfig() fails, cleared on next save attempt
  saveError: string | null;
  setBotConfig: (config: Partial<BotConfig>) => void;
  setRiskConfig: (config: Partial<RiskConfig>) => void;
  setDataConfig: (config: Partial<DataConfig>) => void;
  setIsSaving: (isSaving: boolean) => void;
  saveConfig: () => Promise<void>;
  clearSaveError: () => void;
}

const defaultBotConfig: BotConfig = {
  symbol: 'BTCUSDT',
  timeframe: '1h',
  maxPositions: 3,
  agentType: 'PPO',
  modelPath: 'models/latest',
};

const defaultRiskConfig: RiskConfig = {
  maxDrawdown: 0.2,
  stopLoss: 0.02,
  takeProfit: 0.05,
  positionSizePercent: 0.1,
};

const defaultDataConfig: DataConfig = {
  dataSource: 'binance',
  startDate: '2023-01-01',
  endDate: '2024-01-01',
  trainTestSplit: 0.8,
};

export const useConfigStore = create<ConfigStore>()(
  persist(
    (set, get) => ({
      botConfig: defaultBotConfig,
      riskConfig: defaultRiskConfig,
      dataConfig: defaultDataConfig,
      isSaving: false,
      saveError: null,

      setBotConfig: (config) =>
        set((state) => ({ botConfig: { ...state.botConfig, ...config } })),
      setRiskConfig: (config) =>
        set((state) => ({ riskConfig: { ...state.riskConfig, ...config } })),
      setDataConfig: (config) =>
        set((state) => ({ dataConfig: { ...state.dataConfig, ...config } })),
      setIsSaving: (isSaving) => set({ isSaving }),
      clearSaveError: () => set({ saveError: null }),

      saveConfig: async () => {
        set({ isSaving: true, saveError: null });
        const state = get();
        try {
          await Promise.all([
            api.config.updateBot(state.botConfig),
            api.config.updateRisk(state.riskConfig),
            api.config.updateData(state.dataConfig),
          ]);
        } catch (e) {
          // Expose error so the UI can display it (FE-032)
          const msg = (e as Error).message || 'Failed to save configuration';
          console.error('[configStore] saveConfig failed:', e);
          set({ saveError: msg });
        } finally {
          set({ isSaving: false });
        }
      },
    }),
    {
      name: 'config-storage', // persisted in localStorage (FE-030)
      partialize: (state) => ({
        botConfig: state.botConfig,
        riskConfig: state.riskConfig,
        dataConfig: state.dataConfig,
      }),
    }
  )
);
