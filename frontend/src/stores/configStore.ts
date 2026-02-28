import { create } from 'zustand';
import { BotConfig, RiskConfig, DataConfig } from '../types';

interface ConfigStore {
  botConfig: BotConfig;
  riskConfig: RiskConfig;
  dataConfig: DataConfig;
  isSaving: boolean;
  setBotConfig: (config: Partial<BotConfig>) => void;
  setRiskConfig: (config: Partial<RiskConfig>) => void;
  setDataConfig: (config: Partial<DataConfig>) => void;
  setIsSaving: (isSaving: boolean) => void;
  saveConfig: () => Promise<void>;
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

export const useConfigStore = create<ConfigStore>((set) => ({
  botConfig: defaultBotConfig,
  riskConfig: defaultRiskConfig,
  dataConfig: defaultDataConfig,
  isSaving: false,
  setBotConfig: (config) =>
    set((state) => ({ botConfig: { ...state.botConfig, ...config } })),
  setRiskConfig: (config) =>
    set((state) => ({ riskConfig: { ...state.riskConfig, ...config } })),
  setDataConfig: (config) =>
    set((state) => ({ dataConfig: { ...state.dataConfig, ...config } })),
  setIsSaving: (isSaving) => set({ isSaving }),
  saveConfig: async () => {
    set({ isSaving: true });
    await new Promise((resolve) => setTimeout(resolve, 1000));
    set({ isSaving: false });
  },
}));
