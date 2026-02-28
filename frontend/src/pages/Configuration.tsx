import { useState, useEffect } from 'react';
import { Card, Button, Input, Select } from '../components/ui';
import { api } from '../api/client';
import { Save, RotateCcw } from 'lucide-react';

export function Configuration() {
  const [botConfig, setBotConfig] = useState({
    symbol: 'BTCUSDT',
    timeframe: '1h',
    maxPositions: 3,
    agentType: 'PPO',
    modelPath: 'models/latest',
  });
  const [riskConfig, setRiskConfig] = useState({
    maxDrawdown: 0.2,
    stopLoss: 0.02,
    takeProfit: 0.05,
    positionSizePercent: 0.1,
  });
  const [dataConfig, setDataConfig] = useState({
    dataSource: 'binance',
    startDate: '2023-01-01',
    endDate: '2024-01-01',
    trainTestSplit: 0.8,
  });
  const [paperTrading, setPaperTrading] = useState(true);
  const [autoRebalance, setAutoRebalance] = useState(false);
  const [notifications, setNotifications] = useState(true);
  const [isSaving, setIsSaving] = useState(false);

  useEffect(() => {
    const fetchConfig = async () => {
      try {
        const config = await api.config.get();
        setBotConfig(config.bot);
        setRiskConfig(config.risk);
        setDataConfig(config.data);
      } catch (e) {
        console.error('Failed to fetch config:', e);
      }
    };
    fetchConfig();
  }, []);

  const handleSave = async () => {
    setIsSaving(true);
    try {
      await api.config.updateBot(botConfig);
      await api.config.updateRisk(riskConfig);
      await api.config.updateData(dataConfig);
    } catch (e) {
      console.error('Failed to save config:', e);
    }
    setIsSaving(false);
  };

  return (
    <div className="space-y-6">
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold text-text-primary">Configuration</h1>
          <p className="text-text-secondary">Configure your trading bot parameters</p>
        </div>
        <div className="flex items-center gap-3">
          <Button variant="secondary" onClick={() => window.location.reload()}>
            <RotateCcw size={18} className="mr-2" />
            Reset
          </Button>
          <Button onClick={handleSave} disabled={isSaving}>
            <Save size={18} className="mr-2" />
            {isSaving ? 'Saving...' : 'Save Changes'}
          </Button>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card title="Bot Settings">
          <div className="space-y-4">
            <Select
              label="Trading Pair"
              value={botConfig.symbol}
              onChange={(e) => setBotConfig({ ...botConfig, symbol: e.target.value })}
              options={[
                { value: 'BTCUSDT', label: 'BTC/USDT' },
                { value: 'ETHUSDT', label: 'ETH/USDT' },
                { value: 'SOLUSDT', label: 'SOL/USDT' },
              ]}
            />
            <Select
              label="Timeframe"
              value={botConfig.timeframe}
              onChange={(e) => setBotConfig({ ...botConfig, timeframe: e.target.value })}
              options={[
                { value: '1m', label: '1 Minute' },
                { value: '5m', label: '5 Minutes' },
                { value: '15m', label: '15 Minutes' },
                { value: '1h', label: '1 Hour' },
                { value: '4h', label: '4 Hours' },
                { value: '1d', label: '1 Day' },
              ]}
            />
            <Input
              label="Max Positions"
              type="number"
              value={botConfig.maxPositions}
              onChange={(e) => setBotConfig({ ...botConfig, maxPositions: parseInt(e.target.value) })}
            />
            <Select
              label="Agent Type"
              value={botConfig.agentType}
              onChange={(e) => setBotConfig({ ...botConfig, agentType: e.target.value })}
              options={[
                { value: 'PPO', label: 'PPO (Proximal Policy Optimization)' },
                { value: 'A2C', label: 'A2C (Advantage Actor Critic)' },
                { value: 'DQN', label: 'DQN (Deep Q Network)' },
                { value: 'TD3', label: 'TD3 (Twin Delayed DDPG)' },
              ]}
            />
            <Input
              label="Model Path"
              value={botConfig.modelPath}
              onChange={(e) => setBotConfig({ ...botConfig, modelPath: e.target.value })}
            />
          </div>
        </Card>

        <Card title="Risk Management">
          <div className="space-y-4">
            <Input
              label="Max Drawdown (%)"
              type="number"
              step="0.01"
              value={riskConfig.maxDrawdown * 100}
              onChange={(e) => setRiskConfig({ ...riskConfig, maxDrawdown: parseFloat(e.target.value) / 100 })}
            />
            <Input
              label="Stop Loss (%)"
              type="number"
              step="0.01"
              value={riskConfig.stopLoss * 100}
              onChange={(e) => setRiskConfig({ ...riskConfig, stopLoss: parseFloat(e.target.value) / 100 })}
            />
            <Input
              label="Take Profit (%)"
              type="number"
              step="0.01"
              value={riskConfig.takeProfit * 100}
              onChange={(e) => setRiskConfig({ ...riskConfig, takeProfit: parseFloat(e.target.value) / 100 })}
            />
            <Input
              label="Position Size (%)"
              type="number"
              step="0.01"
              value={riskConfig.positionSizePercent * 100}
              onChange={(e) => setRiskConfig({ ...riskConfig, positionSizePercent: parseFloat(e.target.value) / 100 })}
            />
          </div>
        </Card>

        <Card title="Data Settings">
          <div className="space-y-4">
            <Select
              label="Data Source"
              value={dataConfig.dataSource}
              onChange={(e) => setDataConfig({ ...dataConfig, dataSource: e.target.value })}
              options={[
                { value: 'binance', label: 'Binance' },
                { value: 'coinbase', label: 'Coinbase' },
                { value: 'kraken', label: 'Kraken' },
                { value: 'csv', label: 'CSV File' },
              ]}
            />
            <Input
              label="Start Date"
              type="date"
              value={dataConfig.startDate}
              onChange={(e) => setDataConfig({ ...dataConfig, startDate: e.target.value })}
            />
            <Input
              label="End Date"
              type="date"
              value={dataConfig.endDate}
              onChange={(e) => setDataConfig({ ...dataConfig, endDate: e.target.value })}
            />
            <Input
              label="Train/Test Split"
              type="number"
              step="0.05"
              min="0.5"
              max="0.9"
              value={dataConfig.trainTestSplit}
              onChange={(e) => setDataConfig({ ...dataConfig, trainTestSplit: parseFloat(e.target.value) })}
            />
          </div>
        </Card>

        <Card title="Advanced Settings">
          <div className="space-y-4">
            <div className="flex items-center justify-between p-3 bg-background rounded-lg">
              <div>
                <p className="text-text-primary font-medium">Paper Trading</p>
                <p className="text-text-secondary text-sm">Trade with simulated money</p>
              </div>
              <button 
                className={`w-12 h-6 rounded-full relative transition-colors ${paperTrading ? 'bg-green-500' : 'bg-gray-600'}`}
                onClick={() => setPaperTrading(!paperTrading)}
              >
                <span className={`absolute top-1 w-4 h-4 bg-white rounded-full transition-transform ${paperTrading ? 'right-1' : 'left-1'}`}></span>
              </button>
            </div>
            <div className="flex items-center justify-between p-3 bg-background rounded-lg">
              <div>
                <p className="text-text-primary font-medium">Auto Rebalance</p>
                <p className="text-text-secondary text-sm">Automatically rebalance portfolio</p>
              </div>
              <button 
                className={`w-12 h-6 rounded-full relative transition-colors ${autoRebalance ? 'bg-green-500' : 'bg-gray-600'}`}
                onClick={() => setAutoRebalance(!autoRebalance)}
              >
                <span className={`absolute top-1 w-4 h-4 bg-white rounded-full transition-transform ${autoRebalance ? 'right-1' : 'left-1'}`}></span>
              </button>
            </div>
            <div className="flex items-center justify-between p-3 bg-background rounded-lg">
              <div>
                <p className="text-text-primary font-medium">Notifications</p>
                <p className="text-text-secondary text-sm">Receive alerts for trades</p>
              </div>
              <button 
                className={`w-12 h-6 rounded-full relative transition-colors ${notifications ? 'bg-green-500' : 'bg-gray-600'}`}
                onClick={() => setNotifications(!notifications)}
              >
                <span className={`absolute top-1 w-4 h-4 bg-white rounded-full transition-transform ${notifications ? 'right-1' : 'left-1'}`}></span>
              </button>
            </div>
          </div>
        </Card>
      </div>
    </div>
  );
}
