import { useState, useEffect } from 'react';
import { Card, Button } from '../components/ui';
import { api } from '../api/client';
import { Brain, Upload, Download, Trash2, Play, Clock, HardDrive, TrendingUp, AlertCircle } from 'lucide-react';

// Local model type matching the API response from client.ts
interface Model {
  id: number;
  name: string;
  type: string;
  created: string;
  size: string;
  status: 'active' | 'trained' | 'not_trained';
  // sharpe is optional — the API may not return it for all models
  sharpe?: number;
  source: string;
}

// Training history entry matching the TrainingJob type from client.ts
interface TrainingJob {
  id: string;
  modelName: string;
  status: 'pending' | 'running' | 'completed' | 'failed';
  progress: number;
  startTime: string;
  endTime?: string;
  metrics?: {
    loss: number;
    reward: number;
    sharpe?: number;
  };
}

export function Models() {
  const [models, setModels] = useState<Model[]>([]);
  const [trainingHistory, setTrainingHistory] = useState<TrainingJob[]>([]);
  const [isTraining, setIsTraining] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchData = async () => {
      setError(null);
      try {
        const [modelsData, historyData] = await Promise.all([
          api.models.list(),
          api.models.getTrainingHistory(),
        ]);
        setModels(modelsData as unknown as Model[]);
        setTrainingHistory(historyData as unknown as TrainingJob[]);
      } catch (e) {
        const msg = (e as Error).message || 'Failed to load models';
        console.error('[Models] fetch failed:', e);
        setError(msg);
      }
    };
    fetchData();
  }, []);

  const handleTrain = async () => {
    setIsTraining(true);
    setError(null);
    try {
      await api.models.train();
      const modelsData = await api.models.list();
      setModels(modelsData as unknown as Model[]);
    } catch (e) {
      const msg = (e as Error).message || 'Training failed';
      console.error('[Models] train failed:', e);
      setError(msg);
    }
    setIsTraining(false);
  };

  const handleDelete = async (id: number) => {
    try {
      await api.models.delete(id);
      setModels(models.filter(m => m.id !== id));
    } catch (e) {
      const msg = (e as Error).message || 'Delete failed';
      console.error('[Models] delete failed:', e);
      setError(msg);
    }
  };

  const totalModels = models.length;
  // parseFloat handles size strings like "1.2" — non-numeric values default to 0
  const totalSize = models.reduce((acc, m) => acc + (parseFloat(m.size) || 0), 0);
  // Guard against empty array and undefined sharpe values
  const bestSharpe =
    models.length > 0
      ? Math.max(...models.map(m => m.sharpe ?? 0))
      : 0;

  return (
    <div className="space-y-6">
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold text-text-primary">Models</h1>
          <p className="text-text-secondary">Manage your trained RL models</p>
        </div>
        <div className="flex items-center gap-3">
          <Button variant="secondary">
            <Upload size={18} className="mr-2" />
            Import Model
          </Button>
          <Button onClick={handleTrain} disabled={isTraining}>
            <Brain size={18} className="mr-2" />
            {isTraining ? 'Training...' : 'Train New Model'}
          </Button>
        </div>
      </div>

      {/* Error banner */}
      {error && (
        <div className="flex items-center gap-2 p-3 rounded-lg bg-red-500/10 border border-red-500/30 text-red-400 text-sm">
          <AlertCircle size={16} className="shrink-0" />
          <span>{error}</span>
        </div>
      )}

      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card className="text-center">
          <Brain size={32} className="mx-auto mb-2 text-bitcoin-orange" />
          <p className="text-2xl font-bold text-text-primary">{totalModels}</p>
          <p className="text-sm text-text-secondary">Total Models</p>
        </Card>
        <Card className="text-center">
          <HardDrive size={32} className="mx-auto mb-2 text-blue-400" />
          <p className="text-2xl font-bold text-text-primary">{totalSize.toFixed(1)} GB</p>
          <p className="text-sm text-text-secondary">Storage Used</p>
        </Card>
        <Card className="text-center">
          <Clock size={32} className="mx-auto mb-2 text-green-400" />
          <p className="text-2xl font-bold text-text-primary">12h</p>
          <p className="text-sm text-text-secondary">Avg Training Time</p>
        </Card>
        <Card className="text-center">
          <TrendingUp size={32} className="mx-auto mb-2 text-purple-400" />
          <p className="text-2xl font-bold text-text-primary">{bestSharpe.toFixed(2)}</p>
          <p className="text-sm text-text-secondary">Best Sharpe</p>
        </Card>
      </div>

      <Card title="Model Library">
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="border-b border-border">
                <th className="text-left py-3 px-4 text-sm font-medium text-text-secondary">Name</th>
                <th className="text-left py-3 px-4 text-sm font-medium text-text-secondary">Type</th>
                <th className="text-left py-3 px-4 text-sm font-medium text-text-secondary">Created</th>
                <th className="text-left py-3 px-4 text-sm font-medium text-text-secondary">Size</th>
                <th className="text-left py-3 px-4 text-sm font-medium text-text-secondary">Sharpe</th>
                <th className="text-left py-3 px-4 text-sm font-medium text-text-secondary">Status</th>
                <th className="text-right py-3 px-4 text-sm font-medium text-text-secondary">Actions</th>
              </tr>
            </thead>
            <tbody>
              {models.map((model) => (
                <tr key={model.id} className="border-b border-border/50 hover:bg-background/50">
                  <td className="py-3 px-4">
                    <div className="flex items-center gap-2">
                      <Brain size={18} className="text-bitcoin-orange" />
                      <span className="text-text-primary font-medium">{model.name}</span>
                    </div>
                  </td>
                  <td className="py-3 px-4 text-sm text-text-secondary">{model.type}</td>
                  <td className="py-3 px-4 text-sm text-text-secondary">{model.created}</td>
                  <td className="py-3 px-4 text-sm text-text-secondary">{model.size}</td>
                  {/* sharpe is optional — show '—' when not available */}
                  <td className="py-3 px-4 text-sm text-green-400">
                    {model.sharpe != null ? model.sharpe.toFixed(2) : '—'}
                  </td>
                  <td className="py-3 px-4">
                    <span
                      className={`px-2 py-1 text-xs font-medium rounded ${
                        model.status === 'active'
                          ? 'bg-green-500/10 text-green-400'
                          : model.status === 'trained'
                          ? 'bg-blue-500/10 text-blue-400'
                          : 'bg-gray-500/10 text-gray-400'
                      }`}
                    >
                      {model.status}
                    </span>
                  </td>
                  <td className="py-3 px-4">
                    <div className="flex items-center justify-end gap-2">
                      <Button variant="ghost" size="sm">
                        <Play size={16} />
                      </Button>
                      <Button variant="ghost" size="sm">
                        <Download size={16} />
                      </Button>
                      <Button
                        variant="ghost"
                        size="sm"
                        className="text-red-400 hover:text-red-300"
                        onClick={() => handleDelete(model.id)}
                      >
                        <Trash2 size={16} />
                      </Button>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      <Card title="Training History">
        <div className="space-y-3">
          {trainingHistory.map((job) => (
            <div key={job.id} className="flex items-center justify-between p-3 bg-background rounded-lg">
              <div>
                <p className="text-text-primary font-medium">{job.modelName}</p>
                <p className="text-text-secondary text-sm">
                  {job.startTime}
                  {job.endTime ? ` — ${job.endTime}` : ' — in progress'}
                </p>
              </div>
              <div className="text-right">
                {/* Show reward metric if available, otherwise status */}
                <p className="text-green-400 text-sm">
                  {job.metrics?.reward != null
                    ? `Reward: ${job.metrics.reward.toFixed(3)}`
                    : job.status}
                </p>
                <p className="text-text-muted text-xs">{job.status}</p>
              </div>
            </div>
          ))}
          {trainingHistory.length === 0 && (
            <p className="text-text-secondary text-sm text-center py-4">No training history yet</p>
          )}
        </div>
      </Card>
    </div>
  );
}
