import { useState, useEffect } from 'react';
import { Card, Button } from '../components/ui';
import { api } from '../api/client';
import { Brain, Upload, Download, Trash2, Play, Clock, HardDrive } from 'lucide-react';

export function Models() {
  const [models, setModels] = useState<any[]>([]);
  const [trainingHistory, setTrainingHistory] = useState<any[]>([]);
  const [isTraining, setIsTraining] = useState(false);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const [modelsData, historyData] = await Promise.all([
          api.models.list(),
          api.models.getTrainingHistory(),
        ]);
        setModels(modelsData);
        setTrainingHistory(historyData);
      } catch (e) {
        console.error('Failed to fetch models:', e);
      }
    };
    fetchData();
  }, []);

  const handleTrain = async () => {
    setIsTraining(true);
    try {
      await api.models.train();
      const modelsData = await api.models.list();
      setModels(modelsData);
    } catch (e) {
      console.error('Failed to train model:', e);
    }
    setIsTraining(false);
  };

  const handleDelete = async (id: number) => {
    try {
      await api.models.delete(id);
      setModels(models.filter(m => m.id !== id));
    } catch (e) {
      console.error('Failed to delete model:', e);
    }
  };

  const totalModels = models.length;
  const totalSize = models.reduce((acc, m) => acc + parseFloat(m.size), 0);

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
          <TrendingUpIcon size={32} className="mx-auto mb-2 text-purple-400" />
          <p className="text-2xl font-bold text-text-primary">
            {models.length > 0 ? Math.max(...models.map(m => m.sharpe)).toFixed(2) : '0'}
          </p>
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
                  <td className="py-3 px-4 text-sm text-green-400">{model.sharpe.toFixed(2)}</td>
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
          {trainingHistory.map((training, i) => (
            <div key={i} className="flex items-center justify-between p-3 bg-background rounded-lg">
              <div>
                <p className="text-text-primary font-medium">{training.model}</p>
                <p className="text-text-secondary text-sm">
                  {training.start} - {training.end}
                </p>
              </div>
              <div className="text-right">
                <p className="text-green-400 text-sm">{training.result}</p>
                <p className="text-text-muted text-xs">{training.status}</p>
              </div>
            </div>
          ))}
        </div>
      </Card>
    </div>
  );
}

function TrendingUpIcon({ size, className }: { size: number; className?: string }) {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      className={className}
    >
      <polyline points="23 6 13.5 15.5 8.5 10.5 1 18" />
      <polyline points="17 6 23 6 23 12" />
    </svg>
  );
}
