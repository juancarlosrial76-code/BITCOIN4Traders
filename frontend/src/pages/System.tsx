import { useState, useEffect } from 'react';
import { Card, Button } from '../components/ui';
import { api } from '../api/client';
import { Cpu, HardDrive, Activity, RefreshCw, Filter } from 'lucide-react';

export function System() {
  const [metrics, setMetrics] = useState<any>(null);
  const [logs, setLogs] = useState<any[]>([]);
  const [endpoints, setEndpoints] = useState<any[]>([]);
  const [envVars, setEnvVars] = useState<any[]>([]);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const [metricsData, logsData, endpointsData, envData] = await Promise.all([
          api.system.getMetrics(),
          api.system.getLogs(),
          api.system.getEndpoints(),
          api.system.getEnv(),
        ]);
        setMetrics(metricsData);
        setLogs(logsData);
        setEndpoints(endpointsData);
        setEnvVars(envData);
      } catch (e) {
        console.error('Failed to fetch system data:', e);
      }
    };
    fetchData();
    const interval = setInterval(fetchData, 10000);
    return () => clearInterval(interval);
  }, []);

  const handleRefresh = async () => {
    try {
      const [metricsData, logsData] = await Promise.all([
        api.system.getMetrics(),
        api.system.getLogs(),
      ]);
      setMetrics(metricsData);
      setLogs(logsData);
    } catch (e) {
      console.error('Failed to refresh:', e);
    }
  };

  const systemMetrics = [
    { label: 'CPU Usage', value: `${metrics?.cpu_usage || 0}%`, icon: Cpu },
    { label: 'Memory', value: metrics?.memory || '0 GB', icon: HardDrive },
    { label: 'Latency', value: `${metrics?.latency || 0}ms`, icon: Activity },
    { label: 'Uptime', value: metrics?.uptime || '0d 0h', icon: Activity },
  ];

  return (
    <div className="space-y-6">
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold text-text-primary">System</h1>
          <p className="text-text-secondary">Monitor system health and logs</p>
        </div>
        <div className="flex items-center gap-3">
          <Button variant="secondary" size="sm" onClick={handleRefresh}>
            <RefreshCw size={16} className="mr-2" />
            Refresh
          </Button>
        </div>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        {systemMetrics.map((metric) => {
          const Icon = metric.icon;
          return (
            <Card key={metric.label}>
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-text-secondary">{metric.label}</p>
                  <p className="text-2xl font-bold text-text-primary">{metric.value}</p>
                </div>
                <Icon size={24} className="text-bitcoin-orange" />
              </div>
            </Card>
          );
        })}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card
          title="System Logs"
          action={
            <Button variant="ghost" size="sm">
              <Filter size={16} />
            </Button>
          }
        >
          <div className="bg-black rounded-lg p-4 font-mono text-sm h-80 overflow-y-auto">
            {logs.map((log, i) => (
              <div key={i} className="mb-1">
                <span className="text-text-muted">[{log.time}]</span>{' '}
                <span
                  className={
                    log.level === 'ERROR'
                      ? 'text-red-400'
                      : log.level === 'WARN'
                      ? 'text-yellow-400'
                      : 'text-green-400'
                  }
                >
                  {log.level}
                </span>{' '}
                <span className="text-text-primary">{log.message}</span>
              </div>
            ))}
          </div>
        </Card>

        <Card title="API Endpoints">
          <div className="space-y-3 max-h-80 overflow-y-auto">
            {endpoints.map((api, i) => (
              <div key={i} className="flex items-center justify-between p-3 bg-background rounded-lg">
                <div className="flex items-center gap-3">
                  <span
                    className={`px-2 py-1 text-xs font-medium rounded ${
                      api.method === 'GET'
                        ? 'bg-blue-500/10 text-blue-400'
                        : 'bg-green-500/10 text-green-400'
                    }`}
                  >
                    {api.method}
                  </span>
                  <span className="text-text-primary font-mono text-sm">{api.endpoint}</span>
                </div>
                <div className="flex items-center gap-4 text-sm">
                  <span className="text-green-400">{api.status}</span>
                  <span className="text-text-muted">{api.latency}</span>
                </div>
              </div>
            ))}
          </div>
        </Card>
      </div>

      <Card title="Environment Variables">
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="border-b border-border">
                <th className="text-left py-3 px-4 text-sm font-medium text-text-secondary">Variable</th>
                <th className="text-left py-3 px-4 text-sm font-medium text-text-secondary">Value</th>
                <th className="text-left py-3 px-4 text-sm font-medium text-text-secondary">Status</th>
              </tr>
            </thead>
            <tbody>
              {envVars.map((env, i) => (
                <tr key={i} className="border-b border-border/50">
                  <td className="py-3 px-4 text-sm text-text-primary font-mono">{env.name}</td>
                  <td className="py-3 px-4 text-sm text-text-secondary font-mono">{env.value}</td>
                  <td className="py-3 px-4">
                    <span className="px-2 py-1 text-xs font-medium bg-green-500/10 text-green-400 rounded">
                      {env.status}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  );
}
