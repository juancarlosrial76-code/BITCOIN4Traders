import { useState, useEffect } from 'react';
import { Card, Button } from '../components/ui';
import { api } from '../api/client';
import { Cpu, HardDrive, Activity, RefreshCw, Filter, AlertCircle } from 'lucide-react';

// SystemMetrics shape as returned by GET /api/system/metrics
interface SystemMetrics {
  cpu: number;              // CPU usage in percent (0–100)
  memory: number;           // Memory usage in percent (0–100)
  disk: number;             // Disk usage in percent (0–100)
  uptime: number;           // Uptime in seconds
  activeConnections: number;
  requestsPerSecond: number;
  timestamp: string;
}

// Log entry shape as returned by GET /api/system/logs
interface LogEntry {
  id: string;
  timestamp: string;        // ISO timestamp string
  level: 'DEBUG' | 'INFO' | 'WARNING' | 'ERROR';
  message: string;
  source?: string;
}

// API endpoint descriptor as returned by GET /api/system/endpoints
interface EndpointInfo {
  path: string;             // e.g. "/api/trading/status"
  method: string;           // e.g. "GET"
  description: string;
}

// Environment variable as returned by GET /api/system/env
interface EnvVariable {
  key: string;              // variable name
  value: string;            // variable value (may be masked)
}

/** Convert uptime seconds to a human-readable "Xd Yh" or "Xh Ym" string. */
function formatUptime(seconds: number): string {
  const d = Math.floor(seconds / 86400);
  const h = Math.floor((seconds % 86400) / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  return d > 0 ? `${d}d ${h}h` : `${h}h ${m}m`;
}

export function System() {
  const [metrics, setMetrics] = useState<SystemMetrics | null>(null);
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [endpoints, setEndpoints] = useState<EndpointInfo[]>([]);
  const [envVars, setEnvVars] = useState<EnvVariable[]>([]);
  const [error, setError] = useState<string | null>(null);

  const loadAll = async () => {
    setError(null);
    try {
      const [metricsData, logsData, endpointsData, envData] = await Promise.all([
        api.system.getMetrics(),
        api.system.getLogs(),
        api.system.getEndpoints(),
        api.system.getEnv(),
      ]);
      setMetrics(metricsData as unknown as SystemMetrics);
      setLogs(logsData as unknown as LogEntry[]);
      setEndpoints(endpointsData as unknown as EndpointInfo[]);
      setEnvVars(envData as unknown as EnvVariable[]);
    } catch (e) {
      const msg = (e as Error).message || 'Failed to load system data';
      console.error('[System] fetch failed:', e);
      setError(msg);
    }
  };

  useEffect(() => {
    loadAll();
    // Auto-refresh every 10 seconds
    const interval = setInterval(loadAll, 10000);
    return () => clearInterval(interval);
  }, []);

  const handleRefresh = async () => {
    setError(null);
    try {
      const [metricsData, logsData] = await Promise.all([
        api.system.getMetrics(),
        api.system.getLogs(),
      ]);
      setMetrics(metricsData as unknown as SystemMetrics);
      setLogs(logsData as unknown as LogEntry[]);
    } catch (e) {
      const msg = (e as Error).message || 'Refresh failed';
      console.error('[System] refresh failed:', e);
      setError(msg);
    }
  };

  // Map API fields to display cards — all field names match SystemMetrics interface
  const systemMetrics = [
    { label: 'CPU Usage', value: `${metrics?.cpu ?? 0}%`, icon: Cpu },
    { label: 'Memory', value: `${metrics?.memory ?? 0}%`, icon: HardDrive },
    { label: 'Req/s', value: `${metrics?.requestsPerSecond ?? 0}`, icon: Activity },
    { label: 'Uptime', value: metrics ? formatUptime(metrics.uptime) : '0h 0m', icon: Activity },
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

      {/* Error banner */}
      {error && (
        <div className="flex items-center gap-2 p-3 rounded-lg bg-red-500/10 border border-red-500/30 text-red-400 text-sm">
          <AlertCircle size={16} className="shrink-0" />
          <span>System data unavailable: {error}</span>
        </div>
      )}

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
                {/* Use log.timestamp — matches LogEntry interface */}
                <span className="text-text-muted">[{log.timestamp}]</span>{' '}
                <span
                  className={
                    log.level === 'ERROR'
                      ? 'text-red-400'
                      : log.level === 'WARNING'
                      ? 'text-yellow-400'
                      : 'text-green-400'
                  }
                >
                  {log.level}
                </span>{' '}
                <span className="text-text-primary">{log.message}</span>
              </div>
            ))}
            {logs.length === 0 && (
              <p className="text-text-muted text-center py-8">No logs available</p>
            )}
          </div>
        </Card>

        <Card title="API Endpoints">
          <div className="space-y-3 max-h-80 overflow-y-auto">
            {endpoints.map((ep, i) => (
              <div key={i} className="flex items-center justify-between p-3 bg-background rounded-lg">
                <div className="flex items-center gap-3">
                  <span
                    className={`px-2 py-1 text-xs font-medium rounded ${
                      ep.method === 'GET'
                        ? 'bg-blue-500/10 text-blue-400'
                        : 'bg-green-500/10 text-green-400'
                    }`}
                  >
                    {ep.method}
                  </span>
                  {/* Use ep.path — matches EndpointInfo interface */}
                  <span className="text-text-primary font-mono text-sm">{ep.path}</span>
                </div>
                {/* description is the only extra field in EndpointInfo */}
                <span className="text-text-muted text-xs truncate max-w-[120px]">{ep.description}</span>
              </div>
            ))}
            {endpoints.length === 0 && (
              <p className="text-text-secondary text-sm text-center py-4">No endpoints available</p>
            )}
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
                  {/* Use env.key — matches EnvVariable interface */}
                  <td className="py-3 px-4 text-sm text-text-primary font-mono">{env.key}</td>
                  <td className="py-3 px-4 text-sm text-text-secondary font-mono">{env.value}</td>
                  <td className="py-3 px-4">
                    {/* EnvVariable has no status field — show static "set" badge */}
                    <span className="px-2 py-1 text-xs font-medium bg-green-500/10 text-green-400 rounded">
                      set
                    </span>
                  </td>
                </tr>
              ))}
              {envVars.length === 0 && (
                <tr>
                  <td colSpan={3} className="py-4 text-center text-text-secondary text-sm">
                    No environment variables available
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  );
}
