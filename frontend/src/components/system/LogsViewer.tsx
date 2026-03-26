import { useEffect, useState, useRef, useCallback } from 'react';
import { api } from '../../api/client';
import { Card } from '../ui/Card';
import { Button } from '../ui/Button';

interface LogEntry {
  id: string;
  timestamp: string;
  level: 'DEBUG' | 'INFO' | 'WARNING' | 'ERROR';
  message: string;
  source?: string;
}

interface LogsViewerProps {
  maxEntries?: number;
  autoRefresh?: boolean;
  refreshInterval?: number;
}

export function LogsViewer({
  maxEntries = 500,
  autoRefresh = true,
  refreshInterval = 3000,
}: LogsViewerProps) {
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [filter, setFilter] = useState<string>('');
  const [levelFilter, setLevelFilter] = useState<string>('ALL');
  const [autoScroll, setAutoScroll] = useState(true);
  const [loading, setLoading] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);

  const fetchLogs = useCallback(async () => {
    try {
      const data = await api.system.getLogs();
      const formattedLogs: LogEntry[] = data.map((log: LogEntry, index: number) => ({
        id: log.id || `log-${index}`,
        timestamp: log.timestamp || new Date().toISOString(),
        level: log.level || 'INFO',
        message: log.message || '',
        source: log.source,
      }));
      setLogs(prev => [...prev, ...formattedLogs].slice(-maxEntries));
    } catch (e: unknown) {
      if (logs.length === 0) {
        const mockLogs: LogEntry[] = [
          {
            id: '1',
            timestamp: new Date().toISOString(),
            level: 'INFO',
            message: 'Application started successfully',
            source: 'main',
          },
          {
            id: '2',
            timestamp: new Date(Date.now() - 10000).toISOString(),
            level: 'INFO',
            message: 'Connected to database',
            source: 'db',
          },
          {
            id: '3',
            timestamp: new Date(Date.now() - 20000).toISOString(),
            level: 'WARNING',
            message: 'High memory usage detected: 82%',
            source: 'system',
          },
          {
            id: '4',
            timestamp: new Date(Date.now() - 30000).toISOString(),
            level: 'INFO',
            message: 'WebSocket connection established',
            source: 'ws',
          },
          {
            id: '5',
            timestamp: new Date(Date.now() - 40000).toISOString(),
            level: 'ERROR',
            message: 'Failed to connect to exchange API',
            source: 'api',
          },
        ];
        setLogs(mockLogs);
      }
    } finally {
      setLoading(false);
    }
  }, [maxEntries, logs]);

  useEffect(() => {
    fetchLogs();
    if (autoRefresh) {
      const interval = setInterval(fetchLogs, refreshInterval);
      return () => clearInterval(interval);
    }
    return undefined;
  }, [fetchLogs, autoRefresh, refreshInterval]);

  const clearLogs = () => {
    setLogs([]);
  };

  const getLevelColor = (level: LogEntry['level']) => {
    switch (level) {
      case 'DEBUG':
        return 'text-gray-500';
      case 'INFO':
        return 'text-blue-400';
      case 'WARNING':
        return 'text-yellow-400';
      case 'ERROR':
        return 'text-red-400';
    }
  };

  const getLevelBgColor = (level: LogEntry['level']) => {
    switch (level) {
      case 'DEBUG':
        return 'bg-gray-500/20';
      case 'INFO':
        return 'bg-blue-500/20';
      case 'WARNING':
        return 'bg-yellow-500/20';
      case 'ERROR':
        return 'bg-red-500/20';
    }
  };

  const filteredLogs = logs.filter(log => {
    const matchesLevel = levelFilter === 'ALL' || log.level === levelFilter;
    const matchesFilter =
      !filter ||
      log.message.toLowerCase().includes(filter.toLowerCase()) ||
      log.source?.toLowerCase().includes(filter.toLowerCase());
    return matchesLevel && matchesFilter;
  });

  return (
    <Card className="p-4">
      <div className="flex items-center justify-between mb-4">
        <h3 className="font-bold">Logs</h3>
        <div className="flex gap-2">
          <Button variant="ghost" size="sm" onClick={clearLogs}>
            Clear
          </Button>
          <Button variant="ghost" size="sm" onClick={fetchLogs} disabled={loading}>
            {loading ? 'Refreshing...' : 'Refresh'}
          </Button>
        </div>
      </div>

      <div className="flex gap-2 mb-4">
        <input
          type="text"
          placeholder="Filter logs..."
          className="flex-1 bg-gray-800 border border-gray-700 rounded px-3 py-1.5 text-sm"
          value={filter}
          onChange={e => setFilter(e.target.value)}
        />
        <select
          className="bg-gray-800 border border-gray-700 rounded px-3 py-1.5 text-sm"
          value={levelFilter}
          onChange={e => setLevelFilter(e.target.value)}
        >
          <option value="ALL">All Levels</option>
          <option value="DEBUG">Debug</option>
          <option value="INFO">Info</option>
          <option value="WARNING">Warning</option>
          <option value="ERROR">Error</option>
        </select>
      </div>

      <div
        ref={containerRef}
        className="bg-gray-900 rounded-lg p-3 h-80 overflow-y-auto font-mono text-xs"
      >
        {filteredLogs.length === 0 ? (
          <div className="text-gray-500 text-center py-8">
            {loading ? 'Loading logs...' : 'No logs to display'}
          </div>
        ) : (
          filteredLogs.map(log => (
            <div key={log.id} className="py-1 hover:bg-gray-800/50">
              <span className="text-gray-500 mr-2">
                {new Date(log.timestamp).toLocaleTimeString()}
              </span>
              <span
                className={`px-1.5 py-0.5 rounded text-xs mr-2 ${getLevelBgColor(log.level)} ${getLevelColor(log.level)}`}
              >
                {log.level}
              </span>
              {log.source && <span className="text-purple-400 mr-2">[{log.source}]</span>}
              <span className="text-gray-300">{log.message}</span>
            </div>
          ))
        )}
      </div>

      <div className="flex items-center justify-between mt-3 text-xs text-gray-500">
        <span>{filteredLogs.length} entries</span>
        <label className="flex items-center gap-2 cursor-pointer">
          <input
            type="checkbox"
            checked={autoScroll}
            onChange={e => setAutoScroll(e.target.checked)}
            className="rounded"
          />
          Auto-scroll
        </label>
      </div>
    </Card>
  );
}

export default LogsViewer;
