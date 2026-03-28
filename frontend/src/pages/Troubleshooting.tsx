import { Link } from 'react-router-dom';
import { useTranslation } from 'react-i18next';
import { Card } from '../components/ui';
import { ArrowLeft, AlertTriangle, CheckCircle, Terminal } from 'lucide-react';

interface Issue {
  symptom: string;
  cause: string;
  fix: string;
  code?: string;
}

const issues: Issue[] = [
  {
    symptom: 'Frontend shows "VITE_API_URL not set" warning',
    cause: 'The .env file is missing or VITE_API_URL is not defined.',
    fix: 'Create frontend/.env from the template:',
    code: 'cp frontend/.env.example frontend/.env\n# Then edit and set VITE_API_URL=http://localhost:8000',
  },
  {
    symptom: 'Backend returns 401 Unauthorized',
    cause: 'JWT token is missing, expired, or the Authorization header is malformed.',
    fix: 'Log out and log in again. The frontend automatically redirects to /login on 401.',
  },
  {
    symptom: 'Training times out or hangs',
    cause: 'Too many iterations per run, slow CPU, or a deadlock in the environment.',
    fix: 'Reduce --iterations (currently 10). Check logs/training/ for detailed errors.',
    code: 'python train.py --device cpu --iterations 5',
  },
  {
    symptom: 'Win Rate stuck at 0% after many rounds',
    cause: 'The win-rate calculation is broken, or the environment never executes trades.',
    fix: 'auto_12h_train.py detects this after 10 rounds and logs "NOTSTOP". Check logs/training/12h_auto.log.',
  },
  {
    symptom: 'Best model not deployed after training',
    cause: 'The adversarial_trainer.py guard only saves best_model_trader.pth if mean_return > _best_return.',
    fix: 'Check data/models/adversarial/best_model.pth modification time. If the new run had a lower return than the previous best, no deployment is expected — this is correct behavior.',
    code: 'ls -lh data/models/adversarial/best_model.pth\ncat data/models/adversarial/champion.json',
  },
  {
    symptom: 'WebSocket not connecting',
    cause: 'Backend not running, wrong WS_URL, or firewall blocking port 8000.',
    fix: 'Verify backend is running on port 8000. Check VITE_WS_URL in .env (defaults to ws://localhost:8000/ws).',
    code: 'curl http://localhost:8000/api/status',
  },
  {
    symptom: 'Order Book shows "unavailable"',
    cause: 'The backend does not have /api/orderbook/{symbol} implemented yet, returning 404.',
    fix: 'The /api/orderbook/ endpoint needs to be implemented in the backend. Currently returns 404 — the frontend shows an error message instead of fake data.',
  },
  {
    symptom: 'npm run build fails with TypeScript errors',
    cause: 'Type mismatches between API response shapes (snake_case) and frontend interfaces.',
    fix: 'Run npm run build to see the error list. All known type errors were resolved in TODOS/20260328_005_FRONTEND_AUDIT.md.',
    code: 'cd frontend && npm run build 2>&1 | grep error',
  },
  {
    symptom: 'Page shows no data / API fetch failed banner',
    cause: 'Backend is not running, or authentication token expired.',
    fix: 'Check the browser console for the specific error. Restart the backend if needed.',
    code: 'python run.py\n# or for the backend only:\ncd backend && uvicorn main:app --reload',
  },
  {
    symptom: 'auto_12h_train.py exits before 24h',
    cause: 'A Python exception in run_training(), or an OS-level kill signal.',
    fix: 'Check logs/training/12h_errors.log for the exception. Common cause: conda environment not activated, PYTHONPATH not set.',
    code: 'tail -50 logs/training/12h_errors.log',
  },
];

export function Troubleshooting() {
  const { t } = useTranslation();

  return (
    <div className="space-y-6 max-w-3xl">
      <div className="flex items-center gap-3">
        <Link to="/docs" className="text-text-muted hover:text-bitcoin-orange transition-colors">
          <ArrowLeft size={20} />
        </Link>
        <div>
          <h1 className="text-3xl font-bold text-text-primary">
            {t('docs.troubleshooting.title', 'Troubleshooting')}
          </h1>
          <p className="text-text-secondary">
            {t('docs.troubleshooting.subtitle', 'Common errors, fixes and diagnostic tips.')}
          </p>
        </div>
      </div>

      {/* Quick diagnostic */}
      <Card title="Quick Diagnostics">
        <p className="text-sm text-text-secondary mb-3">Run these checks first:</p>
        <div className="space-y-2">
          {[
            'Backend running?  →  curl http://localhost:8000/api/status',
            'Frontend env set? →  cat frontend/.env',
            'Training logs?    →  tail -20 logs/training/12h_auto.log',
            'Model exists?     →  ls -lh data/models/adversarial/best_model.pth',
          ].map(cmd => (
            <div key={cmd} className="flex items-start gap-2">
              <Terminal size={14} className="text-text-muted mt-0.5 flex-shrink-0" />
              <code className="text-xs font-mono text-green-400">{cmd}</code>
            </div>
          ))}
        </div>
      </Card>

      {/* Issues */}
      <div className="space-y-4">
        {issues.map((issue, i) => (
          <Card key={i}>
            <div className="space-y-2">
              <div className="flex items-start gap-2">
                <AlertTriangle size={16} className="text-yellow-400 flex-shrink-0 mt-0.5" />
                <h3 className="font-semibold text-text-primary text-sm">{issue.symptom}</h3>
              </div>
              <div className="ml-6 space-y-1">
                <p className="text-xs text-text-muted">
                  <strong className="text-text-secondary">Cause:</strong> {issue.cause}
                </p>
                <div className="flex items-start gap-2">
                  <CheckCircle size={12} className="text-green-400 flex-shrink-0 mt-0.5" />
                  <p className="text-xs text-text-secondary">{issue.fix}</p>
                </div>
                {issue.code && (
                  <pre className="bg-black rounded p-3 font-mono text-xs text-green-400 overflow-x-auto mt-2">
                    <code>{issue.code}</code>
                  </pre>
                )}
              </div>
            </div>
          </Card>
        ))}
      </div>

      <Card className="border-blue-500/30 bg-blue-500/5">
        <p className="text-sm text-blue-400">
          Still stuck? Check the{' '}
          <Link to="/faq" className="underline hover:text-blue-300">FAQ</Link>
          {' '}or open an issue on GitHub.
        </p>
      </Card>

      <div className="flex gap-4">
        <Link to="/docs/glossary" className="text-sm text-text-muted hover:text-bitcoin-orange">
          ← Glossary
        </Link>
        <Link to="/faq" className="text-sm text-bitcoin-orange hover:underline">
          Next: FAQ →
        </Link>
      </div>
    </div>
  );
}
