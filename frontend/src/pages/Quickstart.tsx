import { Link } from 'react-router-dom';
import { useTranslation } from 'react-i18next';
import { Card } from '../components/ui';
import { ArrowLeft, CheckCircle } from 'lucide-react';

function Step({ n, title, children }: { n: number; title: string; children: React.ReactNode }) {
  return (
    <div className="flex gap-4">
      <div className="flex-shrink-0 w-8 h-8 rounded-full bg-bitcoin-orange/20 text-bitcoin-orange flex items-center justify-center font-bold text-sm">
        {n}
      </div>
      <div className="flex-1 pb-6 border-b border-border/50 last:border-0">
        <h3 className="font-semibold text-text-primary mb-2">{title}</h3>
        {children}
      </div>
    </div>
  );
}

function Code({ children }: { children: string }) {
  return (
    <pre className="bg-black rounded-lg p-4 font-mono text-sm text-green-400 overflow-x-auto mt-2 mb-2">
      <code>{children}</code>
    </pre>
  );
}

export function Quickstart() {
  const { t } = useTranslation();

  return (
    <div className="space-y-6 max-w-3xl">
      <div className="flex items-center gap-3">
        <Link to="/docs" className="text-text-muted hover:text-bitcoin-orange transition-colors">
          <ArrowLeft size={20} />
        </Link>
        <div>
          <h1 className="text-3xl font-bold text-text-primary">
            {t('docs.quickstart.title', 'Quickstart')}
          </h1>
          <p className="text-text-secondary">
            {t('docs.quickstart.subtitle', 'Set up and run BITCOIN4Traders in 5 minutes.')}
          </p>
        </div>
      </div>

      <Card title={t('docs.quickstart.requirements', 'Requirements')}>
        <ul className="space-y-2">
          {[
            'Python 3.11 or higher',
            'Node.js 18 or higher',
            'Git',
            'A Binance account (for live trading — not required for paper trading)',
          ].map(item => (
            <li key={item} className="flex items-center gap-2 text-sm text-text-secondary">
              <CheckCircle size={14} className="text-green-400 flex-shrink-0" />
              {item}
            </li>
          ))}
        </ul>
      </Card>

      <Card title={t('docs.quickstart.installation', 'Installation')}>
        <div className="space-y-1">
          <Step n={1} title="Clone the repository">
            <Code>{`git clone https://github.com/juancarlosrial76-code/BITCOIN4Traders.git
cd BITCOIN4Traders`}</Code>
          </Step>

          <Step n={2} title="Create a virtual environment">
            <Code>{`python -m venv venv
source venv/bin/activate   # Linux / Mac
# or
venv\\Scripts\\activate       # Windows`}</Code>
          </Step>

          <Step n={3} title="Install Python dependencies">
            <Code>{`pip install -r requirements.txt`}</Code>
          </Step>

          <Step n={4} title="Start the backend (FastAPI)">
            <Code>{`cd backend
uvicorn main:app --reload`}</Code>
            <p className="text-sm text-text-secondary">
              Backend available at <code className="text-bitcoin-orange">http://localhost:8000</code>
            </p>
          </Step>

          <Step n={5} title="Start the frontend (Vite)">
            <Code>{`cd frontend
npm install
npm run dev`}</Code>
            <p className="text-sm text-text-secondary">
              Dashboard available at <code className="text-bitcoin-orange">http://localhost:5173</code>
            </p>
          </Step>
        </div>
      </Card>

      <Card title={t('docs.quickstart.firstLogin', 'First Login')}>
        <p className="text-sm text-text-secondary mb-3">
          Open <code className="text-bitcoin-orange">http://localhost:5173</code> and sign in with the default credentials:
        </p>
        <div className="bg-black rounded-lg p-4 font-mono text-sm space-y-1">
          <p><span className="text-text-muted">Username: </span><span className="text-green-400">admin</span></p>
          <p><span className="text-text-muted">Password: </span><span className="text-green-400">admin123</span></p>
        </div>
        <p className="text-xs text-yellow-400 mt-3">
          Change the default password immediately in a production environment.
        </p>
      </Card>

      <Card title={t('docs.quickstart.paperTrading', 'Start Paper Trading')}>
        <ol className="space-y-2 text-sm text-text-secondary list-decimal list-inside">
          <li>Go to the <Link to="/trading" className="text-bitcoin-orange hover:underline">Trading</Link> page</li>
          <li>Make sure mode is set to <strong className="text-text-primary">Paper Trading</strong></li>
          <li>Click <strong className="text-text-primary">Start Bot</strong></li>
          <li>Watch the <Link to="/dashboard" className="text-bitcoin-orange hover:underline">Dashboard</Link> for live signals and P&L</li>
        </ol>
      </Card>

      <div className="flex gap-3">
        <Link
          to="/docs/trading-guide"
          className="text-sm text-bitcoin-orange hover:underline"
        >
          Next: Trading Guide →
        </Link>
      </div>
    </div>
  );
}
