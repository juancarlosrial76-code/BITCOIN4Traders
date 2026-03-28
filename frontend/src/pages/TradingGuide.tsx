import { Link } from 'react-router-dom';
import { useTranslation } from 'react-i18next';
import { Card } from '../components/ui';
import { ArrowLeft } from 'lucide-react';

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <Card title={title}>
      <div className="space-y-3 text-sm text-text-secondary">{children}</div>
    </Card>
  );
}

export function TradingGuide() {
  const { t } = useTranslation();

  return (
    <div className="space-y-6 max-w-3xl">
      <div className="flex items-center gap-3">
        <Link to="/docs" className="text-text-muted hover:text-bitcoin-orange transition-colors">
          <ArrowLeft size={20} />
        </Link>
        <div>
          <h1 className="text-3xl font-bold text-text-primary">
            {t('docs.tradingGuide.title', 'Trading Guide')}
          </h1>
          <p className="text-text-secondary">
            {t('docs.tradingGuide.subtitle', 'Paper trading, live mode, risk profiles and Kelly sizing.')}
          </p>
        </div>
      </div>

      <Section title={t('docs.tradingGuide.modes', 'Trading Modes')}>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="p-4 rounded-lg bg-background border border-border">
            <h4 className="font-semibold text-text-primary mb-2">Paper Trading</h4>
            <p>Simulated trading with a virtual balance. No real money at risk. Ideal for testing strategies and getting familiar with the bot.</p>
            <p className="mt-2 text-green-400 text-xs font-medium">Recommended for beginners</p>
          </div>
          <div className="p-4 rounded-lg bg-background border border-border">
            <h4 className="font-semibold text-text-primary mb-2">Live Trading</h4>
            <p>Real orders on Binance using your API keys. Requires API keys with trading permissions. Profits and losses are real.</p>
            <p className="mt-2 text-yellow-400 text-xs font-medium">Only after testing in paper mode</p>
          </div>
        </div>
      </Section>

      <Section title={t('docs.tradingGuide.riskProfiles', 'Risk Profiles')}>
        <p>
          The bot offers four risk profiles, set during registration via the risk questionnaire. Each profile controls position sizing, drawdown limits and Kelly fraction.
        </p>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mt-3">
          {[
            { name: 'Conservative', color: 'text-blue-400', bg: 'bg-blue-400/10', kelly: '10%', dd: '10%' },
            { name: 'Balanced', color: 'text-green-400', bg: 'bg-green-400/10', kelly: '25%', dd: '20%' },
            { name: 'Growth', color: 'text-bitcoin-orange', bg: 'bg-bitcoin-orange/10', kelly: '40%', dd: '30%' },
            { name: 'Speculative', color: 'text-red-400', bg: 'bg-red-400/10', kelly: '60%', dd: '50%' },
          ].map(p => (
            <div key={p.name} className={`p-3 rounded-lg ${p.bg} border border-border`}>
              <p className={`font-semibold text-sm ${p.color}`}>{p.name}</p>
              <p className="text-xs text-text-muted mt-1">Kelly: {p.kelly}</p>
              <p className="text-xs text-text-muted">Max DD: {p.dd}</p>
            </div>
          ))}
        </div>
      </Section>

      <Section title={t('docs.tradingGuide.kelly', 'Kelly Criterion')}>
        <p>
          The bot uses the Kelly Criterion to determine the optimal fraction of capital to risk per trade. The formula:
        </p>
        <pre className="bg-black rounded-lg p-4 font-mono text-sm text-green-400 overflow-x-auto">
          <code>{`kelly_fraction = (win_rate - (1 - win_rate) / profit_ratio) * kelly_cap`}</code>
        </pre>
        <p>
          <strong className="text-text-primary">Phase 1 (training):</strong> Kelly is bypassed — the agent explores freely without position-sizing constraints.
        </p>
        <p>
          <strong className="text-text-primary">Phase 2 (production):</strong> Adaptive Kelly re-enabled once win rate exceeds 18%. Position size is scaled by your risk profile's Kelly cap.
        </p>
      </Section>

      <Section title={t('docs.tradingGuide.circuitBreakers', 'Circuit Breakers')}>
        <p>Automatic trading halts to protect capital:</p>
        <ul className="list-disc list-inside space-y-1 mt-2">
          <li><strong className="text-text-primary">Max drawdown per session</strong> — stops trading when cumulative loss exceeds the profile limit</li>
          <li><strong className="text-text-primary">Consecutive losses</strong> — halts after N consecutive losing trades</li>
          <li><strong className="text-text-primary">Volatility spike</strong> — reduces position size when market volatility exceeds threshold</li>
          <li><strong className="text-text-primary">Daily loss limit</strong> — hard stop for the day when daily P&L crosses -2%</li>
        </ul>
      </Section>

      <Section title={t('docs.tradingGuide.signals', 'Trading Signals')}>
        <p>
          The RL agent produces one of three signals on every candle close:
        </p>
        <div className="grid grid-cols-3 gap-3 mt-3">
          {[
            { signal: 'BUY', color: 'text-green-400', bg: 'bg-green-400/10', desc: 'Open or increase long position' },
            { signal: 'SELL', color: 'text-red-400', bg: 'bg-red-400/10', desc: 'Close long / reduce position' },
            { signal: 'HOLD', color: 'text-text-secondary', bg: 'bg-background', desc: 'No action — stay in current state' },
          ].map(s => (
            <div key={s.signal} className={`p-3 rounded-lg ${s.bg} border border-border text-center`}>
              <p className={`font-bold ${s.color}`}>{s.signal}</p>
              <p className="text-xs text-text-muted mt-1">{s.desc}</p>
            </div>
          ))}
        </div>
      </Section>

      <div className="flex gap-4">
        <Link to="/docs/quickstart" className="text-sm text-text-muted hover:text-bitcoin-orange">
          ← Quickstart
        </Link>
        <Link to="/docs/api" className="text-sm text-bitcoin-orange hover:underline">
          Next: API Reference →
        </Link>
      </div>
    </div>
  );
}
