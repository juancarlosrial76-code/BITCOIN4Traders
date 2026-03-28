import { Link } from 'react-router-dom';
import { useTranslation } from 'react-i18next';
import { Card } from '../components/ui';
import { ArrowLeft } from 'lucide-react';
import { useState } from 'react';
import { Search } from 'lucide-react';

interface GlossaryEntry {
  term: string;
  category: 'rl' | 'trading' | 'risk' | 'technical';
  definition: string;
}

const entries: GlossaryEntry[] = [
  // Reinforcement Learning
  { term: 'Agent', category: 'rl', definition: 'The RL model that observes market state and decides which action to take (BUY / SELL / HOLD).' },
  { term: 'Environment', category: 'rl', definition: 'The simulated trading market the agent interacts with. Provides observations and rewards.' },
  { term: 'Observation', category: 'rl', definition: 'The set of features the agent sees at each step: OHLCV data, indicators, portfolio state.' },
  { term: 'Reward', category: 'rl', definition: 'Signal that guides learning. Positive for profitable trades, negative for losses, shaped by the WinRateAwareReward function.' },
  { term: 'PPO', category: 'rl', definition: 'Proximal Policy Optimization — the RL algorithm used. Balances exploration and exploitation with a clipped surrogate objective.' },
  { term: 'Episode', category: 'rl', definition: 'One complete run through the training data from start to end. Each episode trains the agent on historical price data.' },
  { term: 'Policy', category: 'rl', definition: 'The agent\'s decision function — maps observations to action probabilities.' },
  { term: 'Curriculum Training', category: 'rl', definition: 'Two-phase training: Phase 1 bypasses Kelly for free exploration; Phase 2 re-enables risk management once win rate > 18%.' },

  // Trading
  { term: 'BTC/USDT', category: 'trading', definition: 'The primary trading pair. Bitcoin priced in USDT (Tether, a USD-pegged stablecoin).' },
  { term: 'Long', category: 'trading', definition: 'Buying BTC with the expectation the price will rise.' },
  { term: 'Short', category: 'trading', definition: 'Selling BTC (or borrowing to sell) with the expectation the price will fall.' },
  { term: 'Paper Trading', category: 'trading', definition: 'Simulated trading with virtual money. No real funds at risk. Used for testing strategies.' },
  { term: 'PnL', category: 'trading', definition: 'Profit and Loss. Realized PnL from closed positions; unrealized PnL from open positions.' },
  { term: 'Spread', category: 'trading', definition: 'Difference between the best ask (lowest sell) and best bid (highest buy) price in the order book.' },
  { term: 'Order Book', category: 'trading', definition: 'List of all open buy (bids) and sell (asks) orders at various price levels.' },
  { term: 'Candle / OHLCV', category: 'trading', definition: 'Open, High, Low, Close, Volume — price summary for a time period (e.g. 1h candle).' },

  // Risk
  { term: 'Kelly Criterion', category: 'risk', definition: 'Mathematical formula to size positions optimally based on win rate and profit ratio. Prevents overbetting.' },
  { term: 'Max Drawdown', category: 'risk', definition: 'The largest peak-to-trough decline in portfolio value. Key risk metric — lower is safer.' },
  { term: 'Sharpe Ratio', category: 'risk', definition: 'Return per unit of risk (standard deviation). Higher is better. >1 is good, >2 is excellent.' },
  { term: 'Win Rate', category: 'risk', definition: 'Percentage of trades that are profitable. The bot targets 65–76% through the WinRateAwareReward function.' },
  { term: 'Circuit Breaker', category: 'risk', definition: 'Automatic trading halt triggered when drawdown, consecutive losses, or volatility exceed safe thresholds.' },
  { term: 'Profit Factor', category: 'risk', definition: 'Gross profit divided by gross loss. >1.5 is considered good; <1.0 means the strategy is losing money overall.' },

  // Technical
  { term: 'FastAPI', category: 'technical', definition: 'Python web framework used for the backend REST API. Fast, typed, auto-generates OpenAPI docs.' },
  { term: 'Vite', category: 'technical', definition: 'Frontend build tool. Powers the React development server and production build.' },
  { term: 'Zustand', category: 'technical', definition: 'Lightweight React state management library used for trading, config, and analytics stores.' },
  { term: 'WebSocket', category: 'technical', definition: 'Real-time bidirectional connection. Used to stream live price updates to the dashboard.' },
  { term: 'PyTorch', category: 'technical', definition: 'Deep learning framework used to train and run the RL agent neural network.' },
  { term: 'Champion Model', category: 'technical', definition: 'The best-performing model as determined by mean_return. Saved to best_model_trader.pth and deployed automatically.' },
];

const categoryLabel: Record<string, string> = {
  all: 'All',
  rl: 'Reinforcement Learning',
  trading: 'Trading',
  risk: 'Risk Management',
  technical: 'Technical',
};

const categoryColor: Record<string, string> = {
  rl: 'bg-purple-500/10 text-purple-400',
  trading: 'bg-bitcoin-orange/10 text-bitcoin-orange',
  risk: 'bg-red-500/10 text-red-400',
  technical: 'bg-blue-500/10 text-blue-400',
};

export function Glossary() {
  const { t } = useTranslation();
  const [search, setSearch] = useState('');
  const [category, setCategory] = useState('all');

  const filtered = entries.filter(e => {
    const matchCat = category === 'all' || e.category === category;
    const q = search.toLowerCase();
    const matchSearch = e.term.toLowerCase().includes(q) || e.definition.toLowerCase().includes(q);
    return matchCat && matchSearch;
  });

  return (
    <div className="space-y-6 max-w-3xl">
      <div className="flex items-center gap-3">
        <Link to="/docs" className="text-text-muted hover:text-bitcoin-orange transition-colors">
          <ArrowLeft size={20} />
        </Link>
        <div>
          <h1 className="text-3xl font-bold text-text-primary">
            {t('docs.glossary.title', 'Glossary')}
          </h1>
          <p className="text-text-secondary">
            {t('docs.glossary.subtitle', 'Key trading and RL terms explained simply.')}
          </p>
        </div>
      </div>

      {/* Search */}
      <div className="relative">
        <Search className="absolute left-3 top-1/2 -translate-y-1/2 text-text-muted" size={18} />
        <input
          type="text"
          placeholder={t('common.search', 'Search...')}
          value={search}
          onChange={e => setSearch(e.target.value)}
          className="w-full bg-background border border-border rounded-lg pl-10 pr-4 py-2.5 text-text-primary placeholder-text-muted focus:outline-none focus:ring-2 focus:ring-bitcoin-orange/50 text-sm"
        />
      </div>

      {/* Category filter */}
      <div className="flex flex-wrap gap-2">
        {Object.entries(categoryLabel).map(([key, label]) => (
          <button
            key={key}
            onClick={() => setCategory(key)}
            className={`px-3 py-1.5 rounded-lg text-sm transition-colors ${
              category === key
                ? 'bg-bitcoin-orange text-white'
                : 'bg-card border border-border text-text-secondary hover:bg-bitcoin-orange/10'
            }`}
          >
            {label}
          </button>
        ))}
      </div>

      {/* Entries */}
      <div className="space-y-3">
        {filtered.map(entry => (
          <Card key={entry.term} className="flex items-start gap-4">
            <div className="flex-1">
              <div className="flex items-center gap-2 mb-1">
                <h3 className="font-semibold text-text-primary">{entry.term}</h3>
                <span className={`px-2 py-0.5 text-xs rounded font-medium ${categoryColor[entry.category]}`}>
                  {categoryLabel[entry.category]}
                </span>
              </div>
              <p className="text-sm text-text-secondary">{entry.definition}</p>
            </div>
          </Card>
        ))}
        {filtered.length === 0 && (
          <p className="text-center text-text-muted py-8">No terms found.</p>
        )}
      </div>

      <div className="flex gap-4">
        <Link to="/docs/api" className="text-sm text-text-muted hover:text-bitcoin-orange">
          ← API Reference
        </Link>
        <Link to="/docs/troubleshooting" className="text-sm text-bitcoin-orange hover:underline">
          Next: Troubleshooting →
        </Link>
      </div>
    </div>
  );
}
