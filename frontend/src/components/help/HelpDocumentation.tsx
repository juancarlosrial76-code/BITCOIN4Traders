import { useState, memo } from 'react';
import { useTranslation } from 'react-i18next';
import { Card } from '../ui/Card';
import {
  BookOpen,
  Calculator,
  TrendingUp,
  AlertTriangle,
  Info,
  ChevronRight,
  ChevronDown,
} from 'lucide-react';

interface GlossaryItemProps {
  term: string;
  definition: string;
  formula?: string;
  interpretation: string;
}

const GlossaryItem = memo(function GlossaryItem({
  term,
  definition,
  formula,
  interpretation,
}: GlossaryItemProps) {
  const [isExpanded, setIsExpanded] = useState(false);

  return (
    <div className="border border-gray-700 rounded-lg overflow-hidden">
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className="w-full flex items-center justify-between p-4 bg-gray-800/50 hover:bg-gray-800 transition-colors text-left"
        aria-expanded={isExpanded}
      >
        <span className="font-medium text-bitcoin-orange">{term}</span>
        {isExpanded ? <ChevronDown size={18} /> : <ChevronRight size={18} />}
      </button>

      {isExpanded && (
        <div className="p-4 space-y-3 bg-gray-900/50">
          <p className="text-gray-300">{definition}</p>

          {formula && (
            <div className="bg-gray-800 p-3 rounded-lg">
              <div className="text-xs text-gray-500 mb-1">Formula</div>
              <code className="text-sm text-green-400 font-mono">{formula}</code>
            </div>
          )}

          <div className="bg-blue-500/10 border border-blue-500/30 p-3 rounded-lg">
            <div className="text-xs text-blue-400 mb-1">Interpretation</div>
            <p className="text-sm text-gray-300">{interpretation}</p>
          </div>
        </div>
      )}
    </div>
  );
});

interface MetricCardProps {
  title: string;
  value: string | number;
  description: string;
  tooltip?: string;
  icon?: React.ReactNode;
}

const MetricCard = memo(function MetricCard({ title, value, description, icon }: MetricCardProps) {
  return (
    <div className="bg-gray-800/50 border border-gray-700 rounded-lg p-4 hover:border-gray-600 transition-colors">
      <div className="flex items-start justify-between mb-2">
        <h4 className="font-medium text-gray-200">{title}</h4>
        {icon && <span className="text-bitcoin-orange">{icon}</span>}
      </div>
      <div className="text-2xl font-bold text-bitcoin-orange mb-1">{value}</div>
      <p className="text-xs text-gray-500">{description}</p>
    </div>
  );
});

export const HelpDocumentation = memo(function HelpDocumentation() {
  const { t } = useTranslation();
  const [activeTab, setActiveTab] = useState<'glossary' | 'metrics' | 'guide'>('glossary');
  const [searchTerm, setSearchTerm] = useState('');

  const glossaryItems = [
    {
      term: t('glossary.sharpeRatio.term'),
      definition: t('glossary.sharpeRatio.definition'),
      formula: t('glossary.sharpeRatio.formula'),
      interpretation: t('glossary.sharpeRatio.interpretation'),
    },
    {
      term: t('glossary.profitFactor.term'),
      definition: t('glossary.profitFactor.definition'),
      formula: t('glossary.profitFactor.formula'),
      interpretation: t('glossary.profitFactor.interpretation'),
    },
    {
      term: t('glossary.maxDrawdown.term'),
      definition: t('glossary.maxDrawdown.definition'),
      formula: t('glossary.maxDrawdown.formula'),
      interpretation: t('glossary.maxDrawdown.interpretation'),
    },
    {
      term: t('glossary.winRate.term'),
      definition: t('glossary.winRate.definition'),
      formula: t('glossary.winRate.formula'),
      interpretation: t('glossary.winRate.interpretation'),
    },
    {
      term: t('glossary.calmarRatio.term'),
      definition: t('glossary.calmarRatio.definition'),
      formula: t('glossary.calmarRatio.formula'),
      interpretation: t('glossary.calmarRatio.interpretation'),
    },
    {
      term: t('glossary.sortinoRatio.term'),
      definition: t('glossary.sortinoRatio.definition'),
      formula: t('glossary.sortinoRatio.formula'),
      interpretation: t('glossary.sortinoRatio.interpretation'),
    },
    {
      term: t('glossary.alpha.term'),
      definition: t('glossary.alpha.definition'),
      formula: t('glossary.alpha.formula'),
      interpretation: t('glossary.alpha.interpretation'),
    },
    {
      term: t('glossary.beta.term'),
      definition: t('glossary.beta.definition'),
      formula: t('glossary.beta.formula'),
      interpretation: t('glossary.beta.interpretation'),
    },
    {
      term: t('glossary.volatility.term'),
      definition: t('glossary.volatility.definition'),
      formula: t('glossary.volatility.formula'),
      interpretation: t('glossary.volatility.interpretation'),
    },
    {
      term: t('glossary.informationRatio.term'),
      definition: t('glossary.informationRatio.definition'),
      formula: t('glossary.informationRatio.formula'),
      interpretation: t('glossary.informationRatio.interpretation'),
    },
  ];

  const filteredGlossary = glossaryItems.filter(
    item =>
      item.term.toLowerCase().includes(searchTerm.toLowerCase()) ||
      item.definition.toLowerCase().includes(searchTerm.toLowerCase())
  );

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold flex items-center gap-2">
            <BookOpen className="text-bitcoin-orange" />
            {t('help.title')}
          </h2>
          <p className="text-gray-400 mt-1">
            Learn about trading metrics, formulas, and strategies
          </p>
        </div>
      </div>

      <div className="flex gap-2 border-b border-gray-700 pb-2">
        <button
          onClick={() => setActiveTab('glossary')}
          className={`px-4 py-2 rounded-t-lg transition-colors ${
            activeTab === 'glossary'
              ? 'bg-bitcoin-orange text-white'
              : 'text-gray-400 hover:text-white'
          }`}
        >
          <span className="flex items-center gap-2">
            <BookOpen size={16} />
            {t('help.glossary')}
          </span>
        </button>
        <button
          onClick={() => setActiveTab('metrics')}
          className={`px-4 py-2 rounded-t-lg transition-colors ${
            activeTab === 'metrics'
              ? 'bg-bitcoin-orange text-white'
              : 'text-gray-400 hover:text-white'
          }`}
        >
          <span className="flex items-center gap-2">
            <Calculator size={16} />
            {t('analytics.metrics')}
          </span>
        </button>
        <button
          onClick={() => setActiveTab('guide')}
          className={`px-4 py-2 rounded-t-lg transition-colors ${
            activeTab === 'guide'
              ? 'bg-bitcoin-orange text-white'
              : 'text-gray-400 hover:text-white'
          }`}
        >
          <span className="flex items-center gap-2">
            <TrendingUp size={16} />
            {t('help.gettingStarted')}
          </span>
        </button>
      </div>

      {activeTab === 'glossary' && (
        <div className="space-y-4">
          <input
            type="text"
            placeholder="Search terms..."
            value={searchTerm}
            onChange={e => setSearchTerm(e.target.value)}
            className="w-full bg-gray-800 border border-gray-700 rounded-lg px-4 py-2 text-white placeholder-gray-500 focus:outline-none focus:border-bitcoin-orange"
          />

          <div className="grid gap-3">
            {filteredGlossary.map((item, index) => (
              <GlossaryItem
                key={index}
                term={item.term}
                definition={item.definition}
                formula={item.formula}
                interpretation={item.interpretation}
              />
            ))}
          </div>
        </div>
      )}

      {activeTab === 'metrics' && (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          <MetricCard
            title={t('portfolio.sharpeRatio')}
            value="&gt; 1.0"
            description={t('tooltips.sharpeRatio')}
            icon={<TrendingUp size={20} />}
          />
          <MetricCard
            title={t('portfolio.profitFactor')}
            value="&gt; 1.5"
            description={t('tooltips.profitFactor')}
            icon={<Calculator size={20} />}
          />
          <MetricCard
            title={t('portfolio.winRate')}
            value="&gt; 50%"
            description={t('tooltips.winRate')}
            icon={<TrendingUp size={20} />}
          />
          <MetricCard
            title={t('portfolio.maxDrawdown')}
            value="&lt; 20%"
            description={t('tooltips.maxDrawdown')}
            icon={<AlertTriangle size={20} />}
          />
          <MetricCard
            title="Calmar Ratio"
            value="&gt; 1.0"
            description="Return / Max Drawdown - Risk-adjusted return"
            icon={<TrendingUp size={20} />}
          />
          <MetricCard
            title="Sortino Ratio"
            value="> 1.5"
            description="Like Sharpe but only downside risk"
            icon={<Calculator size={20} />}
          />
        </div>
      )}

      {activeTab === 'guide' && (
        <div className="space-y-6">
          <Card className="p-6">
            <h3 className="text-xl font-bold mb-4 flex items-center gap-2">
              <TrendingUp className="text-bitcoin-orange" />
              Getting Started with BITCOIN4Traders
            </h3>

            <div className="space-y-6">
              <section>
                <h4 className="font-semibold text-lg mb-2 text-bitcoin-orange">
                  1. Paper Trading First
                </h4>
                <p className="text-gray-300">
                  Always start with paper trading to test your strategies without risking real
                  money.
                  {t('tooltips.paperTrading')}
                </p>
              </section>

              <section>
                <h4 className="font-semibold text-lg mb-2 text-bitcoin-orange">
                  2. Understand Risk Management
                </h4>
                <ul className="list-disc list-inside text-gray-300 space-y-2">
                  <li>{t('tooltips.stopLoss')}</li>
                  <li>{t('tooltips.takeProfit')}</li>
                  <li>Never risk more than 2% per trade</li>
                  <li>Set maximum drawdown limits (recommended: 20%)</li>
                </ul>
              </section>

              <section>
                <h4 className="font-semibold text-lg mb-2 text-bitcoin-orange">
                  3. Key Metrics to Watch
                </h4>
                <div className="bg-gray-800 p-4 rounded-lg">
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="text-left text-gray-500 border-b border-gray-700">
                        <th className="pb-2">Metric</th>
                        <th className="pb-2">Good</th>
                        <th className="pb-2">Excellent</th>
                      </tr>
                    </thead>
                    <tbody className="text-gray-300">
                      <tr className="border-b border-gray-700">
                        <td className="py-2">Sharpe Ratio</td>
                        <td className="py-2 text-green-400">&gt; 1.0</td>
                        <td className="py-2 text-green-400">&gt; 2.0</td>
                      </tr>
                      <tr className="border-b border-gray-700">
                        <td className="py-2">Win Rate</td>
                        <td className="py-2 text-green-400">&gt; 50%</td>
                        <td className="py-2 text-green-400">&gt; 60%</td>
                      </tr>
                      <tr className="border-b border-gray-700">
                        <td className="py-2">Profit Factor</td>
                        <td className="py-2 text-green-400">&gt; 1.5</td>
                        <td className="py-2 text-green-400">&gt; 2.0</td>
                      </tr>
                      <tr className="border-b border-gray-700">
                        <td className="py-2">Max Drawdown</td>
                        <td className="py-2 text-yellow-400">&lt; 20%</td>
                        <td className="py-2 text-green-400">&lt; 10%</td>
                      </tr>
                    </tbody>
                  </table>
                </div>
              </section>

              <section>
                <h4 className="font-semibold text-lg mb-2 text-bitcoin-orange">
                  4. Model Training
                </h4>
                <p className="text-gray-300">
                  Train your AI model with historical data. Key parameters:
                </p>
                <ul className="list-disc list-inside text-gray-300 space-y-1 mt-2">
                  <li>
                    <strong>Episodes:</strong> 1000-5000 (more = better but slower)
                  </li>
                  <li>
                    <strong>Learning Rate:</strong> 0.0001 - 0.001 (lower = more stable)
                  </li>
                  <li>
                    <strong>Batch Size:</strong> 32 - 256
                  </li>
                </ul>
              </section>
            </div>
          </Card>

          <Card className="p-6">
            <h3 className="text-xl font-bold mb-4 flex items-center gap-2">
              <Info className="text-bitcoin-orange" />
              Understanding the Dashboard
            </h3>

            <div className="space-y-4 text-gray-300">
              <div className="bg-gray-800 p-4 rounded-lg">
                <h4 className="font-semibold mb-2">📊 Equity Curve</h4>
                <p>{t('tooltips.equityCurve')}</p>
              </div>

              <div className="bg-gray-800 p-4 rounded-lg">
                <h4 className="font-semibold mb-2">📉 Drawdown Gauge</h4>
                <p>{t('tooltips.drawdown')}</p>
              </div>

              <div className="bg-gray-800 p-4 rounded-lg">
                <h4 className="font-semibold mb-2">🎯 Trading Signal</h4>
                <p>AI-generated trading signals (LONG/SHORT/FLAT) with confidence levels</p>
              </div>

              <div className="bg-gray-800 p-4 rounded-lg">
                <h4 className="font-semibold mb-2">💰 Positions & Orders</h4>
                <p>Monitor your open positions and pending orders in real-time</p>
              </div>
            </div>
          </Card>
        </div>
      )}
    </div>
  );
});

export default HelpDocumentation;
