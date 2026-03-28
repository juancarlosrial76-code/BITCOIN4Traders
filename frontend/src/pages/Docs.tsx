import { Link } from 'react-router-dom';
import { useTranslation } from 'react-i18next';
import { Card } from '../components/ui';
import {
  Rocket,
  LineChart,
  Code,
  BookOpen,
  Wrench,
  HelpCircle,
  ArrowRight,
} from 'lucide-react';

const sections = [
  {
    path: '/docs/quickstart',
    icon: Rocket,
    color: 'text-green-400',
    bg: 'bg-green-400/10',
    titleKey: 'docs.quickstart.title',
    titleDefault: 'Quickstart',
    descKey: 'docs.quickstart.subtitle',
    descDefault: 'Set up and run BITCOIN4Traders in 5 minutes.',
  },
  {
    path: '/docs/trading-guide',
    icon: LineChart,
    color: 'text-bitcoin-orange',
    bg: 'bg-bitcoin-orange/10',
    titleKey: 'docs.tradingGuide.title',
    titleDefault: 'Trading Guide',
    descKey: 'docs.tradingGuide.subtitle',
    descDefault: 'Paper trading, live mode, risk profiles and Kelly sizing.',
  },
  {
    path: '/docs/api',
    icon: Code,
    color: 'text-blue-400',
    bg: 'bg-blue-400/10',
    titleKey: 'docs.api.title',
    titleDefault: 'API Reference',
    descKey: 'docs.api.subtitle',
    descDefault: 'All REST endpoints, request/response formats and authentication.',
  },
  {
    path: '/docs/glossary',
    icon: BookOpen,
    color: 'text-purple-400',
    bg: 'bg-purple-400/10',
    titleKey: 'docs.glossary.title',
    titleDefault: 'Glossary',
    descKey: 'docs.glossary.subtitle',
    descDefault: 'Key trading and RL terms explained simply.',
  },
  {
    path: '/docs/troubleshooting',
    icon: Wrench,
    color: 'text-yellow-400',
    bg: 'bg-yellow-400/10',
    titleKey: 'docs.troubleshooting.title',
    titleDefault: 'Troubleshooting',
    descKey: 'docs.troubleshooting.subtitle',
    descDefault: 'Common errors, fixes and diagnostic tips.',
  },
  {
    path: '/faq',
    icon: HelpCircle,
    color: 'text-red-400',
    bg: 'bg-red-400/10',
    titleKey: 'faq.title',
    titleDefault: 'FAQ',
    descKey: 'faq.subtitle',
    descDefault: 'Frequently asked questions about BITCOIN4Traders.',
  },
];

export function Docs() {
  const { t } = useTranslation();

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-3xl font-bold text-text-primary">
          {t('docs.title', 'Documentation')}
        </h1>
        <p className="text-text-secondary mt-1">
          {t('docs.subtitle', 'Everything you need to understand, configure, and run BITCOIN4Traders.')}
        </p>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
        {sections.map(section => {
          const Icon = section.icon;
          return (
            <Link key={section.path} to={section.path} className="group block">
              <Card className="h-full transition-all hover:border-bitcoin-orange/40 hover:shadow-lg">
                <div className="flex items-start gap-4">
                  <div className={`p-3 rounded-lg ${section.bg} flex-shrink-0`}>
                    <Icon size={22} className={section.color} />
                  </div>
                  <div className="flex-1 min-w-0">
                    <h2 className="font-semibold text-text-primary group-hover:text-bitcoin-orange transition-colors">
                      {t(section.titleKey, section.titleDefault)}
                    </h2>
                    <p className="text-sm text-text-secondary mt-1">
                      {t(section.descKey, section.descDefault)}
                    </p>
                  </div>
                  <ArrowRight
                    size={16}
                    className="text-text-muted group-hover:text-bitcoin-orange transition-colors flex-shrink-0 mt-1"
                  />
                </div>
              </Card>
            </Link>
          );
        })}
      </div>

      {/* Disclaimer */}
      <Card className="border-yellow-500/30 bg-yellow-500/5">
        <p className="text-sm text-yellow-400 font-medium">
          {t(
            'docs.disclaimer',
            'BITCOIN4Traders is a research project, not a regulated financial service. All trading involves risk. Past performance does not guarantee future results.'
          )}
        </p>
      </Card>
    </div>
  );
}
