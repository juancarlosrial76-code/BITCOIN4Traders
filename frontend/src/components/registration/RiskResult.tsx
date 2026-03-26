import { useState } from 'react';
import type { RiskProfileResponse } from '../../api/userProfiling';

interface RiskResultProps {
  profile: RiskProfileResponse;
  onConfirm: (consentGiven: boolean) => void;
  language: 'de' | 'en';
}

type Category = RiskProfileResponse['category'];

const CATEGORY_COLORS: Record<Category, { bg: string; border: string; text: string; badge: string }> = {
  KONSERVATIV: {
    bg: 'bg-blue-500/10',
    border: 'border-blue-500/40',
    text: 'text-blue-400',
    badge: 'bg-blue-500/20 border-blue-500/40 text-blue-300',
  },
  AUSGEWOGEN: {
    bg: 'bg-green-500/10',
    border: 'border-green-500/40',
    text: 'text-green-400',
    badge: 'bg-green-500/20 border-green-500/40 text-green-300',
  },
  WACHSTUM: {
    bg: 'bg-orange-500/10',
    border: 'border-orange-500/40',
    text: 'text-orange-400',
    badge: 'bg-orange-500/20 border-orange-500/40 text-orange-300',
  },
  SPEKULATIV: {
    bg: 'bg-red-500/10',
    border: 'border-red-500/40',
    text: 'text-red-400',
    badge: 'bg-red-500/20 border-red-500/40 text-red-300',
  },
};

interface Translations {
  categoryLabels: Record<Category, string>;
  categoryDescriptions: Record<Category, string>;
  continuumScore: string;
  tScore: string;
  dimensions: Record<string, string>;
  riskParamsTitle: string;
  riskParamLabels: Record<string, string>;
  summaryTitle: string;
  maxLossTitle: string;
  speedingWarning: string;
  riskWarning: string;
  consent1: (category: string) => string;
  consent2: string;
  confirmButton: string;
  confirmLoading: string;
  assessedAt: string;
}

const TRANSLATIONS: Record<'de' | 'en', Translations> = {
  de: {
    categoryLabels: {
      KONSERVATIV: 'Konservativ',
      AUSGEWOGEN: 'Ausgewogen',
      WACHSTUM: 'Wachstum',
      SPEKULATIV: 'Spekulativ',
    },
    categoryDescriptions: {
      KONSERVATIV: 'Sie bevorzugen Kapitalerhalt mit minimalem Risiko',
      AUSGEWOGEN: 'Sie suchen eine Balance zwischen Wachstum und Sicherheit',
      WACHSTUM: 'Sie akzeptieren höheres Risiko für überdurchschnittliche Renditen',
      SPEKULATIV: 'Sie sind bereit, erhebliche Risiken für maximale Gewinne einzugehen',
    },
    continuumScore: 'Gesamt-Score',
    tScore: 'T-Score',
    dimensions: {
      tolerance: 'Risikotoleranz',
      capacity: 'Risikokapazität',
      knowledge: 'Finanzwissen',
      horizon: 'Anlagehorizont',
      bias: 'Verhaltenstendenzen',
    },
    riskParamsTitle: 'Risikoparameter',
    riskParamLabels: {
      max_position_size: 'Max. Positionsgröße',
      kelly_fraction: 'Kelly-Fraction',
      max_drawdown_per_session: 'Max. Drawdown/Session',
      max_consecutive_losses: 'Max. Verluste in Folge',
      volatility_target: 'Volatilitätsziel',
    },
    summaryTitle: 'Zusammenfassung',
    maxLossTitle: 'Verlustbeispiel',
    speedingWarning:
      'Achtung: Die Befragung wurde sehr schnell ausgefüllt. Für ein genaues Profil empfehlen wir eine erneute Durchführung.',
    riskWarning:
      'RISIKOHINWEIS: Trading mit dem BITCOIN4Traders Bot beinhaltet erhebliche Risiken. Sie können Ihr gesamtes eingesetztes Kapital verlieren. Vergangene Performance ist kein Indikator für zukünftige Ergebnisse. Dies stellt keine Finanzberatung im Sinne des WpHG dar. Alle Handelsentscheidungen liegen in Ihrer persönlichen Verantwortung.',
    consent1: (category: string) =>
      `Ich verstehe, dass mein Risikoprofil ${category} bedeutet, dass ich Trading-Risiken akzeptiere und diese meiner finanziellen Situation entsprechen.`,
    consent2:
      'Ich habe die Risikowarnung gelesen und stimme zu, dass alle Handelsentscheidungen in meiner persönlichen Verantwortung liegen.',
    confirmButton: 'Registrierung abschließen',
    confirmLoading: 'Wird gespeichert...',
    assessedAt: 'Bewertet am',
  },
  en: {
    categoryLabels: {
      KONSERVATIV: 'Conservative',
      AUSGEWOGEN: 'Balanced',
      WACHSTUM: 'Growth',
      SPEKULATIV: 'Speculative',
    },
    categoryDescriptions: {
      KONSERVATIV: 'You prefer capital preservation with minimal risk',
      AUSGEWOGEN: 'You seek a balance between growth and security',
      WACHSTUM: 'You accept higher risk for above-average returns',
      SPEKULATIV: 'You are willing to take substantial risks for maximum gains',
    },
    continuumScore: 'Overall Score',
    tScore: 'T-Score',
    dimensions: {
      tolerance: 'Risk Tolerance',
      capacity: 'Risk Capacity',
      knowledge: 'Financial Knowledge',
      horizon: 'Investment Horizon',
      bias: 'Behavioral Tendencies',
    },
    riskParamsTitle: 'Risk Parameters',
    riskParamLabels: {
      max_position_size: 'Max. Position Size',
      kelly_fraction: 'Kelly Fraction',
      max_drawdown_per_session: 'Max. Drawdown/Session',
      max_consecutive_losses: 'Max. Consecutive Losses',
      volatility_target: 'Volatility Target',
    },
    summaryTitle: 'Summary',
    maxLossTitle: 'Loss Example',
    speedingWarning:
      'Warning: The assessment was completed very quickly. For an accurate profile, we recommend redoing the assessment.',
    riskWarning:
      'RISK WARNING: Trading with the BITCOIN4Traders Bot involves substantial risks. You may lose all of your invested capital. Past performance is not indicative of future results. This does not constitute financial advice. All trading decisions are your personal responsibility.',
    consent1: (category: string) =>
      `I understand that my risk profile ${category} means that I accept trading risks and that these correspond to my financial situation.`,
    consent2:
      'I have read the risk warning and agree that all trading decisions are my personal responsibility.',
    confirmButton: 'Complete Registration',
    confirmLoading: 'Saving...',
    assessedAt: 'Assessed on',
  },
};

interface DimensionBarProps {
  label: string;
  value: number;
  colorClass: string;
}

function DimensionBar({ label, value, colorClass }: DimensionBarProps) {
  const clamped = Math.min(100, Math.max(0, value));
  return (
    <div>
      <div className="flex justify-between items-center mb-1">
        <span className="text-sm text-text-secondary">{label}</span>
        <span className="text-sm font-medium text-text-primary">{clamped.toFixed(1)}</span>
      </div>
      <div className="w-full h-2 bg-background rounded-full overflow-hidden">
        <div
          className={`h-full rounded-full transition-all duration-500 ${colorClass}`}
          style={{ width: `${clamped}%` }}
        />
      </div>
    </div>
  );
}

function formatParamValue(key: string, value: number): string {
  if (key === 'max_position_size' || key === 'kelly_fraction' || key === 'volatility_target') {
    return `${(value * 100).toFixed(1)}%`;
  }
  if (key === 'max_drawdown_per_session') {
    return `${(value * 100).toFixed(1)}%`;
  }
  if (key === 'max_consecutive_losses') {
    return value.toString();
  }
  return value.toString();
}

export function RiskResult({ profile, onConfirm, language }: RiskResultProps) {
  const [consent1, setConsent1] = useState(false);
  const [consent2, setConsent2] = useState(false);
  const [isConfirming, setIsConfirming] = useState(false);

  const t = TRANSLATIONS[language];
  const colors = CATEGORY_COLORS[profile.category];
  const categoryLabel = t.categoryLabels[profile.category];
  const categoryDesc = t.categoryDescriptions[profile.category];

  const bothConsented = consent1 && consent2;

  const handleConfirm = async () => {
    if (!bothConsented) return;
    setIsConfirming(true);
    try {
      await onConfirm(true);
    } finally {
      setIsConfirming(false);
    }
  };

  const dimensionEntries = Object.entries(profile.dimension_scores) as [
    keyof typeof profile.dimension_scores,
    number,
  ][];

  const riskParamEntries = Object.entries(profile.risk_params) as [
    keyof typeof profile.risk_params,
    number,
  ][];

  const assessedDate = new Date(profile.assessed_at).toLocaleDateString(
    language === 'de' ? 'de-DE' : 'en-US',
    { year: 'numeric', month: 'long', day: 'numeric' }
  );

  return (
    <div className="space-y-6">
      {/* Speeding warning */}
      {profile.speeding_detected && (
        <div className="flex items-start gap-3 p-4 bg-yellow-500/10 border border-yellow-500/30 rounded-xl">
          <svg
            className="w-5 h-5 text-yellow-400 flex-shrink-0 mt-0.5"
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
            strokeWidth={2}
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              d="M12 9v3.75m-9.303 3.376c-.866 1.5.217 3.374 1.948 3.374h14.71c1.73 0 2.813-1.874 1.948-3.374L13.949 3.378c-.866-1.5-3.032-1.5-3.898 0L2.697 16.126zM12 15.75h.007v.008H12v-.008z"
            />
          </svg>
          <p className="text-sm text-yellow-300">{t.speedingWarning}</p>
        </div>
      )}

      {/* Category badge */}
      <div className={`rounded-xl border p-6 ${colors.bg} ${colors.border}`}>
        <div className="flex flex-col sm:flex-row sm:items-center gap-4">
          <div className={`inline-flex px-4 py-2 rounded-full border text-lg font-bold ${colors.badge}`}>
            {categoryLabel}
          </div>
          <div>
            <p className={`font-medium ${colors.text}`}>{categoryDesc}</p>
            <p className="text-xs text-text-muted mt-1">
              {t.assessedAt}: {assessedDate}
            </p>
          </div>
        </div>
      </div>

      {/* Continuum gauge */}
      <div className="bg-card border border-border rounded-xl p-5">
        <div className="flex justify-between items-center mb-3">
          <span className="text-sm font-medium text-text-secondary">{t.continuumScore}</span>
          <span className={`text-xl font-bold ${colors.text}`}>
            {profile.continuum_score.toFixed(1)}
            <span className="text-sm text-text-muted font-normal"> / 100</span>
          </span>
        </div>
        <div className="relative w-full h-4 rounded-full overflow-hidden">
          {/* gradient track */}
          <div className="absolute inset-0 bg-gradient-to-r from-blue-500 via-green-500 via-orange-500 to-red-500 opacity-20 rounded-full" />
          <div className="absolute inset-0 bg-background rounded-full" style={{ left: `${profile.continuum_score}%` }} />
          <div
            className={`h-full rounded-full transition-all duration-700 ${
              profile.category === 'KONSERVATIV'
                ? 'bg-gradient-to-r from-blue-500 to-blue-400'
                : profile.category === 'AUSGEWOGEN'
                  ? 'bg-gradient-to-r from-blue-400 to-green-500'
                  : profile.category === 'WACHSTUM'
                    ? 'bg-gradient-to-r from-green-500 to-orange-500'
                    : 'bg-gradient-to-r from-orange-500 to-red-500'
            }`}
            style={{ width: `${profile.continuum_score}%` }}
          />
        </div>
        <div className="flex justify-between text-xs text-text-muted mt-1">
          <span>{language === 'de' ? 'Konservativ' : 'Conservative'}</span>
          <span>{language === 'de' ? 'Spekulativ' : 'Speculative'}</span>
        </div>

        <div className="mt-3 pt-3 border-t border-border flex items-center gap-2">
          <span className="text-sm text-text-muted">{t.tScore}:</span>
          <span className="text-sm font-semibold text-text-primary">{profile.t_score.toFixed(1)}</span>
        </div>
      </div>

      {/* Dimension scores */}
      <div className="bg-card border border-border rounded-xl p-5">
        <h4 className="font-semibold text-text-primary mb-4">
          {language === 'de' ? 'Dimensionswerte' : 'Dimension Scores'}
        </h4>
        <div className="space-y-4">
          {dimensionEntries.map(([key, value]) => (
            <DimensionBar
              key={key}
              label={t.dimensions[key] ?? key}
              value={value}
              colorClass={colors.text.replace('text-', 'bg-').replace('-400', '-500')}
            />
          ))}
        </div>
      </div>

      {/* Risk parameters */}
      <div className="bg-card border border-border rounded-xl p-5">
        <h4 className="font-semibold text-text-primary mb-4">{t.riskParamsTitle}</h4>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <tbody className="divide-y divide-border">
              {riskParamEntries.map(([key, value]) => (
                <tr key={key} className="py-2">
                  <td className="py-2.5 pr-4 text-text-secondary">
                    {t.riskParamLabels[key] ?? key}
                  </td>
                  <td className={`py-2.5 text-right font-semibold font-mono ${colors.text}`}>
                    {formatParamValue(key, value)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Summary text */}
      {profile.risk_summary_text && (
        <div className="bg-card border border-border rounded-xl p-5">
          <h4 className="font-semibold text-text-primary mb-3">{t.summaryTitle}</h4>
          <p className="text-sm text-text-secondary leading-relaxed">{profile.risk_summary_text}</p>
        </div>
      )}

      {/* Max loss example */}
      {profile.max_loss_example && (
        <div className="bg-red-500/5 border border-red-500/20 rounded-xl p-5">
          <h4 className="font-semibold text-red-400 mb-3">{t.maxLossTitle}</h4>
          <p className="text-sm text-text-secondary leading-relaxed">{profile.max_loss_example}</p>
        </div>
      )}

      {/* Risk warning */}
      <div className="bg-yellow-500/5 border border-yellow-500/20 rounded-xl p-5">
        <div className="flex items-start gap-3">
          <svg
            className="w-5 h-5 text-yellow-400 flex-shrink-0 mt-0.5"
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
            strokeWidth={2}
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              d="M12 9v3.75m-9.303 3.376c-.866 1.5.217 3.374 1.948 3.374h14.71c1.73 0 2.813-1.874 1.948-3.374L13.949 3.378c-.866-1.5-3.032-1.5-3.898 0L2.697 16.126zM12 15.75h.007v.008H12v-.008z"
            />
          </svg>
          <p className="text-xs text-yellow-200 leading-relaxed">{t.riskWarning}</p>
        </div>
      </div>

      {/* Consent checkboxes */}
      <div className="bg-card border border-border rounded-xl p-5 space-y-4">
        <label className="flex items-start gap-3 cursor-pointer group">
          <div className="mt-0.5 flex-shrink-0">
            <input
              type="checkbox"
              checked={consent1}
              onChange={e => setConsent1(e.target.checked)}
              className="w-4 h-4 rounded border-border bg-background accent-bitcoin-orange cursor-pointer"
            />
          </div>
          <span className="text-sm text-text-secondary leading-relaxed group-hover:text-text-primary transition-colors">
            {t.consent1(categoryLabel)}
          </span>
        </label>

        <label className="flex items-start gap-3 cursor-pointer group">
          <div className="mt-0.5 flex-shrink-0">
            <input
              type="checkbox"
              checked={consent2}
              onChange={e => setConsent2(e.target.checked)}
              className="w-4 h-4 rounded border-border bg-background accent-bitcoin-orange cursor-pointer"
            />
          </div>
          <span className="text-sm text-text-secondary leading-relaxed group-hover:text-text-primary transition-colors">
            {t.consent2}
          </span>
        </label>
      </div>

      {/* Confirm button */}
      <button
        onClick={handleConfirm}
        disabled={!bothConsented || isConfirming}
        className="w-full flex items-center justify-center gap-2 py-3.5 px-6 rounded-xl bg-bitcoin-orange hover:bg-bitcoin-orange/90 disabled:opacity-40 disabled:cursor-not-allowed text-white font-semibold text-base transition-colors focus:outline-none focus:ring-2 focus:ring-bitcoin-orange/50"
      >
        {isConfirming ? (
          <>
            <div className="animate-spin rounded-full h-4 w-4 border-t-2 border-b-2 border-white" />
            {t.confirmLoading}
          </>
        ) : (
          <>
            <svg
              className="w-5 h-5"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
              strokeWidth={2}
            >
              <path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" />
            </svg>
            {t.confirmButton}
          </>
        )}
      </button>
    </div>
  );
}
