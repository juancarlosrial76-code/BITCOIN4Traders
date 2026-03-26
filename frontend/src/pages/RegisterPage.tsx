import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { userProfilingApi } from '../api/userProfiling';
import type { QuestionData, AnswerSubmission, RiskProfileResponse } from '../api/userProfiling';
import { RiskQuestionnaire } from '../components/registration/RiskQuestionnaire';
import { RiskResult } from '../components/registration/RiskResult';

type Step = 1 | 2 | 3 | 4;
type Language = 'de' | 'en';

const LABELS = {
  de: {
    steps: ['Persönliche Daten', 'Risikobefragung', 'Risikoprofil', 'Fertig'],
    title: 'Registrierung',
    subtitle: 'Erstellen Sie Ihr wissenschaftliches Risikoprofil',
    username: 'Benutzername',
    email: 'E-Mail-Adresse',
    password: 'Passwort',
    passwordHint: 'Mindestens 8 Zeichen',
    language: 'Sprache',
    next: 'Weiter',
    back: 'Zurück',
    loading: 'Laden...',
    submitting: 'Wird ausgewertet...',
    successTitle: 'Registrierung erfolgreich!',
    successText: 'Ihr Risikoprofil wurde gespeichert. Sie werden zum Login weitergeleitet.',
    toLogin: 'Zum Login',
    errorRequired: 'Alle Felder sind erforderlich.',
    errorPassword: 'Passwort muss mindestens 8 Zeichen haben.',
    errorEmail: 'Bitte geben Sie eine gültige E-Mail-Adresse ein.',
    loadingQuestions: 'Fragen werden geladen...',
  },
  en: {
    steps: ['Personal Info', 'Risk Survey', 'Risk Profile', 'Done'],
    title: 'Registration',
    subtitle: 'Create your scientific risk profile',
    username: 'Username',
    email: 'Email Address',
    password: 'Password',
    passwordHint: 'At least 8 characters',
    language: 'Language',
    next: 'Continue',
    back: 'Back',
    loading: 'Loading...',
    submitting: 'Evaluating...',
    successTitle: 'Registration successful!',
    successText: 'Your risk profile has been saved. Redirecting to login.',
    toLogin: 'Go to Login',
    errorRequired: 'All fields are required.',
    errorPassword: 'Password must be at least 8 characters.',
    errorEmail: 'Please enter a valid email address.',
    loadingQuestions: 'Loading questions...',
  },
};

function PasswordStrength({ password }: { password: string }) {
  const score = [
    password.length >= 8,
    /[A-Z]/.test(password),
    /[0-9]/.test(password),
    /[^A-Za-z0-9]/.test(password),
  ].filter(Boolean).length;

  const colors = ['bg-red-500', 'bg-orange-500', 'bg-yellow-500', 'bg-green-500'];
  const labels = ['Schwach', 'Mäßig', 'Gut', 'Stark'];

  if (!password) return null;
  return (
    <div className="mt-1">
      <div className="flex gap-1 mb-1">
        {[0, 1, 2, 3].map(i => (
          <div
            key={i}
            className={`h-1 flex-1 rounded ${i < score ? colors[score - 1] : 'bg-border'}`}
          />
        ))}
      </div>
      <p className={`text-xs ${score < 2 ? 'text-red-400' : score < 4 ? 'text-yellow-400' : 'text-green-400'}`}>
        {labels[score - 1] ?? ''}
      </p>
    </div>
  );
}

function Stepper({ step, lang }: { step: Step; lang: Language }) {
  const labels = LABELS[lang].steps;
  return (
    <div className="flex items-center justify-center mb-8">
      {labels.map((label, i) => {
        const num = (i + 1) as Step;
        const active = num === step;
        const done = num < step;
        return (
          <div key={i} className="flex items-center">
            <div className="flex flex-col items-center">
              <div
                className={`w-8 h-8 rounded-full flex items-center justify-center text-sm font-bold transition-colors ${
                  done
                    ? 'bg-green-500 text-white'
                    : active
                    ? 'bg-bitcoin-orange text-white'
                    : 'bg-border text-text-muted'
                }`}
              >
                {done ? '✓' : num}
              </div>
              <span
                className={`text-xs mt-1 hidden sm:block ${
                  active ? 'text-bitcoin-orange font-medium' : done ? 'text-green-400' : 'text-text-muted'
                }`}
              >
                {label}
              </span>
            </div>
            {i < labels.length - 1 && (
              <div className={`w-12 sm:w-20 h-0.5 mx-1 mb-4 ${done ? 'bg-green-500' : 'bg-border'}`} />
            )}
          </div>
        );
      })}
    </div>
  );
}

export function RegisterPage() {
  const navigate = useNavigate();
  const [step, setStep] = useState<Step>(1);
  const [lang, setLang] = useState<Language>('de');
  const L = LABELS[lang];

  // Step 1 state
  const [username, setUsername] = useState('');
  const [email, setEmail]       = useState('');
  const [password, setPassword] = useState('');
  const [formError, setFormError] = useState<string | null>(null);

  // Step 2 state
  const [questions, setQuestions]   = useState<QuestionData[]>([]);
  const [loadingQ, setLoadingQ]     = useState(false);
  const [answers, setAnswers]       = useState<AnswerSubmission[]>([]);

  // Step 3 state
  const [profile, setProfile]       = useState<RiskProfileResponse | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [submitError, setSubmitError] = useState<string | null>(null);

  // Load questions when entering step 2
  useEffect(() => {
    if (step === 2 && questions.length === 0) {
      setLoadingQ(true);
      userProfilingApi.getQuestionnaire(lang)
        .then(qs => setQuestions(qs))
        .catch(() => setQuestions([]))
        .finally(() => setLoadingQ(false));
    }
  }, [step, lang, questions.length]);

  function validateStep1(): boolean {
    if (!username.trim() || !email.trim() || !password.trim()) {
      setFormError(L.errorRequired);
      return false;
    }
    if (password.length < 8) {
      setFormError(L.errorPassword);
      return false;
    }
    if (!/^[^@]+@[^@]+\.[^@]+$/.test(email)) {
      setFormError(L.errorEmail);
      return false;
    }
    setFormError(null);
    return true;
  }

  function handleStep1Next() {
    if (validateStep1()) setStep(2);
  }

  async function handleQuestionnaireComplete(completedAnswers: AnswerSubmission[]) {
    setAnswers(completedAnswers);
    setSubmitting(true);
    setSubmitError(null);
    try {
      const result = await userProfilingApi.submitAssessment({
        answers: completedAnswers,
        consent_given: true,
        language: lang,
      });
      setProfile(result);
      setStep(3);
    } catch (e: unknown) {
      setSubmitError(e instanceof Error ? e.message : 'Fehler bei der Auswertung');
    } finally {
      setSubmitting(false);
    }
  }

  function handleResultConfirm(_consentGiven: boolean) {
    setStep(4);
    setTimeout(() => navigate('/login'), 3000);
  }

  return (
    <div className="min-h-screen bg-background flex items-center justify-center p-4">
      <div className="w-full max-w-2xl">
        {/* Header */}
        <div className="text-center mb-6">
          <div className="inline-flex items-center justify-center w-14 h-14 rounded-full bg-bitcoin-orange/10 mb-3">
            <svg viewBox="0 0 24 24" className="w-7 h-7 text-bitcoin-orange fill-current">
              <path d="M23.638 14.904c-1.602 6.425-8.113 10.34-14.542 8.736C2.67 22.04-1.244 15.525.362 9.105 1.962 2.68 8.475-1.243 14.9.358c6.43 1.605 10.342 8.115 8.738 14.546z" />
              <path fill="#fff" d="M17.16 10.49c.24-1.6-.975-2.46-2.635-3.035l.54-2.156-1.315-.328-.524 2.1c-.346-.086-.7-.167-1.054-.247l.527-2.112-1.315-.327-.54 2.154c-.286-.065-.567-.13-.84-.198l.002-.007-1.815-.453-.35 1.404s.975.224.954.238c.533.133.63.486.613.766L8.79 12.68c.046.012.106.028.172.054l-.175-.044-.86 3.445c-.065.16-.23.402-.602.31.013.02-.954-.238-.954-.238L5.7 17.66l1.71.426c.32.08.633.163.941.242l-.545 2.183 1.314.328.54-2.158c.358.097.705.186 1.046.272l-.538 2.15 1.316.329.545-2.18c2.245.425 3.933.254 4.644-1.777.574-1.636-.028-2.578-1.211-3.192.861-.2 1.51-.765 1.682-1.934zm-3.012 4.224c-.408 1.636-3.168.751-4.063.53l.724-2.903c.896.224 3.766.667 3.34 2.373zm.408-4.248c-.373 1.493-2.672.734-3.418.548l.657-2.634c.746.186 3.147.534 2.762 2.086z" />
            </svg>
          </div>
          <h1 className="text-2xl font-bold text-text-primary">BITCOIN4Traders</h1>
          <p className="text-text-secondary text-sm mt-1">{L.subtitle}</p>
        </div>

        <div className="bg-card border border-border rounded-xl p-6 shadow-2xl">
          <Stepper step={step} lang={lang} />

          {/* ── Step 1: Basic Info ────────────────────────────────────── */}
          {step === 1 && (
            <div className="space-y-4">
              {/* Language selector */}
              <div className="flex justify-end mb-2">
                <select
                  value={lang}
                  onChange={e => setLang(e.target.value as Language)}
                  className="bg-background border border-border rounded-lg px-3 py-1.5 text-sm text-text-primary focus:outline-none focus:ring-2 focus:ring-bitcoin-orange/50"
                >
                  <option value="de">🇩🇪 Deutsch</option>
                  <option value="en">🇬🇧 English</option>
                </select>
              </div>

              <div>
                <label className="block text-sm font-medium text-text-secondary mb-1">{L.username}</label>
                <input
                  type="text"
                  value={username}
                  onChange={e => setUsername(e.target.value)}
                  placeholder="trader42"
                  className="w-full bg-background border border-border rounded-lg px-4 py-3 text-text-primary placeholder-text-muted focus:outline-none focus:ring-2 focus:ring-bitcoin-orange/50 focus:border-bitcoin-orange transition-colors"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-text-secondary mb-1">{L.email}</label>
                <input
                  type="email"
                  value={email}
                  onChange={e => setEmail(e.target.value)}
                  placeholder="you@example.com"
                  className="w-full bg-background border border-border rounded-lg px-4 py-3 text-text-primary placeholder-text-muted focus:outline-none focus:ring-2 focus:ring-bitcoin-orange/50 focus:border-bitcoin-orange transition-colors"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-text-secondary mb-1">{L.password}</label>
                <input
                  type="password"
                  value={password}
                  onChange={e => setPassword(e.target.value)}
                  placeholder="••••••••"
                  className="w-full bg-background border border-border rounded-lg px-4 py-3 text-text-primary placeholder-text-muted focus:outline-none focus:ring-2 focus:ring-bitcoin-orange/50 focus:border-bitcoin-orange transition-colors"
                />
                <PasswordStrength password={password} />
                <p className="text-xs text-text-muted mt-1">{L.passwordHint}</p>
              </div>

              {formError && (
                <div className="p-3 bg-red-500/10 border border-red-500/20 rounded-lg text-sm text-red-400">
                  {formError}
                </div>
              )}

              <button
                onClick={handleStep1Next}
                className="w-full bg-bitcoin-orange hover:bg-bitcoin-orange/90 text-white font-semibold py-3 rounded-lg transition-colors"
              >
                {L.next} →
              </button>

              <p className="text-center text-xs text-text-muted">
                {lang === 'de' ? 'Bereits registriert?' : 'Already registered?'}{' '}
                <a href="/login" className="text-bitcoin-orange hover:underline">
                  {lang === 'de' ? 'Zum Login' : 'Sign in'}
                </a>
              </p>
            </div>
          )}

          {/* ── Step 2: Questionnaire ─────────────────────────────────── */}
          {step === 2 && (
            <div>
              {loadingQ ? (
                <div className="text-center py-12 text-text-secondary">{L.loadingQuestions}</div>
              ) : submitting ? (
                <div className="text-center py-12 text-text-secondary">
                  <div className="animate-spin rounded-full h-8 w-8 border-t-2 border-b-2 border-bitcoin-orange mx-auto mb-3" />
                  {L.submitting}
                </div>
              ) : questions.length > 0 ? (
                <>
                  <RiskQuestionnaire
                    questions={questions}
                    onComplete={handleQuestionnaireComplete}
                    language={lang}
                  />
                  {submitError && (
                    <div className="mt-4 p-3 bg-red-500/10 border border-red-500/20 rounded-lg text-sm text-red-400">
                      {submitError}
                    </div>
                  )}
                </>
              ) : (
                <div className="text-center py-12 text-red-400">
                  {lang === 'de' ? 'Fragen konnten nicht geladen werden.' : 'Failed to load questions.'}
                </div>
              )}

              <button
                onClick={() => setStep(1)}
                className="mt-4 text-sm text-text-muted hover:text-text-secondary transition-colors"
              >
                ← {L.back}
              </button>
            </div>
          )}

          {/* ── Step 3: Result ────────────────────────────────────────── */}
          {step === 3 && profile && (
            <RiskResult
              profile={profile}
              onConfirm={handleResultConfirm}
              language={lang}
            />
          )}

          {/* ── Step 4: Success ───────────────────────────────────────── */}
          {step === 4 && (
            <div className="text-center py-8 space-y-4">
              <div className="inline-flex items-center justify-center w-16 h-16 rounded-full bg-green-500/10 mb-2">
                <svg className="w-8 h-8 text-green-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                </svg>
              </div>
              <h2 className="text-xl font-bold text-text-primary">{L.successTitle}</h2>
              <p className="text-text-secondary text-sm">{L.successText}</p>
              <button
                onClick={() => navigate('/login')}
                className="mt-4 bg-bitcoin-orange hover:bg-bitcoin-orange/90 text-white font-semibold py-2 px-6 rounded-lg transition-colors"
              >
                {L.toLogin}
              </button>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default RegisterPage;
