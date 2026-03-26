import { useState, useEffect, useCallback, useRef } from 'react';
import type { QuestionData, AnswerSubmission } from '../../api/userProfiling';

interface RiskQuestionnaireProps {
  questions: QuestionData[];
  onComplete: (answers: AnswerSubmission[]) => void;
  language: 'de' | 'en';
}

type DimensionKey = 'tolerance' | 'capacity' | 'knowledge' | 'horizon' | 'bias';

const DIMENSION_LABELS: Record<'de' | 'en', Record<DimensionKey, string>> = {
  de: {
    tolerance: 'Risikotoleranz',
    capacity: 'Risikokapazität',
    knowledge: 'Finanzwissen',
    horizon: 'Anlagehorizont',
    bias: 'Verhaltenstendenzen',
  },
  en: {
    tolerance: 'Risk Tolerance',
    capacity: 'Risk Capacity',
    knowledge: 'Financial Knowledge',
    horizon: 'Investment Horizon',
    bias: 'Behavioral Tendencies',
  },
};

const OPTION_LABELS = ['a', 'b', 'c', 'd'];

function getDimensionKey(dimension: string): DimensionKey {
  const lower = dimension.toLowerCase();
  if (lower.includes('toler') || lower === 'tolerance') return 'tolerance';
  if (lower.includes('capac') || lower === 'capacity') return 'capacity';
  if (lower.includes('know') || lower === 'knowledge') return 'knowledge';
  if (lower.includes('horiz') || lower === 'horizon') return 'horizon';
  return 'bias';
}

function getUniqueDimensions(questions: QuestionData[]): string[] {
  const seen = new Set<string>();
  const order: string[] = [];
  for (const q of questions) {
    if (!seen.has(q.dimension)) {
      seen.add(q.dimension);
      order.push(q.dimension);
    }
  }
  return order;
}

export function RiskQuestionnaire({ questions, onComplete, language }: RiskQuestionnaireProps) {
  const [currentIndex, setCurrentIndex] = useState(0);
  const [selectedAnswers, setSelectedAnswers] = useState<Record<number, number>>({});
  const [timings, setTimings] = useState<Record<number, number>>({});
  const questionStartTime = useRef<number>(Date.now());

  const dimensions = getUniqueDimensions(questions);
  const labels = DIMENSION_LABELS[language];

  const currentQuestion = questions[currentIndex];
  const totalQuestions = questions.length;

  const ui = language === 'de'
    ? {
        questionOf: `Frage ${currentIndex + 1} von ${totalQuestions}`,
        previous: 'Zurück',
        next: 'Weiter',
        submit: 'Auswertung starten',
      }
    : {
        questionOf: `Question ${currentIndex + 1} of ${totalQuestions}`,
        previous: 'Previous',
        next: 'Next',
        submit: 'Start Evaluation',
      };

  // Reset timer whenever question changes
  useEffect(() => {
    questionStartTime.current = Date.now();
  }, [currentIndex]);

  const recordTiming = useCallback(() => {
    const elapsed = (Date.now() - questionStartTime.current) / 1000;
    setTimings(prev => ({ ...prev, [currentQuestion.id]: elapsed }));
  }, [currentQuestion]);

  const handleSelectAnswer = (answerIndex: number) => {
    recordTiming();
    setSelectedAnswers(prev => ({ ...prev, [currentQuestion.id]: answerIndex }));
  };

  const handleNext = () => {
    if (selectedAnswers[currentQuestion.id] === undefined) return;
    if (currentIndex < totalQuestions - 1) {
      setCurrentIndex(prev => prev + 1);
    }
  };

  const handlePrevious = () => {
    if (currentIndex > 0) {
      setCurrentIndex(prev => prev - 1);
    }
  };

  const handleSubmit = () => {
    if (selectedAnswers[currentQuestion.id] === undefined) return;

    const answers: AnswerSubmission[] = questions.map(q => ({
      question_id: q.id,
      answer_index: selectedAnswers[q.id] ?? 0,
      time_taken_sec: timings[q.id] ?? 0,
    }));

    onComplete(answers);
  };

  const isLastQuestion = currentIndex === totalQuestions - 1;
  const currentAnswerSelected = selectedAnswers[currentQuestion.id] !== undefined;

  // Determine active dimension index
  const currentDimensionKey = getDimensionKey(currentQuestion.dimension);
  const activeDimensionIndex = dimensions.findIndex(d => d === currentQuestion.dimension);

  // Progress within total
  const progressPercent = ((currentIndex + 1) / totalQuestions) * 100;

  return (
    <div className="space-y-6">
      {/* Dimension progress stepper */}
      <div className="flex items-center gap-1 overflow-x-auto pb-1">
        {dimensions.map((dim, idx) => {
          const dimKey = getDimensionKey(dim);
          const dimLabel = labels[dimKey];
          const isActive = idx === activeDimensionIndex;
          const isPast = idx < activeDimensionIndex;

          return (
            <div key={dim} className="flex items-center flex-shrink-0">
              <div className="flex flex-col items-center gap-1">
                <div
                  className={`w-8 h-8 rounded-full flex items-center justify-center text-xs font-bold transition-colors
                    ${isActive ? 'bg-bitcoin-orange text-white ring-2 ring-bitcoin-orange/30' : ''}
                    ${isPast ? 'bg-green-500/20 text-green-400 border border-green-500/40' : ''}
                    ${!isActive && !isPast ? 'bg-background border border-border text-text-muted' : ''}
                  `}
                >
                  {isPast ? (
                    <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2.5}>
                      <path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" />
                    </svg>
                  ) : (
                    idx + 1
                  )}
                </div>
                <span
                  className={`text-xs whitespace-nowrap ${
                    isActive ? 'text-bitcoin-orange font-medium' : isPast ? 'text-green-400' : 'text-text-muted'
                  }`}
                >
                  {dimLabel}
                </span>
              </div>
              {idx < dimensions.length - 1 && (
                <div
                  className={`h-0.5 w-6 mx-1 mt-[-14px] transition-colors ${
                    isPast ? 'bg-green-500/40' : 'bg-border'
                  }`}
                />
              )}
            </div>
          );
        })}
      </div>

      {/* Overall progress bar */}
      <div>
        <div className="flex justify-between items-center mb-1">
          <span className="text-sm font-medium text-text-secondary">{ui.questionOf}</span>
          <span className="text-xs text-text-muted">{Math.round(progressPercent)}%</span>
        </div>
        <div className="w-full h-1.5 bg-background rounded-full overflow-hidden">
          <div
            className="h-full bg-bitcoin-orange rounded-full transition-all duration-300"
            style={{ width: `${progressPercent}%` }}
          />
        </div>
      </div>

      {/* Dimension badge */}
      <div className="flex items-center gap-2">
        <span className="px-3 py-1 bg-bitcoin-orange/10 border border-bitcoin-orange/20 rounded-full text-xs font-medium text-bitcoin-orange">
          {labels[currentDimensionKey]}
        </span>
      </div>

      {/* Question card */}
      <div className="bg-card border border-border rounded-xl p-6">
        <p className="text-lg font-semibold text-text-primary leading-relaxed">
          {currentQuestion.text}
        </p>
      </div>

      {/* Answer options */}
      <div className="space-y-3">
        {currentQuestion.options.map((option, idx) => {
          const isSelected = selectedAnswers[currentQuestion.id] === idx;
          return (
            <button
              key={idx}
              onClick={() => handleSelectAnswer(idx)}
              className={`w-full flex items-start gap-4 p-4 rounded-xl border text-left transition-all duration-150 focus:outline-none focus:ring-2 focus:ring-bitcoin-orange/50
                ${
                  isSelected
                    ? 'bg-bitcoin-orange/10 border-bitcoin-orange/50 text-text-primary'
                    : 'bg-background border-border text-text-secondary hover:border-bitcoin-orange/30 hover:bg-bitcoin-orange/5'
                }
              `}
            >
              <span
                className={`flex-shrink-0 w-7 h-7 rounded-full flex items-center justify-center text-xs font-bold uppercase border transition-colors
                  ${
                    isSelected
                      ? 'bg-bitcoin-orange border-bitcoin-orange text-white'
                      : 'border-border text-text-muted'
                  }
                `}
              >
                {OPTION_LABELS[idx] ?? idx + 1}
              </span>
              <span className="leading-relaxed">{option}</span>
            </button>
          );
        })}
      </div>

      {/* Navigation */}
      <div className="flex justify-between items-center pt-2">
        <button
          onClick={handlePrevious}
          disabled={currentIndex === 0}
          className="flex items-center gap-2 px-4 py-2.5 rounded-lg border border-border text-text-secondary hover:text-text-primary hover:border-border/80 disabled:opacity-40 disabled:cursor-not-allowed transition-colors text-sm font-medium"
        >
          <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M15 19l-7-7 7-7" />
          </svg>
          {ui.previous}
        </button>

        {isLastQuestion ? (
          <button
            onClick={handleSubmit}
            disabled={!currentAnswerSelected}
            className="flex items-center gap-2 px-6 py-2.5 rounded-lg bg-bitcoin-orange hover:bg-bitcoin-orange/90 disabled:opacity-40 disabled:cursor-not-allowed text-white font-semibold transition-colors text-sm"
          >
            {ui.submit}
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" />
            </svg>
          </button>
        ) : (
          <button
            onClick={handleNext}
            disabled={!currentAnswerSelected}
            className="flex items-center gap-2 px-5 py-2.5 rounded-lg bg-bitcoin-orange hover:bg-bitcoin-orange/90 disabled:opacity-40 disabled:cursor-not-allowed text-white font-semibold transition-colors text-sm"
          >
            {ui.next}
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M9 5l7 7-7 7" />
            </svg>
          </button>
        )}
      </div>
    </div>
  );
}
