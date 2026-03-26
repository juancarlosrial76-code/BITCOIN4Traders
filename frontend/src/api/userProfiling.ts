import { fetchWithRetry } from './client';

export interface AnswerSubmission {
  question_id: number;
  answer_index: number;
  time_taken_sec: number;
}

export interface AssessmentRequest {
  answers: AnswerSubmission[];
  consent_given: boolean;
  language: string;
}

export interface RiskProfileResponse {
  profile_id: string;
  category: 'KONSERVATIV' | 'AUSGEWOGEN' | 'WACHSTUM' | 'SPEKULATIV';
  continuum_score: number;
  t_score: number;
  dimension_scores: {
    tolerance: number;
    capacity: number;
    knowledge: number;
    horizon: number;
    bias: number;
  };
  risk_params: {
    max_position_size: number;
    kelly_fraction: number;
    max_drawdown_per_session: number;
    max_consecutive_losses: number;
    volatility_target: number;
  };
  assessed_at: string;
  speeding_detected: boolean;
  risk_summary_text: string;
  max_loss_example: string;
}

export interface QuestionData {
  id: number;
  dimension: string;
  text: string;
  options: string[];
}

export const userProfilingApi = {
  getQuestionnaire: (lang: string) =>
    fetchWithRetry<QuestionData[]>(`/api/user-profiling/questionnaire?lang=${lang}`, {}),

  submitAssessment: (data: AssessmentRequest) =>
    fetchWithRetry<RiskProfileResponse>('/api/user-profiling/submit', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data),
    }),

  getProfile: () =>
    fetchWithRetry<RiskProfileResponse>('/api/user-profiling/profile', {}),
};
