import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import { Language, languages, defaultLanguage } from '../i18n/languages';

interface I18nStore {
  language: Language;
  setLanguage: (lang: Language) => void;
}

export const useI18n = create<I18nStore>()(
  persist(
    (set) => ({
      language: defaultLanguage,
      setLanguage: (language) => set({ language }),
    }),
    {
      name: 'bt4t-language',
    }
  )
);

export { languages, type Language };
