/**
 * i18n Initialization
 *
 * Initializes i18next with:
 * - Language detection from browser/localStorage
 * - HTTP backend for loading translations
 * - React integration
 * - Caching for performance
 *
 * @usage
 * import './i18n';
 * import { useTranslation } from 'react-i18next';
 *
 * const { t } = useTranslation();
 * <h1>{t('common.welcome')}</h1>
 */

import i18n from 'i18next';
import { initReactI18next } from 'react-i18next';
import LanguageDetector from 'i18next-browser-languagedetector';
import Backend from 'i18next-http-backend';
import { defaultLanguage, supportedLanguages } from './config';

i18n
  .use(Backend)
  .use(LanguageDetector)
  .use(initReactI18next)
  .init({
    // Backend configuration
    backend: {
      loadPath: '/locales/{{lng}}/{{ns}}.json',
    },

    // Language detection
    detection: {
      order: ['localStorage', 'navigator', 'htmlTag'],
      caches: ['localStorage'],
      lookupLocalStorage: 'language',
    },

    // Supported languages
    supportedLngs: supportedLanguages,
    fallbackLng: defaultLanguage,
    defaultNS: 'translation',

    // React configuration
    react: {
      useSuspense: true,
      bindI18nStore: 'added',
    },

    // Interpolation
    interpolation: {
      escapeValue: false,
    },

    // Caching
    cache: {
      enabled: true,
      expirationTime: 7 * 24 * 60 * 60 * 1000, // 7 days
    },

    // Debug (set to true for development)
    debug: false,
  });

export default i18n;
