/**
 * Internationalization (i18n) Configuration
 *
 * This file defines the language configuration for BITCOIN4Traders frontend.
 * Supports 14 most spoken languages for global accessibility.
 *
 * @documentation
 * - i18next: https://www.i18next.com/
 * - react-i18next: https://react.i18next.com/
 * - Language Detection: https://github.com/i18next/i18next-browser-languageDetector
 */

export interface LanguageConfig {
  code: string;
  name: string;
  nativeName: string;
  flag: string;
  dir: 'ltr' | 'rtl';
  dateFormat: string;
  numberFormat: {
    decimal: string;
    thousand: string;
  };
}

export const languages: LanguageConfig[] = [
  {
    code: 'en',
    name: 'English',
    nativeName: 'English',
    flag: '🇺🇸',
    dir: 'ltr',
    dateFormat: 'MM/DD/YYYY',
    numberFormat: { decimal: '.', thousand: ',' },
  },
  {
    code: 'de',
    name: 'German',
    nativeName: 'Deutsch',
    flag: '🇩🇪',
    dir: 'ltr',
    dateFormat: 'DD.MM.YYYY',
    numberFormat: { decimal: ',', thousand: '.' },
  },
  {
    code: 'es',
    name: 'Spanish',
    nativeName: 'Español',
    flag: '🇪🇸',
    dir: 'ltr',
    dateFormat: 'DD/MM/YYYY',
    numberFormat: { decimal: ',', thousand: '.' },
  },
  {
    code: 'fr',
    name: 'French',
    nativeName: 'Français',
    flag: '🇫🇷',
    dir: 'ltr',
    dateFormat: 'DD/MM/YYYY',
    numberFormat: { decimal: ',', thousand: ' ' },
  },
  {
    code: 'it',
    name: 'Italian',
    nativeName: 'Italiano',
    flag: '🇮🇹',
    dir: 'ltr',
    dateFormat: 'DD/MM/YYYY',
    numberFormat: { decimal: ',', thousand: '.' },
  },
  {
    code: 'pt',
    name: 'Portuguese',
    nativeName: 'Português',
    flag: '🇧🇷',
    dir: 'ltr',
    dateFormat: 'DD/MM/YYYY',
    numberFormat: { decimal: ',', thousand: '.' },
  },
  {
    code: 'nl',
    name: 'Dutch',
    nativeName: 'Nederlands',
    flag: '🇳🇱',
    dir: 'ltr',
    dateFormat: 'DD-MM-YYYY',
    numberFormat: { decimal: ',', thousand: '.' },
  },
  {
    code: 'pl',
    name: 'Polish',
    nativeName: 'Polski',
    flag: '🇵🇱',
    dir: 'ltr',
    dateFormat: 'DD.MM.YYYY',
    numberFormat: { decimal: ',', thousand: ' ' },
  },
  {
    code: 'sv',
    name: 'Swedish',
    nativeName: 'Svenska',
    flag: '🇸🇪',
    dir: 'ltr',
    dateFormat: 'YYYY-MM-DD',
    numberFormat: { decimal: ',', thousand: ' ' },
  },
  {
    code: 'da',
    name: 'Danish',
    nativeName: 'Dansk',
    flag: '🇩🇰',
    dir: 'ltr',
    dateFormat: 'DD-MM-YYYY',
    numberFormat: { decimal: ',', thousand: '.' },
  },
  {
    code: 'fi',
    name: 'Finnish',
    nativeName: 'Suomi',
    flag: '🇫🇮',
    dir: 'ltr',
    dateFormat: 'DD.MM.YYYY',
    numberFormat: { decimal: ',', thousand: ' ' },
  },
  {
    code: 'no',
    name: 'Norwegian',
    nativeName: 'Norsk',
    flag: '🇳🇴',
    dir: 'ltr',
    dateFormat: 'DD.MM.YYYY',
    numberFormat: { decimal: ',', thousand: ' ' },
  },
  {
    code: 'cs',
    name: 'Czech',
    nativeName: 'Čeština',
    flag: '🇨🇿',
    dir: 'ltr',
    dateFormat: 'DD.MM.YYYY',
    numberFormat: { decimal: ',', thousand: ' ' },
  },
  {
    code: 'hu',
    name: 'Hungarian',
    nativeName: 'Magyar',
    flag: '🇭🇺',
    dir: 'ltr',
    dateFormat: 'YYYY.MM.DD',
    numberFormat: { decimal: ',', thousand: ' ' },
  },
  {
    code: 'ro',
    name: 'Romanian',
    nativeName: 'Română',
    flag: '🇷🇴',
    dir: 'ltr',
    dateFormat: 'DD.MM.YYYY',
    numberFormat: { decimal: ',', thousand: ' ' },
  },
  {
    code: 'uk',
    name: 'Ukrainian',
    nativeName: 'Українська',
    flag: '🇺🇦',
    dir: 'ltr',
    dateFormat: 'DD.MM.YYYY',
    numberFormat: { decimal: ',', thousand: ' ' },
  },
  {
    code: 'ru',
    name: 'Russian',
    nativeName: 'Русский',
    flag: '🇷🇺',
    dir: 'ltr',
    dateFormat: 'DD.MM.YYYY',
    numberFormat: { decimal: ',', thousand: ' ' },
  },
  {
    code: 'zh',
    name: 'Chinese (Simplified)',
    nativeName: '简体中文',
    flag: '🇨🇳',
    dir: 'ltr',
    dateFormat: 'YYYY/MM/DD',
    numberFormat: { decimal: '.', thousand: ',' },
  },
  {
    code: 'zh-TW',
    name: 'Chinese (Traditional)',
    nativeName: '繁體中文',
    flag: '🇹🇼',
    dir: 'ltr',
    dateFormat: 'YYYY/MM/DD',
    numberFormat: { decimal: '.', thousand: ',' },
  },
  {
    code: 'ja',
    name: 'Japanese',
    nativeName: '日本語',
    flag: '🇯🇵',
    dir: 'ltr',
    dateFormat: 'YYYY/MM/DD',
    numberFormat: { decimal: '.', thousand: ',' },
  },
  {
    code: 'ko',
    name: 'Korean',
    nativeName: '한국어',
    flag: '🇰🇷',
    dir: 'ltr',
    dateFormat: 'YYYY.MM.DD',
    numberFormat: { decimal: '.', thousand: ',' },
  },
  {
    code: 'ar',
    name: 'Arabic',
    nativeName: 'العربية',
    flag: '🇸🇦',
    dir: 'rtl',
    dateFormat: 'DD/MM/YYYY',
    numberFormat: { decimal: '٫', thousand: '٬' },
  },
  {
    code: 'he',
    name: 'Hebrew',
    nativeName: 'עברית',
    flag: '🇮🇱',
    dir: 'rtl',
    dateFormat: 'DD/MM/YYYY',
    numberFormat: { decimal: '.', thousand: ',' },
  },
  {
    code: 'hi',
    name: 'Hindi',
    nativeName: 'हिन्दी',
    flag: '🇮🇳',
    dir: 'ltr',
    dateFormat: 'DD/MM/YYYY',
    numberFormat: { decimal: '.', thousand: ',' },
  },
  {
    code: 'th',
    name: 'Thai',
    nativeName: 'ไทย',
    flag: '🇹🇭',
    dir: 'ltr',
    dateFormat: 'DD/MM/YYYY',
    numberFormat: { decimal: '.', thousand: ',' },
  },
  {
    code: 'vi',
    name: 'Vietnamese',
    nativeName: 'Tiếng Việt',
    flag: '🇻🇳',
    dir: 'ltr',
    dateFormat: 'DD/MM/YYYY',
    numberFormat: { decimal: ',', thousand: '.' },
  },
  {
    code: 'id',
    name: 'Indonesian',
    nativeName: 'Bahasa Indonesia',
    flag: '🇮🇩',
    dir: 'ltr',
    dateFormat: 'DD/MM/YYYY',
    numberFormat: { decimal: ',', thousand: '.' },
  },
  {
    code: 'ms',
    name: 'Malay',
    nativeName: 'Bahasa Melayu',
    flag: '🇲🇾',
    dir: 'ltr',
    dateFormat: 'DD/MM/YYYY',
    numberFormat: { decimal: '.', thousand: ',' },
  },
  {
    code: 'tr',
    name: 'Turkish',
    nativeName: 'Türkçe',
    flag: '🇹🇷',
    dir: 'ltr',
    dateFormat: 'DD.MM.YYYY',
    numberFormat: { decimal: ',', thousand: '.' },
  },
  {
    code: 'el',
    name: 'Greek',
    nativeName: 'Ελληνικά',
    flag: '🇬🇷',
    dir: 'ltr',
    dateFormat: 'DD/MM/YYYY',
    numberFormat: { decimal: ',', thousand: '.' },
  },
];

export const defaultLanguage = 'en';

export const supportedLanguages = languages.map(lang => lang.code);

export function getLanguageConfig(code: string): LanguageConfig {
  return languages.find(lang => lang.code === code) || languages[0];
}

export function isRTL(code: string): boolean {
  return getLanguageConfig(code).dir === 'rtl';
}
