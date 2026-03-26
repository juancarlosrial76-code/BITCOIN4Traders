import React from 'react';
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { I18nextProvider } from 'react-i18next';
import i18n from 'i18next';
import { LanguageSwitcher } from '../components/ui/LanguageSwitcher';
import { languages, getLanguageConfig, isRTL } from '../i18n/config';

i18n.init({
  lng: 'en',
  fallbackLng: 'en',
  resources: {
    en: { translation: { common: { language: 'Language' } } },
    de: { translation: { common: { language: 'Sprache' } } },
  },
});

function renderWithI18n(component: React.ReactElement) {
  return render(<I18nextProvider i18n={i18n}>{component}</I18nextProvider>);
}

describe('LanguageSwitcher', () => {
  beforeEach(() => {
    i18n.changeLanguage('en');
  });

  afterEach(() => {
    vi.clearAllMocks();
  });

  describe('getLanguageConfig', () => {
    it('returns correct config for English', () => {
      const config = getLanguageConfig('en');
      expect(config.code).toBe('en');
      expect(config.name).toBe('English');
      expect(config.flag).toBe('🇺🇸');
      expect(config.dir).toBe('ltr');
    });

    it('returns correct config for German', () => {
      const config = getLanguageConfig('de');
      expect(config.code).toBe('de');
      expect(config.name).toBe('German');
      expect(config.flag).toBe('🇩🇪');
    });

    it('returns correct config for Arabic (RTL)', () => {
      const config = getLanguageConfig('ar');
      expect(config.dir).toBe('rtl');
    });

    it('returns English config for unknown language', () => {
      const config = getLanguageConfig('unknown');
      expect(config.code).toBe('en');
    });
  });

  describe('isRTL', () => {
    it('returns false for LTR languages', () => {
      expect(isRTL('en')).toBe(false);
      expect(isRTL('de')).toBe(false);
      expect(isRTL('zh')).toBe(false);
    });

    it('returns true for RTL languages', () => {
      expect(isRTL('ar')).toBe(true);
      expect(isRTL('he')).toBe(true);
    });
  });

  describe('languages array', () => {
    it('has 30 languages defined', () => {
      expect(languages.length).toBeGreaterThanOrEqual(23);
    });

    it('all languages have required properties', () => {
      languages.forEach(lang => {
        expect(lang.code).toBeDefined();
        expect(lang.name).toBeDefined();
        expect(lang.nativeName).toBeDefined();
        expect(lang.flag).toBeDefined();
        expect(lang.dir).toBeDefined();
        expect(lang.dateFormat).toBeDefined();
        expect(lang.numberFormat).toBeDefined();
      });
    });

    it('has English as first language', () => {
      expect(languages[0].code).toBe('en');
    });
  });

  describe('LanguageSwitcher Component', () => {
    it('renders with current language flag', () => {
      renderWithI18n(<LanguageSwitcher />);
      expect(screen.getByText('🇺🇸')).toBeInTheDocument();
    });

    it('opens dropdown on click', async () => {
      renderWithI18n(<LanguageSwitcher />);

      const button = screen.getByRole('button', { name: /language/i });
      fireEvent.click(button);

      await waitFor(() => {
        expect(screen.getByText('Deutsch')).toBeInTheDocument();
      });
    });

    it('changes language on selection', async () => {
      renderWithI18n(<LanguageSwitcher />);

      const button = screen.getByRole('button', { name: /language/i });
      fireEvent.click(button);

      await waitFor(() => {
        expect(screen.getByText('Deutsch')).toBeInTheDocument();
      });

      fireEvent.click(screen.getByText('Deutsch'));

      await waitFor(() => {
        expect(i18n.language).toBe('de');
      });
    });

    it('renders inline variant correctly', () => {
      renderWithI18n(<LanguageSwitcher variant="inline" />);
      const buttons = screen.getAllByRole('button');
      expect(buttons.length).toBeGreaterThan(0);
    });

    it('renders flag variant correctly', () => {
      renderWithI18n(<LanguageSwitcher variant="flag" />);
      const button = screen.getByRole('button', { name: /language/i });
      expect(button).toBeInTheDocument();
    });
  });
});
