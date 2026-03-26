import { useState, useRef, useEffect, memo } from 'react';
import { useTranslation } from 'react-i18next';
import { languages, getLanguageConfig, isRTL } from '../../i18n/config';
import { ChevronDown, Check } from 'lucide-react';

interface LanguageSwitcherProps {
  variant?: 'dropdown' | 'inline' | 'flag';
  showName?: boolean;
  size?: 'sm' | 'md' | 'lg';
}

export const LanguageSwitcher = memo(function LanguageSwitcher({
  variant = 'dropdown',
  showName = true,
  size = 'md',
}: LanguageSwitcherProps) {
  const { i18n, t } = useTranslation();
  const [isOpen, setIsOpen] = useState(false);
  const dropdownRef = useRef<HTMLDivElement>(null);

  const currentLanguage = getLanguageConfig(i18n.language);

  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
        setIsOpen(false);
      }
    }

    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  useEffect(() => {
    document.documentElement.dir = isRTL(i18n.language) ? 'rtl' : 'ltr';
    document.documentElement.lang = i18n.language;
  }, [i18n.language]);

  const handleLanguageChange = (langCode: string) => {
    i18n.changeLanguage(langCode);
    setIsOpen(false);
  };

  const sizeClasses = {
    sm: 'text-xs gap-1 px-2 py-1',
    md: 'text-sm gap-2 px-3 py-2',
    lg: 'text-base gap-2 px-4 py-2',
  };

  if (variant === 'flag') {
    return (
      <button
        onClick={() => setIsOpen(!isOpen)}
        className={`flex items-center justify-center rounded-lg hover:bg-gray-800 transition-colors ${sizeClasses[size]}`}
        aria-label={t('common.language')}
        aria-expanded={isOpen}
        aria-haspopup="listbox"
      >
        <span className="text-lg" aria-hidden="true">
          {currentLanguage.flag}
        </span>
        {showName && <span>{currentLanguage.nativeName}</span>}
        <ChevronDown size={14} className={`transition-transform ${isOpen ? 'rotate-180' : ''}`} />
      </button>
    );
  }

  if (variant === 'inline') {
    return (
      <div className="flex flex-wrap gap-1">
        {languages.slice(0, 6).map(lang => (
          <button
            key={lang.code}
            onClick={() => handleLanguageChange(lang.code)}
            className={`flex items-center gap-1 px-2 py-1 rounded transition-colors ${
              i18n.language === lang.code
                ? 'bg-bitcoin-orange text-white'
                : 'hover:bg-gray-800 text-gray-400'
            }`}
            aria-label={`Switch to ${lang.name}`}
            aria-pressed={i18n.language === lang.code}
          >
            <span>{lang.flag}</span>
            <span className="text-xs">{lang.code.toUpperCase()}</span>
          </button>
        ))}
      </div>
    );
  }

  return (
    <div className="relative" ref={dropdownRef}>
      <button
        onClick={() => setIsOpen(!isOpen)}
        className={`flex items-center justify-between w-full rounded-lg border border-gray-700 bg-gray-800 hover:bg-gray-750 transition-colors ${sizeClasses[size]}`}
        aria-label={t('common.language')}
        aria-expanded={isOpen}
        aria-haspopup="listbox"
      >
        <div className="flex items-center gap-2">
          <span className="text-lg" aria-hidden="true">
            {currentLanguage.flag}
          </span>
          {showName && <span className="truncate max-w-[120px]">{currentLanguage.nativeName}</span>}
        </div>
        <ChevronDown size={14} className={`transition-transform ${isOpen ? 'rotate-180' : ''}`} />
      </button>

      {isOpen && (
        <div
          className="absolute z-50 mt-1 w-64 max-h-96 overflow-y-auto bg-gray-800 border border-gray-700 rounded-lg shadow-xl"
          role="listbox"
          aria-label={t('common.language')}
        >
          <div className="p-2">
            <div className="text-xs text-gray-500 px-2 py-1 uppercase tracking-wider">
              {t('common.language')}
            </div>
            {languages.map(lang => (
              <button
                key={lang.code}
                onClick={() => handleLanguageChange(lang.code)}
                className={`w-full flex items-center gap-3 px-2 py-2 rounded-md transition-colors ${
                  i18n.language === lang.code
                    ? 'bg-bitcoin-orange/20 text-bitcoin-orange'
                    : 'hover:bg-gray-700 text-gray-200'
                }`}
                role="option"
                aria-selected={i18n.language === lang.code}
              >
                <span className="text-lg" aria-hidden="true">
                  {lang.flag}
                </span>
                <div className="flex-1 text-left">
                  <div className="text-sm font-medium">{lang.nativeName}</div>
                  <div className="text-xs text-gray-500">{lang.name}</div>
                </div>
                {i18n.language === lang.code && <Check size={16} className="text-bitcoin-orange" />}
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  );
});

export default LanguageSwitcher;
