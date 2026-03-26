import { useState, useRef, useEffect } from 'react';
import { useI18n, languages } from '../../stores/i18nStore';
import { ChevronDown, Globe, Check } from 'lucide-react';

export function LanguageSelector() {
  const { language, setLanguage } = useI18n();
  const [isOpen, setIsOpen] = useState(false);
  const dropdownRef = useRef<HTMLDivElement>(null);

  const currentLang = languages.find(l => l.code === language) || languages[0];

  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
        setIsOpen(false);
      }
    }
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  return (
    <div className="relative" ref={dropdownRef}>
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="flex items-center gap-2 px-3 py-2 bg-background hover:bg-background/80 border border-border rounded-lg text-sm transition-colors"
      >
        <Globe size={16} className="text-text-secondary" />
        <span className="hidden sm:inline">{currentLang.flag} {currentLang.code.toUpperCase()}</span>
        <ChevronDown size={14} className={`transition-transform ${isOpen ? 'rotate-180' : ''}`} />
      </button>

      {isOpen && (
        <div className="absolute right-0 mt-2 w-64 bg-card border border-border rounded-lg shadow-xl z-50 max-h-80 overflow-y-auto">
          <div className="p-2">
            <p className="px-3 py-1 text-xs text-text-muted uppercase">Sprache / Language</p>
            {languages.map((lang) => (
              <button
                key={lang.code}
                onClick={() => {
                  setLanguage(lang.code);
                  setIsOpen(false);
                }}
                className={`w-full flex items-center justify-between px-3 py-2 rounded-md text-sm transition-colors ${
                  language === lang.code 
                    ? 'bg-bitcoin-orange/10 text-bitcoin-orange' 
                    : 'text-text-secondary hover:bg-background hover:text-text-primary'
                }`}
              >
                <div className="flex items-center gap-3">
                  <span className="text-lg">{lang.flag}</span>
                  <div className="text-left">
                    <p className="font-medium">{lang.nativeName}</p>
                    <p className="text-xs text-text-muted">{lang.name}</p>
                  </div>
                </div>
                {language === lang.code && <Check size={16} />}
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
