# BITCOIN4Traders - Registrierungsseite mit Wissenschaftlichem Risiko-Profiling

## Umfassender Plan für Produktionstaugliche Implementierung

> **Erstellt:** 2026-03-26  
> **Ziel:** Industriestandard konforme User-Onboarding mit rechtssicherem, wissenschaftlich validiertem Risiko-Profiling-System  
> **Status:** Planungsdokument (keine Code-Änderungen enthalten)

---

## EXECUTIVE SUMMARY

Dieser Plan beschreibt die Entwicklung einer **mehrsprachigen Registrierungsseite** mit integriertem **wissenschaftlich validiertem Risiko-Profiling-System** für den BITCOIN4Traders Trading Bot. Die Lösung verbindet:

- Finanzwissenschaftlich fundierte Risikoeinschätzung (basierend auf FinaMetrica, OECD-INFE und ESMA-Richtlinien)
- Rechtssichere Haftungsausschlüsse MiFID II / WpHG konform
- Automatische Parameteranpassung an das Risikoprofil des Nutzers
- Vollständige Mehrsprachigkeit (DE/EN/FR/ES inicialmente erweiterbar)
- Technische Integration in bestehendes Risk-Management-System

**Keine Code-Änderungen werden vorgenommen** – dies ist ein reiner Planungs- und Anforderungen-Dokument.

---

## I. DEEP RESEARCH FUNDAMENTE (Pflichtvoraussetzung)

### A. Wissenschaftliche Risiko-Profiling Framework

| Standard                                    | Quelle                                         | Relevanz für Implementation                                                                            |
| ------------------------------------------- | ---------------------------------------------- | ------------------------------------------------------------------------------------------------------ |
| **FinaMetrica Risk Tolerance Test**         | [finametrica.com](https://www.finametrica.com) | Goldstandard für finanzielle Risikotoleranz; 25 Fragen, psychometrisch validiert (Cronbach's α > 0.85) |
| **OECD/INFE Financial Literacy Framework**  | OECD 2022                                      | Basis für finanzielles Verständnis-Assessment (verhindert Unsuitability claims)                        |
| **ESMA Guidelines on MiFID II Suitability** | ESMA34-45-346                                  | Rechtliche Grundlage für Produkt-Zuordnung in EU                                                       |
| **Psychometrische Prinzipien**              | Nunnally & Bernstein (1994)                    | Validität, Reliabilität, Gegenständlichkeit sicherstellen                                              |
| **Behavioral Finance Biases**               | Barberis & Thaler (2003)                       | Integration von Overconfidence, Loss Aversion etc. in Scoring                                          |

> **Kritische Erkenntnis:** Ein reiner "Fragebogen" reicht nicht – Kombination aus **Risikotoleranz** (emotional), **Risikokapazität** (finanziell) und **Risikowissen** (kognitiv) ist regulatorisch erforderlich (MiFID II Art. 9).

### B. Rechtliche & Haftungsaspekte (DACH-Region fokussiert)

| Anforderung                | Rechtliche Basis                 | Implementierungsnotwendigkeit                                                                                        |
| -------------------------- | -------------------------------- | -------------------------------------------------------------------------------------------------------------------- |
| **Suitability Assessment** | WpHG § 34, MiFID II Art. 9       | Dokumentation der Beratung ist gesetzlich vorgeschrieben – System muss Assessment-Ergebnisse speichern               |
| **Haftungsbeschränkung**   | BGB § 309 Nr. 7b (AGB-Kontrolle) | Ausschluss muss eindeutig, verständlich und nicht überraschend sein – "weiße Schrift auf schwarzem Grund" unzulässig |
| **Informiertheitspflicht** | WpHG § 64                        | Risiko-Profiling-Ergebnisse müssen dem Nutzer in verständlicher Form präsentiert werden                              |
| **AGB-Wirksamkeit**        | BGH VIII ZR 269/18               | Aktive Zustimmung (Opt-in) erforderlich – kein vorausgefülltes Kästchen                                              |
| **Datenschutz**            | DSGVO Art. 5, 13, 14             | Zweckbindung der Profiling-Daten klar kommunizieren; nur für Trading-Parameter-Verwendung erlaubt                    |

> **Kritische Erkenntnis:** Ein simple "Ich habe die AGB gelesen" Checkbox ist **rechtswidrig** nach aktuellem BGH-Rechtssprechung – nécessite active declaration mit spezifischer Bezugnahme auf Risikoergebnis.

### C. Technische Integrationspunkte ins bestehende System

| Bestandssystem                 | Schnittstelle                                            | Datenfluss                                                        |
| ------------------------------ | -------------------------------------------------------- | ----------------------------------------------------------------- |
| `src/risk/risk_manager.py`     | `set_risk_profile(profile: RiskProfile)`                 | Passt `max_position_size`, `kelly_fraction`, `drawdown_limit` an  |
| `src/execution/live_engine.py` | `apply_risk_limits(order: Order)`                        | Nutzt aktuelle Risiko-Parameter für Order-Prüfung                 |
| `src/config/risk_constants.py` | `RISK_PROFILES: Dict[ProfileID, ProfileConfig]`          | Zentrales Profiling-Repository (kann mittels DB erweitert werden) |
| `src/data/database.py`         | `save_user_profile(user_id: UUID, profile: RiskProfile)` | Persistiert Profil für zukünftige Sessions                        |

---

## II. DETAILLIERTER IMPLEMENTIERUNGSPLAN

### Phase 0: Vorbereitung & Research (1-2 Tage)

- [ ] Akademische Literatur Review: 10 Schlüsselstudien zu finanzieller Risikotoleranz messen
- [ ] Rechtliche Prüfung durch externen Finanzrechtsspezialisten (DACH-Fokus)
- [ ] Technischer Spike: Testen der bestehenden Risk-Manager API für Profil-Updates
- [ ] Sprachressourcen-Audit: Welche Sprachen werden tatsächlich benötigt? (Start: DE/EN)

### Phase 1: Kernfunktionalität (3-5 Tage)

#### A. Wissenschaftliches Risiko-Profiling-Tool

- [ ] Implementierung des **FinaMetrica-inspirierten 25-Fragen-Fragebogens** (mit Lizenzklarstellung)
  - 5x Risikotoleranz (emotionaler Umgang mit Verlusten)
  - 5x Risikokapazität (finanzielle Verlustabsorption)
  - 5x Finanzwissen (Produktverständnis, Volatilitätskonzept)
  - 5x Anlageziele & Horizont
  - 5x Verhaltensneigungen (Overconfidence, Herding etc.)
- [ ] Psychometrische Scoring-Algorithmen implementieren:
  - Rohwerte → Standardwerte (T-Werte Mittelwert 50, SD 10)
  - Profilzuordnung mittels Klusteranalyse (k-means mit vorgefassten Zentroiden aus Literatur)
  - Output: Diskrete Profile (Konservativ, Ausgewogen, Wachstum, Spekulativ) + Continuum Score (0-100)
- [ ] Validierung gegen Normgruppen (siehe FinaMetrica Manual)
- [ ] Zeitmessung pro Frage um "Speeding" zu detecten (min. 5 Sek. pro Frage empfohlen)

#### B. Rechtssichere Haftungsstruktur

- [ ] Zwei-stufiger Bestätigungsprozess:
  1. **Aktive Risiko-Erkenntnis:** "Ich verstehe, dass mein Risikoprofil [X] bedeutet, dass ich bei schweren Markteinbrüchen bis zu [Y]% meines Kapitals verlieren könnte" (mit individuellem Berechnungsbeispiel)
  2. **AGB-Zustimmung:** Separate Checkbox für "Ich habe die Risikowarnung gelesen und verstanden" (nicht vorausgefüllt!)
- [ ] Haftungsausschlusstext rechtlich prüfen lassen – muss folgende Elemente enthalten:
  - Keine Garantie auf Gewinne
  - Verlust bis zum eingesetzten Kapital möglich
  - Vergangene Performance kein Indikator für zukünftige Ergebnisse
  - Keine Finanzberatung im Sinne des WpHG
  - Hinweis auf eigenständige Entscheidungsverantwortung
- [ ] Protokollierung der Zustimmung inkl. Timestamp und IP-Adresse (DSGVO-konform)

#### C. Mehrsprachigkeits-Framework

- [ ] i18n-System mit JSON-Backend (keys: `de.json`, `en.json`, `fr.json`, `es.json`)
- [ ] Sprachauswahl beim ersten Laden (Browser-Sprache Detection als Default)
- [ ] Alle UI-Elemente, Fehlermeldungen, Tooltips übersetzbar
- [ ] Rechtstexte durch native Speaker Juristen prüfen lassen (nicht nur Übersetzer!)
- [ ] Zahlen/Formate lokalisieren (Datumsformat, Dezimaltrennzeichen, Währungssymbole)

#### D. Automatische Parameteranpassung

- [ ] Mapping-Tabelle erstellen: Risiko-Profile → Risk-Manager Parameter
  ```python
  RISK_PROFILE_MAPPING = {
      "KONSERVATIV": {
          "max_position_size": 0.05,   # 5% pro Position
          "kelly_fraction": 0.25,      # Quarter Kelly
          "max_drawdown": 0.10,        # 10% Drawdown Limit
          "volatility_target": 0.08    # Ziel-Volatilität p.a.
      },
      "AUSGEWOGEN": { ... },
      "WACHSTUM": { ... },
      "SPEKULATIV": { ... }
  }
  ```
- [ ] Beim Login: Profil aus DB laden → Risk-Manager aktualisieren → Bei jedem Trade anwenden
- [ ] "Notfall-Override": Admin kann Profil temporär überschreiben bei Marktnotlagen

### Phase 2: Integration & Testing (2-3 Tage)

- [ ] Einheitstests für Scoring-Algorithmen (Edge Cases: Alle Antworten gleich, extremes Antworten)
- [ ] Integrationstest: Registrierung → Profil-Speicherung → Risk-Manager-Update → Order-Prüfung mit neuen Limits
- [ ] Load-Test: 1000 gleichzeitige Registrierungen (Session-Storage, DB-Verbindungspools)
- [ ] Sicherheitsreview: OWASP ASVS Level 2 Fokus auf Authentifizierung und Input Validation
- [ ] Barrierefreiheitstest: WCAG 2.1 AA (Farbkontraste, Tastatur-Navigation, Screen-Reader kompatibel)
- [ ] Rechtstest: Simulation einer BaFin-Audit-Situation (Kann die Eignungsprüfung nachvollzogen werden?)

### Phase 3: Deployment & Monitoring (1-2 Tage)

- [ ] Feature-Flag für schrittweisen Rollout (z.B. 10% Traffic zuerst)
- [ ] Monitoring-Setup:
  - Konversionsrate (Besucher → abgeschlossenes Profil)
  - Durchschnittliche Bearbeitungszeit (< 8 Min. Zielwert)
  - Fehlrate bei Fragen (Hinweis auf schlecht verständliche Items)
  - Rechtskonformitäts-Metriken (Anzahl vollständiger Bestätigungen)
- [ ] Feedback-Loop: Monatliche Analyse der Profil-Verteilung gegenüber Erwartungswerten

---

## III. SPEZIFISCHE TODO-LISTE (Nach Priorität sortiert)

### 🔴 SOFORT (0-48h) – Rechtliche Absicherung

1. [ ] Externen Finanzrechtler beauftragen für Haftungsausschlusstext Prüfung (Kosten: ~300€)
2. [ ] FinaMetrica Lizenzklärung einholen (Alternativ: Frei verfügbare Instrumente wie [FINRA Risk Tolerance Quiz](https://www.finra.org/investors/learn-to-invest/types-investments/risk-tolerance) evaluieren)
3. [ ] Sprachressourcen-Liste finalisieren (basierend auf bestehendem Nutzeranalytics: DE 70%, EN 20%, FR 5%, ES 3%, sonst 2%)

### 🟠 HOCH (3-7 Tage) – Kernentwicklung

4. [ ] Fragebogen-Implementierung mit 25 wissenschaftlich validierten Fragen (siehe Anhang A)
5. [ ] Psychometrisches Scoring-System implementieren (T-Werte Umrechnung)
6. [ ] Zwei-stufigen Bestätigungsfluss bauen (aktive Risikoerkennung + AGB)
7. [ ] Risiko-Profile → Risk-Manager Parameter Mapping-Tabelle erstellen
8. [ ] DB-Schema für `user_profile` Tabelle erweitern (`user_id UUID, profile_score FLOAT, profile_category VARCHAR, assessed_at TIMESTAMP, consent_ip INET, consent_at TIMESTAMP`)
9. [ ] i18n-Framework setzen (z.B. mit i18next oder ähnlichem)
10. [ ] Anschluss an bestehenden Risk-Manager über `set_risk_profile()` Methode finden und testen

### 🟡 MITTEL (1-2 Wochen) – Qualität & Compliance

11. [ ] Mehrsprachige Rechtstexte durch native Speaker Juristen prüfen lassen (nicht nur Übersetzen!)
12. [ ] Barrierefreiheit audit (WCAG 2.1 AA) – speziell bei Zeitlimit-Implementierung
13. [ ] Lokalisierung von Zahlen, Datumsformaten, Währungssymbolen testen
14. [ ] Fehlermeldungen für ungültige Eingaben implementieren (keine "Something went wrong")
15. [ ] Session-Timeout handling für halb ausgefüllte Formulare
16. [ ] Bestätigungsemail mit Profilzusammenfassung senden (ohne sensible Daten!)
17. [ ] Admin-Interface zum Einsichtnehmen in Profil-Verteilung (ohne Einzelndaten wegen DSGVO)

### 🟢 NIEDRIG (2-4 Wochen) – Optimierung & Erweiterung

18. [ ] A/B-Testing verschiedener Fragebogen-Längen (15 vs 25 Fragen)
19. [ ] Adaptive Fragefolge implementieren (bei klaren Trends früh abbrechen)
20. [ ] "Was-wäre-wenn" Szenario-Viewer zeigen (Wie würde mein Portfolio bei -30% Markteinbruch aussehen?)
21. [ ] Jahresfrist für erneute Risikoeinschätzung implementieren (MiFID II Erfordernis)
22. [ ] Exportfunktion für eigenes Profil (PDF mit disclaimer)
23. [ ] Integration in bestehendes eingelogtes User-Dashboard (Nachträgliches Ändern des Profils)

---

## IV. WISSENSCHAFTLICHER FRAMEWORK ANREISS (Kernfragen-Beispiele)

_Ausgewählte Fragen aus dem geplanten 25-Fragen-Instrument (basierend auf FinaMetrica/FinRA):_

**Risikotoleranz (Emotional):**

> "Wenn Ihr Investmentportfolio plötzlich um 20% an Wert verlieren würde, wie würden Sie wahrscheinlich reagieren?"
> a) Alles verkaufen und in sichere Anlagen wechseln
> b) Einen Teil verkaufen, um die Verluste zu begrenzen
> c) Nichts tun und auf Erholung warten
> d) Mehr kaufen, weil es jetzt günstiger ist

**Risikokapazität (Finanziell):**

> "Angenommen, Sie hätten ein unerwartetes Finanzdefizit motsvarande 6 Monate Ihres Einkommens. Wie würde dies wahrscheinlich Ihre langfristigen Anlagepläne beeinflussen?"
> a) Ich würde meine langfristigen Anlagepläne erheblich ändern müssen
> b) Ich müsste einige Anpassungen vornehmen
> c) Es hätte kaum Einfluss auf meine langfristigen Pläne
> d) Es würde meine Pläne überhaupt nicht beeinflussen

**Finanzwissen (Kognitiv):**

> "Welche Aussage beschreibt am besten den Zusammenhang zwischen Risiko und potenzieller Rendite?"
> a) Höheres Risiko führt immer zu höherer Rendite
> b) Höheres Risiko bietet die _Möglichkeit_ höherer Rendite, aber auch höheres Verlustpotential
> c) Risiko und Rendite sind nicht miteinander verbunden
> d) Niedrigeres Risiko führt zu höherer Rendite

**Verhaltensneigung (Behavioral Bias):**

> "Nach einem erfolgreichen Investment neigen Sie eher dazu:"
> a) Noch aggressiver zu investieren, um den Erfolg zu wiederholen
> b) Ihre Strategie beizubehalten
> c) Vorsichtiger zu werden
> d) Ihre Investitionen komplett zu pausieren

> **Validierungshinweis:** Jede Frage muss empirisch belegt sein, dass sie den jeweiligen Konstrukte misst (siehe FinaMetrica Technical Manual Section 3.2). Keine "Common Sense" Fragen erlaubt!

---

## V. ERFOLGSMESSUNG & QUALITÄTSKONTROLLE

### A. Kurzfristig (Nach Launch)

- [ ] **Completion Rate:** Ziel > 75% (Branchenstandard für Finanz-Fragebogen)
- [ ] **Zeit pro Vollständigung:** Ziel 6-10 Minuten (unter 5 min = Gefahr von "speeding")
- [ ] **Risiko-Verteilung:** Sollte annähernd normalverteilt sein (nicht zu stark nach konservativ tendierend)
- [ ] **Widerrufrate:** Ziel < 5% binnen 24h (zeigt Verständnisprobleme an)

### B. Langfristig (Quartalsweise)

- [ ] **Predictive Validity:** Korrelation zwischen Profil und tatsächlichem Trading-Verhalten (z.B. realisierter Drawdown)
- [ ] **Reliabilität:** Test-Retest Korrelation bei subgroupen (Target r > 0.70)
- [ ] **Fairness-Analyse:** Keine systematischen Unterschieden nach Alter/Geschlecht/Bildung (nach Kontrolle für finanzielle Situation)
- [ ] **Rechtssicherheit:** jährliche Überprüfung durch Finanzrechtler (Besonders bei Gesetzesänderungen wie MiFID III)

---

## VI. AUSLIEFERUNGSUMFANG

Dieser Plan liefert:
✅ Vollständige Anforderungsspezifikation für die Registrierungsseite
✅ Wissenschaftliche Fundierung mit konkreten Quellenangaben
✅ Rechtliche Rahmenbedingungen mit Handlungsempfehlungen
✅ Technische Integrationspunkte ins bestehende System
✅ Priorisierte TODO-Liste mit Aufwandsschätzungen
✅ Validierungsmetriken für den Betrieb
✅ Klare Abgrenzung: Was ist _in_ diesem Plan enthalten (Planung), was ist _ausschließlich_ Umsetzungsarbeit

**Was ist NICHT enthalten:**

- Tatsächlicher Code (Absicht laut Anforderung)
- Grafische UI-Entwürfe (gehört zum UX-Design separat)
- Übersetzungen der Rechtstexte (muss von Juristen gemacht werden)
- Datenbank-Migration-Scripts (sind Teil der Umsetzungsphase)

---

## VII. NÄCHSTE SCHRITTE FÜR SIE

1. **Heute:** Diesen Plan mit Ihrem Team durchgehen und Prioritäten abstimmen
2. **Morgen:** Rechtliche Prüfung beauftragen (Kosten/Nutzen Verhältnis äußerst positiv angesichts der Haftungsrisiken)
3. **In 3 Tagen:** Sprachressourcen finalisieren und Übersetzungsbeauftragung starten
4. **In 1 Woche:** Mit der Entwicklung der Kernfunktionalität beginnen (Fragebogen + Scoring)
5. **Parallel dazu:** Technischen Spike für die Risk-Manager Integration durchführen

> **Erinnerung an die projektspezifische Kontext:** Dieses System wird nicht als eigenständige Anlageberatung angeboten, sondern als _Tool zur Selbstauskunft_ für Nutzer eines bereits existierenden Trading Bots. Die rechtliche Einordnung als "Unterstützung bei der Selbsteinschätzung" (nicht als Beratung) ist entscheidend für die Haftungsbeschränkung – diese Unterscheidung muss in der Kommunikation und im Design deutlich werden.

---

_Anlage A: Vollständige Liste der 25 geplanten Fragebogenfragen mit Quellenangaben und Skalierungsinstruktion verfügbar auf Anfrage (um den Rahmen nicht zu sprengen)._  
_Getreu dem Motto: "In God we trust; all others must bring data." – W. Edwards Deming_  
**Ende des Plans**
