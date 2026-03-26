"""
Risk Profiling Questionnaire
=============================
25 scientifically-grounded questions inspired by FinaMetrica and FINRA guidelines.

The questionnaire covers five psychometric dimensions (5 questions each):
    - tolerance  (Q1-5)  : Emotional reaction to losses, overconfidence, risk comfort
    - capacity   (Q6-10) : Financial cushion, income stability, liabilities, liquidity needs
    - knowledge  (Q11-15): Risk/return, diversification, volatility, crypto risks, leverage
    - horizon    (Q16-20): Investment timeline, liquidity needs, plan commitment
    - bias       (Q21-25): Loss aversion, herding, recency bias, overtrading tendency

Options score 1-4 where 4 = most risk-tolerant.
Speeding threshold: answers given in < 5 seconds are flagged as invalid.

References
----------
Nunnally, J.C. & Bernstein, I.H. (1994). Psychometric Theory (3rd ed.). McGraw-Hill.
FinaMetrica Pty Ltd. (2013). FinaMetrica Risk Profiling Technical Manual.
FINRA Investor Education Foundation. (2020). Financial Capability Study.
"""

from dataclasses import dataclass, field
from typing import Dict, List

SUPPORTED_LANGUAGES = ["de", "en"]


@dataclass
class QuestionOption:
    """
    A single answer option for a question.

    Attributes
    ----------
    text  : Translations keyed by ISO-639-1 language code.
    score : Psychometric score for this option (1 = least risk-tolerant, 4 = most).
    """
    text: Dict[str, str]
    score: int


@dataclass
class Question:
    """
    A single questionnaire item.

    Attributes
    ----------
    id        : Unique identifier 1-25.
    dimension : Psychometric dimension this question measures.
    text      : Question text in supported languages.
    options   : Exactly 4 answer options ordered a-d.
    weight    : Psychometric weight used in raw score computation. Default 1.0.
    """
    id: int
    dimension: str
    text: Dict[str, str]
    options: List[QuestionOption]
    weight: float = 1.0


# ---------------------------------------------------------------------------
# DIMENSION: tolerance (Q1-Q5)
# Measures emotional reaction to portfolio losses, comfort with uncertainty,
# and resistance to panic selling.
# ---------------------------------------------------------------------------

_Q1 = Question(
    id=1,
    dimension="tolerance",
    text={
        "de": "Ihr Investitionsportfolio verliert innerhalb eines Monats 20% seines Wertes. Wie reagieren Sie?",
        "en": "Your investment portfolio loses 20% of its value within one month. How do you react?",
    },
    options=[
        QuestionOption(
            text={
                "de": "Ich verkaufe sofort alles, um weitere Verluste zu vermeiden.",
                "en": "I sell everything immediately to avoid further losses.",
            },
            score=1,
        ),
        QuestionOption(
            text={
                "de": "Ich verkaufe einen Teil und wechsle in sicherere Anlagen.",
                "en": "I sell part of it and move into safer assets.",
            },
            score=2,
        ),
        QuestionOption(
            text={
                "de": "Ich halte meine Positionen und warte auf eine Erholung.",
                "en": "I hold my positions and wait for a recovery.",
            },
            score=3,
        ),
        QuestionOption(
            text={
                "de": "Ich kaufe nach, da dies eine günstige Kaufgelegenheit ist.",
                "en": "I buy more, as this represents a good buying opportunity.",
            },
            score=4,
        ),
    ],
    weight=1.2,
)

_Q2 = Question(
    id=2,
    dimension="tolerance",
    text={
        "de": "Wie würden Sie Ihre allgemeine Einstellung gegenüber Risiko bei Finanzanlagen beschreiben?",
        "en": "How would you describe your general attitude towards risk in financial investments?",
    },
    options=[
        QuestionOption(
            text={
                "de": "Ich vermeide Risiken grundsätzlich und akzeptiere dafür niedrige Renditen.",
                "en": "I avoid risk entirely and accept low returns in exchange.",
            },
            score=1,
        ),
        QuestionOption(
            text={
                "de": "Ich bevorzuge geringe Risiken und gelegentlich moderate Renditen.",
                "en": "I prefer low risk and occasionally moderate returns.",
            },
            score=2,
        ),
        QuestionOption(
            text={
                "de": "Ich akzeptiere moderate Schwankungen für bessere langfristige Renditen.",
                "en": "I accept moderate swings for better long-term returns.",
            },
            score=3,
        ),
        QuestionOption(
            text={
                "de": "Ich bin bereit, erhebliche kurzfristige Verluste für maximale Renditen hinzunehmen.",
                "en": "I am willing to accept substantial short-term losses for maximum returns.",
            },
            score=4,
        ),
    ],
    weight=1.0,
)

_Q3 = Question(
    id=3,
    dimension="tolerance",
    text={
        "de": "Stellen Sie sich vor, Sie müssen zwischen zwei Investments wählen. Welches bevorzugen Sie?",
        "en": "Imagine you must choose between two investments. Which do you prefer?",
    },
    options=[
        QuestionOption(
            text={
                "de": "Sicheres Investment: garantierte Rendite von 3% p.a.",
                "en": "Safe investment: guaranteed return of 3% p.a.",
            },
            score=1,
        ),
        QuestionOption(
            text={
                "de": "Wahrscheinlich 7% Rendite, aber möglicher Verlust von 5%.",
                "en": "Likely 7% return, but possible loss of 5%.",
            },
            score=2,
        ),
        QuestionOption(
            text={
                "de": "Wahrscheinlich 15% Rendite, aber möglicher Verlust von 20%.",
                "en": "Likely 15% return, but possible loss of 20%.",
            },
            score=3,
        ),
        QuestionOption(
            text={
                "de": "Wahrscheinlich 40% Rendite, aber möglicher Verlust von 40%.",
                "en": "Likely 40% return, but possible loss of 40%.",
            },
            score=4,
        ),
    ],
    weight=1.2,
)

_Q4 = Question(
    id=4,
    dimension="tolerance",
    text={
        "de": "Wie lange könnten Sie einen Buchverlust von 30% in Ihrem Portfolio aushalten, bevor Sie schlaflosen Nächten hätten?",
        "en": "How long could you endure a paper loss of 30% in your portfolio before losing sleep?",
    },
    options=[
        QuestionOption(
            text={
                "de": "Ich könnte das überhaupt nicht ertragen – schon wenige Tage wären zu viel.",
                "en": "I could not stand it at all — even a few days would be too much.",
            },
            score=1,
        ),
        QuestionOption(
            text={
                "de": "Einige Wochen, danach müsste ich handeln.",
                "en": "A few weeks, after which I would need to act.",
            },
            score=2,
        ),
        QuestionOption(
            text={
                "de": "Mehrere Monate, wenn ich an die langfristige Strategie glaube.",
                "en": "Several months, if I believe in the long-term strategy.",
            },
            score=3,
        ),
        QuestionOption(
            text={
                "de": "So lange wie nötig – Buchverluste sind Teil des Spiels.",
                "en": "As long as necessary — paper losses are part of the game.",
            },
            score=4,
        ),
    ],
    weight=1.1,
)

_Q5 = Question(
    id=5,
    dimension="tolerance",
    text={
        "de": "Ein Freund erzählt Ihnen, sein spekulatives Investment hat sich in drei Monaten verdoppelt. Was denken Sie?",
        "en": "A friend tells you their speculative investment doubled in three months. What do you think?",
    },
    options=[
        QuestionOption(
            text={
                "de": "Das war reines Glück und zu riskant – so etwas würde ich nie machen.",
                "en": "That was pure luck and too risky — I would never do that.",
            },
            score=1,
        ),
        QuestionOption(
            text={
                "de": "Ich bin beeindruckt, aber würde nur einen sehr kleinen Betrag riskieren.",
                "en": "I am impressed but would only risk a very small amount.",
            },
            score=2,
        ),
        QuestionOption(
            text={
                "de": "Ich würde recherchieren und einen moderaten Betrag investieren.",
                "en": "I would research it and invest a moderate amount.",
            },
            score=3,
        ),
        QuestionOption(
            text={
                "de": "Ich würde sofort einsteigen – hohe Chancen rechtfertigen das Risiko.",
                "en": "I would jump in immediately — high upside justifies the risk.",
            },
            score=4,
        ),
    ],
    weight=0.9,
)

# ---------------------------------------------------------------------------
# DIMENSION: capacity (Q6-Q10)
# Measures objective financial ability to absorb losses without lifestyle impact.
# ---------------------------------------------------------------------------

_Q6 = Question(
    id=6,
    dimension="capacity",
    text={
        "de": "Wie lange könnten Sie Ihren aktuellen Lebensstandard aufrechterhalten, wenn Sie Ihr gesamtes Einkommen verlören?",
        "en": "How long could you maintain your current lifestyle if you lost all your income?",
    },
    options=[
        QuestionOption(
            text={
                "de": "Weniger als 3 Monate – ich habe kaum Rücklagen.",
                "en": "Less than 3 months — I have almost no savings buffer.",
            },
            score=1,
        ),
        QuestionOption(
            text={
                "de": "3 bis 6 Monate.",
                "en": "3 to 6 months.",
            },
            score=2,
        ),
        QuestionOption(
            text={
                "de": "6 bis 24 Monate.",
                "en": "6 to 24 months.",
            },
            score=3,
        ),
        QuestionOption(
            text={
                "de": "Mehr als 2 Jahre – ich habe ein solides Notfallpolster.",
                "en": "More than 2 years — I have a solid emergency cushion.",
            },
            score=4,
        ),
    ],
    weight=1.1,
)

_Q7 = Question(
    id=7,
    dimension="capacity",
    text={
        "de": "Wie stabil schätzen Sie Ihr Einkommen in den nächsten 3 Jahren ein?",
        "en": "How stable do you consider your income to be over the next 3 years?",
    },
    options=[
        QuestionOption(
            text={
                "de": "Sehr unsicher – mein Job oder Geschäft ist stark gefährdet.",
                "en": "Very uncertain — my job or business is significantly at risk.",
            },
            score=1,
        ),
        QuestionOption(
            text={
                "de": "Etwas unsicher – es gibt reale Risiken für mein Einkommen.",
                "en": "Somewhat uncertain — there are real risks to my income.",
            },
            score=2,
        ),
        QuestionOption(
            text={
                "de": "Weitgehend stabil mit gelegentlichen Schwankungen.",
                "en": "Largely stable with occasional fluctuations.",
            },
            score=3,
        ),
        QuestionOption(
            text={
                "de": "Sehr stabil – mehrere Einkommensquellen oder unkündbarer Vertrag.",
                "en": "Very stable — multiple income sources or permanent contract.",
            },
            score=4,
        ),
    ],
    weight=1.1,
)

_Q8 = Question(
    id=8,
    dimension="capacity",
    text={
        "de": "Welchen Anteil Ihres monatlichen Einkommens müssen Sie für feste Verbindlichkeiten aufwenden (Miete, Kredite, Unterhalt)?",
        "en": "What share of your monthly income goes to fixed liabilities (rent, loans, maintenance)?",
    },
    options=[
        QuestionOption(
            text={
                "de": "Mehr als 70% – ich habe kaum Spielraum.",
                "en": "More than 70% — I have very little room.",
            },
            score=1,
        ),
        QuestionOption(
            text={
                "de": "50% bis 70%.",
                "en": "50% to 70%.",
            },
            score=2,
        ),
        QuestionOption(
            text={
                "de": "30% bis 50%.",
                "en": "30% to 50%.",
            },
            score=3,
        ),
        QuestionOption(
            text={
                "de": "Weniger als 30% – ich habe erheblichen finanziellen Spielraum.",
                "en": "Less than 30% — I have considerable financial headroom.",
            },
            score=4,
        ),
    ],
    weight=1.0,
)

_Q9 = Question(
    id=9,
    dimension="capacity",
    text={
        "de": "Welchen Anteil Ihres Gesamtvermögens würden Sie in volatile Anlagen wie Kryptowährungen investieren?",
        "en": "What portion of your total assets would you invest in volatile assets like cryptocurrencies?",
    },
    options=[
        QuestionOption(
            text={
                "de": "Nichts oder unter 2% – ich kann es mir nicht leisten zu verlieren.",
                "en": "Nothing or under 2% — I cannot afford to lose it.",
            },
            score=1,
        ),
        QuestionOption(
            text={
                "de": "2% bis 10%.",
                "en": "2% to 10%.",
            },
            score=2,
        ),
        QuestionOption(
            text={
                "de": "10% bis 30%.",
                "en": "10% to 30%.",
            },
            score=3,
        ),
        QuestionOption(
            text={
                "de": "Mehr als 30% – ich habe genug anderes Vermögen als Sicherheitsnetz.",
                "en": "More than 30% — I have enough other assets as a safety net.",
            },
            score=4,
        ),
    ],
    weight=1.2,
)

_Q10 = Question(
    id=10,
    dimension="capacity",
    text={
        "de": "Haben Sie in den nächsten 12 Monaten größere geplante Ausgaben, die Ihr investiertes Kapital beeinflussen könnten?",
        "en": "Do you have major planned expenses in the next 12 months that could affect your invested capital?",
    },
    options=[
        QuestionOption(
            text={
                "de": "Ja, ich werde wahrscheinlich einen erheblichen Teil meines Kapitals benötigen.",
                "en": "Yes, I will likely need a significant portion of my capital.",
            },
            score=1,
        ),
        QuestionOption(
            text={
                "de": "Möglicherweise – einige mittlere Ausgaben sind geplant.",
                "en": "Possibly — some moderate expenses are planned.",
            },
            score=2,
        ),
        QuestionOption(
            text={
                "de": "Kleine Ausgaben geplant, aber mein Investment bleibt weitgehend unberührt.",
                "en": "Small expenses planned but my investment remains largely untouched.",
            },
            score=3,
        ),
        QuestionOption(
            text={
                "de": "Nein, das investierte Kapital wird in diesem Zeitraum nicht benötigt.",
                "en": "No, the invested capital will not be needed during this period.",
            },
            score=4,
        ),
    ],
    weight=1.0,
)

# ---------------------------------------------------------------------------
# DIMENSION: knowledge (Q11-Q15)
# Measures understanding of financial concepts relevant to crypto trading.
# ---------------------------------------------------------------------------

_Q11 = Question(
    id=11,
    dimension="knowledge",
    text={
        "de": "Was beschreibt die Beziehung zwischen Risiko und Rendite am treffendsten?",
        "en": "Which statement best describes the relationship between risk and return?",
    },
    options=[
        QuestionOption(
            text={
                "de": "Höhere Renditen sind immer ohne höheres Risiko erreichbar.",
                "en": "Higher returns are always achievable without taking more risk.",
            },
            score=1,
        ),
        QuestionOption(
            text={
                "de": "Risiko und Rendite sind unabhängig voneinander.",
                "en": "Risk and return are independent of each other.",
            },
            score=2,
        ),
        QuestionOption(
            text={
                "de": "Im Allgemeinen erfordern höhere Renditeerwartungen höhere Risiken.",
                "en": "Generally, higher return expectations require taking higher risks.",
            },
            score=3,
        ),
        QuestionOption(
            text={
                "de": "Höhere Risiken garantieren immer höhere Renditen.",
                "en": "Higher risks always guarantee higher returns.",
            },
            score=4,
        ),
    ],
    weight=1.0,
)

_Q12 = Question(
    id=12,
    dimension="knowledge",
    text={
        "de": "Was ist Diversifikation und welchen Effekt hat sie auf das Portfoliorisiko?",
        "en": "What is diversification and what effect does it have on portfolio risk?",
    },
    options=[
        QuestionOption(
            text={
                "de": "Diversifikation bedeutet, alles in ein einziges, sehr sicheres Asset zu investieren.",
                "en": "Diversification means investing everything in a single, very safe asset.",
            },
            score=1,
        ),
        QuestionOption(
            text={
                "de": "Diversifikation verteilt das Kapital auf verschiedene Assets, kann aber das Gesamtrisiko erhöhen.",
                "en": "Diversification spreads capital across assets but can increase total risk.",
            },
            score=2,
        ),
        QuestionOption(
            text={
                "de": "Diversifikation kann unkorrelierte Risiken reduzieren, eliminiert jedoch das Marktrisiko nicht.",
                "en": "Diversification can reduce uncorrelated risks but does not eliminate market risk.",
            },
            score=3,
        ),
        QuestionOption(
            text={
                "de": "Diversifikation eliminiert alle Risiken vollständig.",
                "en": "Diversification completely eliminates all risks.",
            },
            score=4,
        ),
    ],
    weight=1.0,
)

_Q13 = Question(
    id=13,
    dimension="knowledge",
    text={
        "de": "Was beschreibt Volatilität im Kontext von Finanzmärkten am besten?",
        "en": "Which statement best describes volatility in the context of financial markets?",
    },
    options=[
        QuestionOption(
            text={
                "de": "Volatilität ist der garantierte Verlust eines Investments.",
                "en": "Volatility is the guaranteed loss of an investment.",
            },
            score=1,
        ),
        QuestionOption(
            text={
                "de": "Volatilität beschreibt das durchschnittliche Renditeniveau eines Assets.",
                "en": "Volatility describes the average return level of an asset.",
            },
            score=2,
        ),
        QuestionOption(
            text={
                "de": "Volatilität misst die Schwankungsbreite der Preisänderungen über einen Zeitraum.",
                "en": "Volatility measures the magnitude of price fluctuations over a period.",
            },
            score=3,
        ),
        QuestionOption(
            text={
                "de": "Hohe Volatilität bedeutet immer hohe zukünftige Renditen.",
                "en": "High volatility always means high future returns.",
            },
            score=4,
        ),
    ],
    weight=1.0,
)

_Q14 = Question(
    id=14,
    dimension="knowledge",
    text={
        "de": "Welches der folgenden Risiken ist spezifisch für Kryptowährungen und bei traditionellen Aktien weniger ausgeprägt?",
        "en": "Which of the following risks is specific to cryptocurrencies and less pronounced in traditional stocks?",
    },
    options=[
        QuestionOption(
            text={
                "de": "Inflationsrisiko.",
                "en": "Inflation risk.",
            },
            score=1,
        ),
        QuestionOption(
            text={
                "de": "Dividendenkürzungsrisiko.",
                "en": "Dividend cut risk.",
            },
            score=2,
        ),
        QuestionOption(
            text={
                "de": "Regulatorisches Verbotsrisiko und Exchange-Hack-Risiko.",
                "en": "Regulatory ban risk and exchange hack risk.",
            },
            score=3,
        ),
        QuestionOption(
            text={
                "de": "Zinsänderungsrisiko.",
                "en": "Interest rate change risk.",
            },
            score=4,
        ),
    ],
    weight=1.1,
)

_Q15 = Question(
    id=15,
    dimension="knowledge",
    text={
        "de": "Was passiert bei einem gehebelten Investment (5x Leverage), wenn der Marktpreis um 20% fällt?",
        "en": "What happens in a leveraged investment (5x leverage) when the market price falls 20%?",
    },
    options=[
        QuestionOption(
            text={
                "de": "Ich verliere genau 20% meines eingesetzten Kapitals.",
                "en": "I lose exactly 20% of my invested capital.",
            },
            score=1,
        ),
        QuestionOption(
            text={
                "de": "Der Verlust wird durch den Hebel abgepuffert und ist kleiner als 20%.",
                "en": "The loss is buffered by the leverage and is less than 20%.",
            },
            score=2,
        ),
        QuestionOption(
            text={
                "de": "Ich verliere 100% meines eingesetzten Kapitals (Totalverlust / Liquidation).",
                "en": "I lose 100% of my invested capital (total loss / liquidation).",
            },
            score=3,
        ),
        QuestionOption(
            text={
                "de": "Der Verlust beträgt genau 100%, nicht mehr.",
                "en": "The loss is exactly 100%, no more.",
            },
            score=4,
        ),
    ],
    weight=1.2,
)

# ---------------------------------------------------------------------------
# DIMENSION: horizon (Q16-Q20)
# Measures investment time horizon and commitment to long-term plans.
# ---------------------------------------------------------------------------

_Q16 = Question(
    id=16,
    dimension="horizon",
    text={
        "de": "Über welchen Zeitraum planen Sie primär mit Ihrem investierten Kapital?",
        "en": "Over what time period are you primarily planning with your invested capital?",
    },
    options=[
        QuestionOption(
            text={
                "de": "Weniger als 1 Jahr – kurzfristige Ziele.",
                "en": "Less than 1 year — short-term goals.",
            },
            score=1,
        ),
        QuestionOption(
            text={
                "de": "1 bis 3 Jahre.",
                "en": "1 to 3 years.",
            },
            score=2,
        ),
        QuestionOption(
            text={
                "de": "3 bis 10 Jahre.",
                "en": "3 to 10 years.",
            },
            score=3,
        ),
        QuestionOption(
            text={
                "de": "Mehr als 10 Jahre – langfristiger Vermögensaufbau.",
                "en": "More than 10 years — long-term wealth accumulation.",
            },
            score=4,
        ),
    ],
    weight=1.1,
)

_Q17 = Question(
    id=17,
    dimension="horizon",
    text={
        "de": "Wie weit sind Sie vom Rentenalter oder dem Zeitpunkt entfernt, an dem Sie auf dieses Kapital angewiesen sein werden?",
        "en": "How far are you from retirement or the point when you will depend on this capital?",
    },
    options=[
        QuestionOption(
            text={
                "de": "Ich bin bereits im Ruhestand oder weniger als 3 Jahre davon entfernt.",
                "en": "I am already retired or less than 3 years away.",
            },
            score=1,
        ),
        QuestionOption(
            text={
                "de": "3 bis 10 Jahre.",
                "en": "3 to 10 years.",
            },
            score=2,
        ),
        QuestionOption(
            text={
                "de": "10 bis 20 Jahre.",
                "en": "10 to 20 years.",
            },
            score=3,
        ),
        QuestionOption(
            text={
                "de": "Mehr als 20 Jahre – ich bin jung und habe viel Zeit.",
                "en": "More than 20 years — I am young and have plenty of time.",
            },
            score=4,
        ),
    ],
    weight=1.2,
)

_Q18 = Question(
    id=18,
    dimension="horizon",
    text={
        "de": "Wenn Ihr Portfolio 3 Jahre lang keine positiven Renditen erzielt, was tun Sie?",
        "en": "If your portfolio generates no positive returns for 3 years, what do you do?",
    },
    options=[
        QuestionOption(
            text={
                "de": "Ich steige sofort aus – 3 Jahre Wartezeit ist inakzeptabel.",
                "en": "I exit immediately — 3 years of waiting is unacceptable.",
            },
            score=1,
        ),
        QuestionOption(
            text={
                "de": "Ich überprüfe die Strategie grundlegend und wechsle wahrscheinlich.",
                "en": "I fundamentally review the strategy and likely switch.",
            },
            score=2,
        ),
        QuestionOption(
            text={
                "de": "Ich bleibe dabei, passe aber die Allokation moderat an.",
                "en": "I stay in but moderately adjust the allocation.",
            },
            score=3,
        ),
        QuestionOption(
            text={
                "de": "Ich bleibe vollständig dabei – 3 Jahre sind kurz in einer langfristigen Strategie.",
                "en": "I stay fully committed — 3 years is short in a long-term strategy.",
            },
            score=4,
        ),
    ],
    weight=1.0,
)

_Q19 = Question(
    id=19,
    dimension="horizon",
    text={
        "de": "Benötigen Sie in den nächsten 2 Jahren schnellen Zugang zu Ihrem gesamten investierten Kapital?",
        "en": "Do you need quick access to your entire invested capital within the next 2 years?",
    },
    options=[
        QuestionOption(
            text={
                "de": "Ja, sehr wahrscheinlich – ich könnte das gesamte Kapital kurzfristig benötigen.",
                "en": "Yes, very likely — I may need all the capital at short notice.",
            },
            score=1,
        ),
        QuestionOption(
            text={
                "de": "Möglicherweise – einen Teil davon.",
                "en": "Possibly — a portion of it.",
            },
            score=2,
        ),
        QuestionOption(
            text={
                "de": "Unwahrscheinlich – nur im Notfall.",
                "en": "Unlikely — only in an emergency.",
            },
            score=3,
        ),
        QuestionOption(
            text={
                "de": "Nein, das Kapital ist langfristig gebunden.",
                "en": "No, the capital is locked in for the long term.",
            },
            score=4,
        ),
    ],
    weight=1.1,
)

_Q20 = Question(
    id=20,
    dimension="horizon",
    text={
        "de": "Wenn Ihre gewählte Anlagestrategie 6 Monate lang deutlich schlechter abschneidet als der breite Markt, wie reagieren Sie?",
        "en": "If your chosen investment strategy significantly underperforms the broader market for 6 months, how do you react?",
    },
    options=[
        QuestionOption(
            text={
                "de": "Ich wechsle sofort zur besseren Marktstrategie.",
                "en": "I immediately switch to the better-performing market strategy.",
            },
            score=1,
        ),
        QuestionOption(
            text={
                "de": "Ich analysiere die Ursachen und wechsle wahrscheinlich.",
                "en": "I analyse the causes and likely switch.",
            },
            score=2,
        ),
        QuestionOption(
            text={
                "de": "Ich analysiere, bleibe aber bei der Strategie, wenn die Grundthese stimmt.",
                "en": "I analyse but stay with the strategy if the core thesis holds.",
            },
            score=3,
        ),
        QuestionOption(
            text={
                "de": "6 Monate sind irrelevant – ich folge der Strategie unverändert.",
                "en": "6 months is irrelevant — I follow the strategy unchanged.",
            },
            score=4,
        ),
    ],
    weight=0.9,
)

# ---------------------------------------------------------------------------
# DIMENSION: bias (Q21-Q25)
# Measures behavioral biases that distort rational financial decision-making.
# Score is INVERTED: score=4 means the respondent is least biased (most rational).
# ---------------------------------------------------------------------------

_Q21 = Question(
    id=21,
    dimension="bias",
    text={
        "de": "Welche der folgenden Aussagen beschreibt eher Ihr Gefühl beim Verlieren gegenüber dem Gewinnen?",
        "en": "Which of the following best describes how you feel about losing versus winning?",
    },
    options=[
        QuestionOption(
            text={
                "de": "Verluste schmerzen mich viel stärker als Gewinne in gleicher Höhe mich freuen.",
                "en": "Losses hurt me much more than equivalent gains please me.",
            },
            score=1,
        ),
        QuestionOption(
            text={
                "de": "Verluste schmerzen etwas mehr als Gewinne mich freuen.",
                "en": "Losses hurt somewhat more than gains please me.",
            },
            score=2,
        ),
        QuestionOption(
            text={
                "de": "Verluste und Gewinne beeinflussen mich etwa gleich stark.",
                "en": "Losses and gains affect me roughly equally.",
            },
            score=3,
        ),
        QuestionOption(
            text={
                "de": "Ich treffe meine Entscheidungen rein rational – Emotionen spielen keine Rolle.",
                "en": "I make decisions purely rationally — emotions play no role.",
            },
            score=4,
        ),
    ],
    weight=1.1,
)

_Q22 = Question(
    id=22,
    dimension="bias",
    text={
        "de": "Wenn alle Ihre Freunde und Medien euphorisch über ein bestimmtes Investment berichten, was tun Sie?",
        "en": "When all your friends and the media are euphoric about a specific investment, what do you do?",
    },
    options=[
        QuestionOption(
            text={
                "de": "Ich steige sofort ein – wenn alle kaufen, muss es gut sein.",
                "en": "I jump in immediately — if everyone is buying, it must be good.",
            },
            score=1,
        ),
        QuestionOption(
            text={
                "de": "Ich kaufe einen kleinen Anteil, um dabei zu sein.",
                "en": "I buy a small stake so I am not left out.",
            },
            score=2,
        ),
        QuestionOption(
            text={
                "de": "Ich recherchiere selbstständig vor einer Entscheidung.",
                "en": "I research independently before making any decision.",
            },
            score=3,
        ),
        QuestionOption(
            text={
                "de": "Euphorie der Masse ist für mich eher ein Warnsignal als ein Kaufgrund.",
                "en": "Mass euphoria is for me a warning signal rather than a buying reason.",
            },
            score=4,
        ),
    ],
    weight=1.0,
)

_Q23 = Question(
    id=23,
    dimension="bias",
    text={
        "de": "Nach einem sehr starken Kursanstieg in den letzten 3 Monaten, was erwarten Sie für die nächsten 3 Monate?",
        "en": "After a very strong price increase in the last 3 months, what do you expect for the next 3 months?",
    },
    options=[
        QuestionOption(
            text={
                "de": "Den Trend fortsetzt sich – ich kaufe nach.",
                "en": "The trend continues — I buy more.",
            },
            score=1,
        ),
        QuestionOption(
            text={
                "de": "Wahrscheinlich weitere Gewinne, obwohl eine Korrektur möglich ist.",
                "en": "Probably more gains, although a correction is possible.",
            },
            score=2,
        ),
        QuestionOption(
            text={
                "de": "Ich kann es nicht wissen – vergangene Performance ist kein Indikator für die Zukunft.",
                "en": "I cannot know — past performance is not an indicator of future results.",
            },
            score=3,
        ),
        QuestionOption(
            text={
                "de": "Nach starken Anstiegen besteht erhöhtes Rückschlagsrisiko – ich bin vorsichtig.",
                "en": "After strong rises, there is elevated pullback risk — I am cautious.",
            },
            score=4,
        ),
    ],
    weight=1.0,
)

_Q24 = Question(
    id=24,
    dimension="bias",
    text={
        "de": "Wie oft überprüfen und handeln Sie typischerweise Ihr Portfolio?",
        "en": "How often do you typically review and trade your portfolio?",
    },
    options=[
        QuestionOption(
            text={
                "de": "Mehrmals täglich – ich verfolge jeden Kurs und handle häufig.",
                "en": "Multiple times a day — I track every price and trade frequently.",
            },
            score=1,
        ),
        QuestionOption(
            text={
                "de": "Täglich bis wöchentlich mit gelegentlichen impulsiven Trades.",
                "en": "Daily to weekly with occasional impulsive trades.",
            },
            score=2,
        ),
        QuestionOption(
            text={
                "de": "Wöchentlich bis monatlich mit disziplinierten Einstiegspunkten.",
                "en": "Weekly to monthly with disciplined entry points.",
            },
            score=3,
        ),
        QuestionOption(
            text={
                "de": "Selten – ich halte an meiner Strategie fest und handle nur bei klaren Signalen.",
                "en": "Rarely — I stick to my strategy and only act on clear signals.",
            },
            score=4,
        ),
    ],
    weight=1.1,
)

_Q25 = Question(
    id=25,
    dimension="bias",
    text={
        "de": "Sie haben eine Position, die 40% im Minus ist. Ihre ursprüngliche Analyse ist noch gültig. Was tun Sie?",
        "en": "You have a position that is 40% underwater. Your original analysis is still valid. What do you do?",
    },
    options=[
        QuestionOption(
            text={
                "de": "Ich halte und kaufe nach, weil ich den Einstiegspreis zurückgewinnen will.",
                "en": "I hold and buy more because I want to recover my entry price.",
            },
            score=1,
        ),
        QuestionOption(
            text={
                "de": "Ich halte, aber kaufe nicht nach – ich hoffe auf Erholung.",
                "en": "I hold but do not add — I hope for recovery.",
            },
            score=2,
        ),
        QuestionOption(
            text={
                "de": "Ich evaluiere die Position neu und entscheide basierend auf aktuellen Daten.",
                "en": "I re-evaluate the position and decide based on current data.",
            },
            score=3,
        ),
        QuestionOption(
            text={
                "de": "Ich bewerte rational: Wenn die These stimmt, bleibe ich; falls nicht, verkaufe ich ohne Emotionen.",
                "en": "I assess rationally: if the thesis holds I stay; if not, I sell without emotion.",
            },
            score=4,
        ),
    ],
    weight=1.2,
)


# ---------------------------------------------------------------------------
# Master question list — authoritative order used by scoring engine
# ---------------------------------------------------------------------------

QUESTIONS: List[Question] = [
    _Q1, _Q2, _Q3, _Q4, _Q5,       # tolerance
    _Q6, _Q7, _Q8, _Q9, _Q10,      # capacity
    _Q11, _Q12, _Q13, _Q14, _Q15,  # knowledge
    _Q16, _Q17, _Q18, _Q19, _Q20,  # horizon
    _Q21, _Q22, _Q23, _Q24, _Q25,  # bias
]

# Convenience lookup by question id
QUESTIONS_BY_ID: Dict[int, Question] = {q.id: q for q in QUESTIONS}
