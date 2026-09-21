//! Dependency-free script/language detection, ported verbatim from the Python package's
//! `laya/lang.py`. Routing needs one decision: is this English Latin text, or something the
//! English checkpoint cannot read? Script detection is exact; the Latin-script language guess
//! is a stopword/diacritic heuristic and explicitly best-effort.

use crate::state::State;
use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use unicode_general_category::{get_general_category, GeneralCategory};

/// Unicode blocks the English (50k English BPE) checkpoint cannot read.
const SCRIPT_RANGES: &[(&str, &[(u32, u32)])] = &[
    ("greek", &[(0x0370, 0x03FF), (0x1F00, 0x1FFF)]),
    (
        "cyrillic",
        &[(0x0400, 0x052F), (0x2DE0, 0x2DFF), (0xA640, 0xA69F)],
    ),
    ("armenian", &[(0x0530, 0x058F)]),
    ("hebrew", &[(0x0590, 0x05FF)]),
    (
        "arabic",
        &[
            (0x0600, 0x06FF),
            (0x0750, 0x077F),
            (0x08A0, 0x08FF),
            (0xFB50, 0xFDFF),
            (0xFE70, 0xFEFF),
        ],
    ),
    ("devanagari", &[(0x0900, 0x097F), (0xA8E0, 0xA8FF)]),
    ("bengali", &[(0x0980, 0x09FF)]),
    ("gurmukhi", &[(0x0A00, 0x0A7F)]),
    ("gujarati", &[(0x0A80, 0x0AFF)]),
    ("oriya", &[(0x0B00, 0x0B7F)]),
    ("tamil", &[(0x0B80, 0x0BFF)]),
    ("telugu", &[(0x0C00, 0x0C7F)]),
    ("kannada", &[(0x0C80, 0x0CFF)]),
    ("malayalam", &[(0x0D00, 0x0D7F)]),
    ("sinhala", &[(0x0D80, 0x0DFF)]),
    ("thai", &[(0x0E00, 0x0E7F)]),
    ("lao", &[(0x0E80, 0x0EFF)]),
    ("tibetan", &[(0x0F00, 0x0FFF)]),
    ("myanmar", &[(0x1000, 0x109F)]),
    ("georgian", &[(0x10A0, 0x10FF)]),
    ("ethiopic", &[(0x1200, 0x137F)]),
    ("khmer", &[(0x1780, 0x17FF)]),
    (
        "hangul",
        &[(0x1100, 0x11FF), (0x3130, 0x318F), (0xAC00, 0xD7AF)],
    ),
    (
        "kana",
        &[(0x3040, 0x309F), (0x30A0, 0x30FF), (0x31F0, 0x31FF)],
    ),
    (
        "han",
        &[(0x3400, 0x4DBF), (0x4E00, 0x9FFF), (0xF900, 0xFAFF)],
    ),
];

/// Function words. Latin-script languages overlap heavily, so hits are counted per language
/// and a margin over English is required before calling something non-English.
const STOPWORDS: &[(&str, &[&str])] = &[
    (
        "en",
        &[
            "the", "and", "is", "are", "was", "were", "to", "of", "in", "for", "with", "that",
            "this", "it", "you", "have", "has", "not", "but", "on", "at", "be", "as", "from",
            "will", "can", "would", "there", "their", "what", "which", "please", "we", "i",
        ],
    ),
    (
        "fr",
        &[
            "le", "la", "les", "des", "une", "est", "pour", "dans", "que", "qui", "avec", "sur",
            "pas", "plus", "nous", "vous", "être", "cette", "mais", "sont", "ont", "aux", "ce",
        ],
    ),
    (
        "de",
        &[
            "der", "die", "das", "und", "ist", "ein", "eine", "den", "dem", "nicht", "mit", "für",
            "auf", "von", "zu", "sich", "auch", "werden", "wurde", "haben", "sind", "oder", "aber",
        ],
    ),
    (
        "es",
        &[
            "el", "los", "las", "que", "por", "con", "para", "una", "es", "se", "del", "como",
            "pero", "son", "está", "este", "esta", "todo", "más", "muy", "hay", "sus",
        ],
    ),
    (
        "pt",
        &[
            "os", "as", "que", "em", "um", "uma", "para", "com", "não", "é", "se", "do", "da",
            "dos", "das", "mas", "são", "está", "este", "esta", "muito", "pelo", "pela",
        ],
    ),
    (
        "it",
        &[
            "il", "lo", "gli", "che", "di", "per", "con", "non", "è", "si", "del", "della", "sono",
            "questo", "questa", "anche", "come", "più", "sono", "nella", "alla",
        ],
    ),
    (
        "nl",
        &[
            "het", "een", "van", "is", "op", "te", "dat", "niet", "met", "voor", "zijn", "aan",
            "door", "maar", "ook", "worden", "deze", "naar", "wordt",
        ],
    ),
];
const NON_EN_DIACRITICS: &str = "àâäãáåçéèêëíìîïñóòôöõøúùûüýÿßæœđłşţğıåäö";

/// Detection result for a state, in the Python package's shape.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Detection {
    /// Dominant script: `latin`, `han`, `devanagari`, … or `unknown` when there are no letters.
    pub script: String,
    /// Fraction of letters per detected script.
    pub script_profile: IndexMap<String, f64>,
    /// Best-effort language code for Latin-script text.
    pub language: Option<String>,
    pub is_english: bool,
    pub non_latin_fraction: f64,
}

/// Python `str.isalpha()`: the Unicode letter categories.
fn is_alpha(c: char) -> bool {
    matches!(
        get_general_category(c),
        GeneralCategory::UppercaseLetter
            | GeneralCategory::LowercaseLetter
            | GeneralCategory::TitlecaseLetter
            | GeneralCategory::ModifierLetter
            | GeneralCategory::OtherLetter
    )
}

/// Python `str.isdecimal()` (the `\d` class): decimal digits.
fn is_decimal(c: char) -> bool {
    get_general_category(c) == GeneralCategory::DecimalNumber
}

fn is_latin(cp: u32) -> bool {
    cp < 0x0250 || (0x1E00..=0x1EFF).contains(&cp)
}

fn script_of(cp: u32) -> Option<&'static str> {
    SCRIPT_RANGES
        .iter()
        .find(|(_, ranges)| ranges.iter().any(|&(lo, hi)| (lo..=hi).contains(&cp)))
        .map(|(name, _)| *name)
}

fn collect_text(value: &Value, depth: usize, out: &mut Vec<String>) {
    if depth > 6 {
        return;
    }
    match value {
        Value::String(s) => out.push(s.clone()),
        Value::Object(map) => map.values().for_each(|v| collect_text(v, depth + 1, out)),
        Value::Array(items) => items.iter().for_each(|v| collect_text(v, depth + 1, out)),
        _ => {}
    }
}

/// Flattens a state into the text used for detection: string leaves joined by spaces, keys
/// ignored (they are usually English), capped at 4000 characters.
pub fn state_text(state: &State) -> String {
    let parts = match state {
        State::Text(s) => vec![s.clone()],
        State::Json(v) => {
            let mut out = Vec::new();
            collect_text(v, 0, &mut out);
            out
        }
    };
    parts.join(" ").chars().take(4000).collect()
}

/// Letter counts per script, in first-seen order with `latin` last (as the reference builds it).
fn script_counts(text: &str) -> (IndexMap<&'static str, usize>, usize) {
    let mut counts: IndexMap<&'static str, usize> = IndexMap::new();
    let mut latin = 0;
    for c in text.chars() {
        if !is_alpha(c) {
            continue;
        }
        let cp = c as u32;
        if is_latin(cp) {
            latin += 1;
        } else if let Some(name) = script_of(cp) {
            *counts.entry(name).or_insert(0) += 1;
        }
    }
    (counts, latin)
}

/// Dominant script of `text`, or `unknown` when it has no letters.
pub fn detect_script(text: &str) -> &'static str {
    let (mut counts, latin) = script_counts(text);
    counts.insert("latin", latin);
    let total: usize = counts.values().sum();
    if total == 0 {
        return "unknown";
    }
    // Python's max() keeps the first maximum in insertion order.
    counts
        .iter()
        .fold(("unknown", 0usize), |best, (&name, &n)| {
            if n > best.1 {
                (name, n)
            } else {
                best
            }
        })
        .0
}

/// Fraction of letters belonging to each script present.
pub fn script_profile(text: &str) -> IndexMap<String, f64> {
    let (counts, latin) = script_counts(text);
    let total = latin + counts.values().sum::<usize>();
    let mut profile = IndexMap::new();
    if total == 0 {
        return profile;
    }
    if latin > 0 {
        profile.insert("latin".to_owned(), latin as f64 / total as f64);
    }
    for (name, n) in counts {
        profile.insert(name.to_owned(), n as f64 / total as f64);
    }
    profile
}

/// Best-effort language code for Latin-script text, or `None` when undecided. Requires a clear
/// margin over English function words so ordinary English is never misrouted; short inputs
/// return `None` on purpose.
pub fn guess_latin_language(text: &str) -> Option<&'static str> {
    let words: Vec<String> = text
        .split(|c: char| !(c.is_alphanumeric() && !is_decimal(c)) || c == '_')
        .filter(|w| !w.is_empty())
        .map(|w| w.to_lowercase())
        .collect();
    if words.len() < 4 {
        return None;
    }
    let score = |lang: &str| -> usize {
        let set = STOPWORDS
            .iter()
            .find(|(l, _)| *l == lang)
            .map(|(_, s)| *s)
            .unwrap_or(&[]);
        words.iter().filter(|w| set.contains(&w.as_str())).count()
    };
    let lowered = text.to_lowercase();
    let diac = lowered
        .chars()
        .filter(|c| NON_EN_DIACRITICS.contains(*c))
        .count();
    let diac_rate = diac as f64 / lowered.chars().count().max(1) as f64;
    let en = score("en");
    let (best_lang, best) = STOPWORDS
        .iter()
        .filter(|(l, _)| *l != "en")
        .map(|(l, _)| (*l, score(l)))
        // Python `max(..., default=(None, 0))`: the first language with the highest score, even 0.
        .fold((None, 0usize), |acc, (l, s)| {
            if acc.0.is_none() || s > acc.1 {
                (Some(l), s)
            } else {
                acc
            }
        });
    let english = || if en > 0 { Some("en") } else { None };
    if best == 0 && diac_rate < 0.02 {
        return english();
    }
    if let Some(lang) = best_lang {
        if best >= 2.max(en + 2) {
            return Some(lang);
        }
        if diac_rate >= 0.04 && best >= en {
            return Some(lang);
        }
    }
    english()
}

/// Full detection result for a state.
pub fn analyse(state: &State) -> Detection {
    let text = state_text(state);
    let profile = script_profile(&text);
    let script = detect_script(&text);
    let non_latin = if profile.is_empty() {
        0.0
    } else {
        round4(1.0 - profile.get("latin").copied().unwrap_or(0.0))
    };
    if script == "unknown" {
        return Detection {
            script: script.to_owned(),
            script_profile: profile,
            language: None,
            is_english: true,
            non_latin_fraction: 0.0,
        };
    }
    if script != "latin" {
        return Detection {
            script: script.to_owned(),
            script_profile: profile,
            language: None,
            is_english: false,
            non_latin_fraction: non_latin,
        };
    }
    let language = guess_latin_language(&text);
    Detection {
        script: "latin".to_owned(),
        script_profile: profile,
        is_english: matches!(language, None | Some("en")),
        language: language.map(str::to_owned),
        non_latin_fraction: non_latin,
    }
}

/// True when the English checkpoint can be expected to read this state.
pub fn is_english(state: &State) -> bool {
    analyse(state).is_english
}

fn round4(x: f64) -> f64 {
    crate::agent::round4(x)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn scripts() {
        assert_eq!(detect_script("Hello there"), "latin");
        assert_eq!(detect_script("मुझसे दो बार"), "devanagari");
        assert_eq!(detect_script("1234 !!"), "unknown");
        assert_eq!(detect_script("日本語のテキスト"), "kana");
        assert_eq!(detect_script("请退款给我们"), "han");
    }

    #[test]
    fn english_is_not_misrouted() {
        let d = analyse(&State::from(
            "The invoice is wrong and we were billed twice, please refund it.",
        ));
        assert_eq!(d.language.as_deref(), Some("en"));
        assert!(d.is_english);
        let d = analyse(&State::from(
            "Mein Konto wurde zweimal belastet und ich möchte das Geld zurück.",
        ));
        assert_eq!(d.language.as_deref(), Some("de"));
        assert!(!d.is_english);
        assert!(analyse(&State::from(json!({"n": 3}))).is_english);
    }
}
