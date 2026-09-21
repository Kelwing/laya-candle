use serde::{Deserialize, Serialize};
use std::fmt;
use std::str::FromStr;

/// The bundle repo that holds every checkpoint; only the requested subfolder is downloaded.
pub const BUNDLE_REPO: &str = "convaiinnovations/laya";

/// One of the three published weight sets.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Checkpoint {
    English,
    Multilingual,
    TypedDecisions,
}

impl Checkpoint {
    pub const ALL: [Checkpoint; 3] = [
        Checkpoint::English,
        Checkpoint::Multilingual,
        Checkpoint::TypedDecisions,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Checkpoint::English => "english",
            Checkpoint::Multilingual => "multilingual",
            Checkpoint::TypedDecisions => "typed-decisions",
        }
    }

    /// Subfolder inside [`BUNDLE_REPO`]; `None` for the repo root.
    pub fn subfolder(self) -> Option<&'static str> {
        match self {
            Checkpoint::English => None,
            Checkpoint::Multilingual => Some("multilingual"),
            Checkpoint::TypedDecisions => Some("typed-decisions"),
        }
    }

    /// `repo/subfolder` as the Python router reports it.
    pub fn repo(self) -> String {
        match self.subfolder() {
            Some(sub) => format!("{BUNDLE_REPO}/{sub}"),
            None => BUNDLE_REPO.to_owned(),
        }
    }

    /// Accepts the canonical names and the Python package's aliases (case-insensitive).
    pub fn parse(name: &str) -> Option<Self> {
        Some(match name.trim().to_ascii_lowercase().as_str() {
            "english" | "en" | "laya" | "default" => Checkpoint::English,
            "multilingual" | "multi" | "ml" | "laya-multilingual" => Checkpoint::Multilingual,
            "typed-decisions"
            | "typed"
            | "typed_decisions"
            | "laya-typed-decisions"
            | "decisions" => Checkpoint::TypedDecisions,
            _ => return None,
        })
    }
}

impl fmt::Display for Checkpoint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

impl FromStr for Checkpoint {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Checkpoint::parse(s).ok_or_else(|| {
            format!(
                "unknown checkpoint {s:?}; choose one of english, multilingual, typed-decisions"
            )
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aliases() {
        assert_eq!(Checkpoint::parse(" EN "), Some(Checkpoint::English));
        assert_eq!(Checkpoint::parse("ml"), Some(Checkpoint::Multilingual));
        assert_eq!(
            Checkpoint::parse("typed_decisions"),
            Some(Checkpoint::TypedDecisions)
        );
        assert_eq!(Checkpoint::parse("nope"), None);
        assert_eq!(
            Checkpoint::TypedDecisions.repo(),
            "convaiinnovations/laya/typed-decisions"
        );
        assert_eq!(
            serde_json::to_string(&Checkpoint::TypedDecisions).unwrap(),
            "\"typed-decisions\""
        );
    }
}
