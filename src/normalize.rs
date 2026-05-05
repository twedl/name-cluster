//! Normalization — Rust port of `scripts/_norm.py`.
//!
//! Pipeline (see ARCHITECTURE.md §Normalization for full design):
//!
//!   1. NFKD + strip combining marks
//!   2. Lowercase + non-ASCII -> space (combined with step 3)
//!   3. Punct collapse: `&` -> " and ", drop set -> "", others -> space
//!   4. Whitespace collapse
//!   5. Strip leading garbage `^0+\s+` and `^[#*]+\s+`
//!   6. Strip leading `THE` (only if >=3 tokens remain after)
//!   7. List 3 compound canonicalize (anywhere in name, longest-first)
//!   8. Iterative bidirectional List 1 strip (multi-tail / single-tail / single-head)
//!   9. List 2 canonicalize (compounds first, then singles)
//!  10. Whitespace collapse
//!
//! Output must byte-match `scripts/_norm.py:normalize()` for every input
//! in our audit corpora — that's the validation contract for this port.

use std::borrow::Cow;
use std::collections::{HashMap, HashSet};
use std::sync::LazyLock;
use unicode_normalization::char::is_combining_mark;
use unicode_normalization::UnicodeNormalization;

// ---------------------------------------------------------------------------
// Lookup tables
// ---------------------------------------------------------------------------

static LIST1_SUFFIXES_SINGLE: LazyLock<HashSet<&'static str>> = LazyLock::new(|| {
    [
        // English / Western
        "inc", "llc", "ltd", "limited", "corp", "corporation", "co", "company",
        "companies", "lp", "llp", "plc", "pty",
        // German (DE/AT/CH)
        "gmbh", "ag", "kg", "kgaa", "ohg", "gbr", "ev", "eg", "ggmbh", "mbh",
        "aktiengesellschaft", "kommanditgesellschaft", "gesellschaft",
        // French / Belgian / Swiss-French
        "sa", "sas", "sarl", "sprl", "sagl",
        // Italian
        "srl", "spa", "snc", "ss",
        // Spanish / Portuguese
        "sl", "slu", "slp", "cb", "lda", "ltda",
        // Dutch / Belgian-Dutch
        "bv", "nv", "bvba", "vof", "ua",
        // Nordic
        "oy", "ab", "as", "asa", "oyj", "aps",
        // East Asian (transliterated)
        "kk", "gk", "tmk", "pte",
        // Russian / Slavic (head-strip primary use case)
        "ooo", "oao", "ojsc", "pjsc", "cjsc", "jsc", "zao", "pao", "ao",
        // Middle East
        "fze",
        // Turkish
        "sirketi",
        // Initialisms preserved through period-drop
        "cv",
    ]
    .into_iter()
    .collect()
});

static LIST1_SUFFIXES_MULTI: LazyLock<Vec<Vec<&'static str>>> =
    LazyLock::new(|| vec![vec!["co", "ltd"]]);

/// Build a longest-first compound canonicalize table.
fn build_compound(entries: &[(&[&'static str], &'static str)]) -> Vec<(Vec<&'static str>, &'static str)> {
    let mut v: Vec<(Vec<&'static str>, &'static str)> =
        entries.iter().map(|(p, c)| (p.to_vec(), *c)).collect();
    v.sort_by(|a, b| b.0.len().cmp(&a.0.len()));
    v
}

/// Compound legal-form -> canonical short token. Applied anywhere in name,
/// longest-first to avoid partial overlaps.
static LIST3_COMPOUND_LEGAL: LazyLock<Vec<(Vec<&'static str>, &'static str)>> = LazyLock::new(|| {
    build_compound(&[
        // English
        (&["limited", "liability", "company"], "llc"),
        (&["limited", "liability"], "llc"),
        (&["limited", "liability", "partnership"], "llp"),
        (&["limited", "partnership"], "lp"),
        (&["general", "partnership"], "gp"),
        (&["joint", "stock", "company"], "jsc"),
        (&["public", "joint", "stock", "company"], "pjsc"),
        (&["open", "joint", "stock", "company"], "ojsc"),
        (&["closed", "joint", "stock", "company"], "cjsc"),
        // Russian (transliterated)
        (&["obshchestvo", "s", "ogranichennoi", "otvetstvennostyu"], "ooo"),
        (&["aktsionernoe", "obshchestvo"], "ao"),
        // Spanish / French
        (&["sociedad", "anonima"], "sa"),
        (&["sociedad", "limitada"], "sl"),
        (&["societe", "anonyme"], "sa"),
        // Japanese (transliterated)
        (&["kabushiki", "kaisha"], "kk"),
    ])
});

/// Single-token descriptor abbreviation -> canonical short form.
static LIST2_CANONICAL: LazyLock<HashMap<&'static str, &'static str>> = LazyLock::new(|| {
    [
        ("manufacturing", "mfg"),
        ("manufact", "mfg"),
        ("mfr", "mfg"),
        ("international", "intl"),
        ("import", "imp"),
        ("imports", "imp"),
        ("export", "exp"),
        ("exports", "exp"),
        ("holdings", "hldg"),
        ("holding", "hldg"),
        ("group", "grp"),
        ("enterprises", "ent"),
        ("enterprise", "ent"),
        ("industries", "ind"),
        ("industry", "ind"),
        ("services", "svc"),
        ("svcs", "svc"),
        ("solutions", "sln"),
        ("trading", "trd"),
        ("trade", "trd"),
        ("technology", "tech"),
        ("technologies", "tech"),
        ("techs", "tech"),
        ("development", "dev"),
        ("associates", "assoc"),
    ]
    .into_iter()
    .collect()
});

/// Compound descriptor canonicalize (multi-token in -> single canonical out).
static LIST2_COMPOUND: LazyLock<Vec<(Vec<&'static str>, &'static str)>> = LazyLock::new(|| {
    build_compound(&[
        (&["import", "export"], "impexp"),
        (&["imp", "exp"], "impexp"),
    ])
});

// ---------------------------------------------------------------------------
// Public entry
// ---------------------------------------------------------------------------

pub fn normalize(name: &str) -> String {
    if name.is_empty() {
        return String::new();
    }

    // ASCII fast-path: most business names are pure ASCII; skip NFKD entirely.
    let decomposed: Cow<str> = if name.is_ascii() {
        Cow::Borrowed(name)
    } else {
        Cow::Owned(name.nfkd().filter(|c| !is_combining_mark(*c)).collect())
    };

    // Char-level pass: lowercase + punct policy.
    //   '&'                 -> " and "
    //   '.' '\'' '-' '_' '+' -> drop entirely
    //   a-z / 0-9            -> keep
    //   anything else        -> ' ' (commas, parens, whitespace, non-ASCII)
    let mut s = String::with_capacity(decomposed.len() + 8);
    for c in decomposed.chars() {
        let lower = c.to_ascii_lowercase();
        match lower {
            '&' => s.push_str(" and "),
            '.' | '\'' | '-' | '_' | '+' => {}
            '0'..='9' | 'a'..='z' => s.push(lower),
            _ => s.push(' '),
        }
    }

    let collapsed = collapse_whitespace(&s);
    let stripped = strip_leading_garbage(&collapsed);
    if stripped.is_empty() {
        return String::new();
    }
    let mut tokens: Vec<String> = stripped.split_whitespace().map(String::from).collect();

    // 6. Strip leading THE if >=4 tokens before (>=3 remain after).
    if tokens.len() >= 4 && tokens[0] == "the" {
        tokens.remove(0);
    }

    // 7. List 3 compound canonicalize (anywhere, longest-first).
    tokens = apply_compound(&tokens, &LIST3_COMPOUND_LEGAL);

    // 8. Iterative bidirectional List 1 strip; preserve >=1 token.
    loop {
        let mut changed = false;
        if tokens.len() <= 1 {
            break;
        }

        // multi-word tail
        for multi in LIST1_SUFFIXES_MULTI.iter() {
            let mlen = multi.len();
            if tokens.len() > mlen
                && tokens[tokens.len() - mlen..]
                    .iter()
                    .zip(multi.iter())
                    .all(|(a, b)| a == b)
            {
                tokens.truncate(tokens.len() - mlen);
                changed = true;
                break;
            }
        }
        if changed {
            continue;
        }

        // single-word tail
        if tokens.len() > 1
            && LIST1_SUFFIXES_SINGLE.contains(tokens.last().unwrap().as_str())
        {
            tokens.pop();
            continue;
        }

        // single-word head
        if tokens.len() > 1
            && LIST1_SUFFIXES_SINGLE.contains(tokens.first().unwrap().as_str())
        {
            tokens.remove(0);
            continue;
        }

        // no further change
        break;
    }

    // 9. List 2 canonicalize (compounds first, then singles).
    tokens = apply_compound(&tokens, &LIST2_COMPOUND);
    for tok in tokens.iter_mut() {
        if let Some(canon) = LIST2_CANONICAL.get(tok.as_str()) {
            *tok = (*canon).to_string();
        }
    }

    // 10. Drop empties + join with single space.
    tokens
        .into_iter()
        .filter(|t| !t.is_empty())
        .collect::<Vec<_>>()
        .join(" ")
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

fn collapse_whitespace(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Strip `^0+\s+` or `^[#*]+\s+` if present. Returns the unchanged input
/// (borrowed) when no garbage prefix matches — the common case.
fn strip_leading_garbage(s: &str) -> &str {
    let bytes = s.as_bytes();
    let first = match bytes.first() {
        Some(b) => *b,
        None => return s,
    };
    let pred: fn(u8) -> bool = match first {
        b'0' => |b| b == b'0',
        b'#' | b'*' => |b| b == b'#' || b == b'*',
        _ => return s,
    };

    let prefix_len = bytes.iter().take_while(|&&b| pred(b)).count();
    let after_prefix = &bytes[prefix_len..];
    if after_prefix.first().is_none_or(|&b| !(b as char).is_whitespace()) {
        return s;
    }
    let ws_len = after_prefix
        .iter()
        .take_while(|&&b| (b as char).is_whitespace())
        .count();
    &s[prefix_len + ws_len..]
}

/// Apply compound-token replacements to `tokens`, longest-match-first,
/// left-to-right, anywhere in the name.
fn apply_compound(
    tokens: &[String],
    map: &[(Vec<&'static str>, &'static str)],
) -> Vec<String> {
    let mut out: Vec<String> = Vec::with_capacity(tokens.len());
    let mut i = 0;
    while i < tokens.len() {
        let mut matched = false;
        for (pat, canon) in map.iter() {
            let plen = pat.len();
            if i + plen <= tokens.len()
                && tokens[i..i + plen]
                    .iter()
                    .zip(pat.iter())
                    .all(|(a, b)| a.as_str() == *b)
            {
                out.push((*canon).to_string());
                i += plen;
                matched = true;
                break;
            }
        }
        if !matched {
            out.push(tokens[i].clone());
            i += 1;
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Tests (parity with scripts/_norm.py spot-checks)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn norm(s: &str) -> String {
        normalize(s)
    }

    #[test]
    fn basic_punct_and_suffix() {
        assert_eq!(norm("IBM Corp"), "ibm");
        assert_eq!(norm("Apple Inc."), "apple");
        assert_eq!(norm("Procter & Gamble"), "procter and gamble");
    }

    #[test]
    fn drop_set_chars() {
        assert_eq!(norm("Wal-Mart"), "walmart");
        assert_eq!(norm("McDonald's"), "mcdonalds");
        assert_eq!(norm("S.A."), "sa");
        assert_eq!(norm("S.R.L."), "srl");
        // SA is itself a List 1 entry, so a name that *only* contains a
        // legal-form would be preserved (>=1 token rule). But after period
        // drop "S.A." -> "sa" (1 token, kept).
    }

    #[test]
    fn space_set_chars() {
        assert_eq!(norm("Smith, Jones & Co"), "smith jones and");
        assert_eq!(norm("Acme (USA) Inc"), "acme usa");
        // `/` -> space (separator); resulting "import export" tokens are
        // then collapsed by List 2 compound canonicalize -> "impexp".
        assert_eq!(norm("Import/Export"), "impexp");
    }

    #[test]
    fn numeric_prefix() {
        assert_eq!(norm("00 IBM"), "ibm");
        assert_eq!(norm("000 IBM"), "ibm");
        // Preserve real-name digit prefixes:
        assert_eq!(norm("3M"), "3m");
        assert_eq!(norm("123 Textile Mfg"), "123 textile mfg");
        assert_eq!(norm("7-Eleven"), "7eleven");
    }

    #[test]
    fn the_strip_safety() {
        // 4 tokens before, 3 after -> strip
        assert_eq!(norm("The Boeing Company Limited"), "boeing");
        // 3 tokens before, 2 after -> NOT stripped
        assert_eq!(norm("The North Face"), "the north face");
    }

    #[test]
    fn russian_head_strip() {
        assert_eq!(norm("JSC ROSNEFT"), "rosneft");
        assert_eq!(norm("OOO X CORP"), "x");
        assert_eq!(norm("AKTSIONERNOE OBSHCHESTVO X"), "x");
    }

    #[test]
    fn list3_compound() {
        // BIMLOGIC LIMITED LIABILITY COMPANY -> bimlogic
        assert_eq!(norm("BIMLOGIC LIMITED LIABILITY COMPANY"), "bimlogic");
        // BIMLOGIC LLC -> bimlogic
        assert_eq!(norm("BIMLOGIC LLC"), "bimlogic");
        // OBSHCHESTVO S OGRANICHENNOI OTVETSTVENNOSTYU NPK X -> npk x
        assert_eq!(
            norm("OBSHCHESTVO S OGRANICHENNOI OTVETSTVENNOSTYU NPK X"),
            "npk x"
        );
    }

    #[test]
    fn list2_canonicalize() {
        assert_eq!(norm("Acme Manufacturing Inc"), "acme mfg");
        assert_eq!(norm("Acme Mfg Inc"), "acme mfg");
        assert_eq!(norm("Intl Acme"), "intl acme");
        assert_eq!(norm("International Acme"), "intl acme");
    }

    #[test]
    fn nfkd_accents() {
        assert_eq!(norm("Café"), "cafe");
        assert_eq!(norm("naïve"), "naive");
    }

    #[test]
    fn empty_and_short() {
        assert_eq!(norm(""), "");
        assert_eq!(norm("   "), "");
        assert_eq!(norm("?!"), "");
        assert_eq!(norm("A"), "a");
    }

    #[test]
    fn keeps_at_least_one_token() {
        // All tokens are List 1 -> should keep at least one
        assert_eq!(norm("LLC Inc Corp"), "llc");
    }
}
