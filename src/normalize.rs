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
//!   8. Iterative bidirectional List 1 strip (multi-tail / single-tail /
//!      single-head; head-strip is restricted to LIST1_HEAD_STRIPPABLE)
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
        "inc",
        "incorporated",
        "llc",
        "ltd",
        "limited",
        "corp",
        "corporation",
        "co",
        "company",
        "companies",
        "lp",
        "llp",
        "plc",
        "pty",
        // German (DE/AT/CH)
        "gmbh",
        "ag",
        "kg",
        "kgaa",
        "ohg",
        "gbr",
        "ev",
        "eg",
        "ggmbh",
        "mbh",
        "ug",
        "aktiengesellschaft",
        "kommanditgesellschaft",
        "gesellschaft",
        "unternehmergesellschaft",
        // French / Belgian / Swiss-French
        "sa",
        "sas",
        "sarl",
        "sprl",
        "sagl",
        // Italian
        "srl",
        "spa",
        "snc",
        "ss",
        // Spanish / Portuguese
        "sl",
        "slu",
        "slp",
        "cb",
        "lda",
        "ltda",
        // Dutch / Belgian-Dutch
        "bv",
        "nv",
        "bvba",
        "vof",
        "ua",
        // Nordic
        "oy",
        "ab",
        "as",
        "asa",
        "oyj",
        "aps",
        "aktiebolag",
        // Estonian
        "ou",
        // East Asian (transliterated)
        "kk",
        "gk",
        "tmk",
        "pte",
        // Russian / Slavic (head-strip primary use case)
        "ooo",
        "oao",
        "ojsc",
        "pjsc",
        "cjsc",
        "jsc",
        "zao",
        "pao",
        "ao",
        // Czech / Slovak (post-period-drop: s.r.o. -> sro)
        "sro",
        // Polish (post-List-3 compound canonicalization)
        "spzoo",
        "psa",
        "ska",
        "spk",
        "spj",
        // Hungarian (short forms; long forms canonicalize via List 3 first)
        "kft",
        "bt",
        "kkt",
        "rt",
        "zrt",
        "nyrt",
        "reszvenytarsasag",
        // Italian/Portuguese long forms (in addition to "srl"/"lda")
        "limitata",
        "limitada",
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

/// Legal forms that may be stripped from the HEAD of a name.
///
/// Derived from `LIST1_SUFFIXES_SINGLE` by excluding the ambiguous short
/// codes. Rationale is ambiguity, not nationality: `llc`, `ooo`, `jsc`,
/// `gmbh` are unmistakable legal forms in any position, and prefix-form names
/// are routine (`LLC RUSSKOYE VREMYA`, `JSC ROSNEFT`). A token of ≤2 chars in
/// head position is far more likely to be the company's own initials —
/// stripping it destroys the sole distinguishing token, so `AB International`
/// collapses to `intl` and collides with every other `<2-letter>
/// International` in the corpus.
///
/// `ao` is the deliberate exception: Russian Aktsionernoe Obshchestvo
/// genuinely leads (`AO GAZPROM`), and List 3 canonicalizes its long form to
/// `ao` before this step runs.
static LIST1_HEAD_STRIPPABLE: LazyLock<HashSet<&'static str>> = LazyLock::new(|| {
    LIST1_SUFFIXES_SINGLE
        .iter()
        .copied()
        .filter(|t| t.len() > 2 || *t == "ao")
        .collect()
});

/// Build a longest-first compound canonicalize table.
fn build_compound(
    entries: &[(&[&'static str], &'static str)],
) -> Vec<(Vec<&'static str>, &'static str)> {
    let mut v: Vec<(Vec<&'static str>, &'static str)> =
        entries.iter().map(|(p, c)| (p.to_vec(), *c)).collect();
    v.sort_by(|a, b| b.0.len().cmp(&a.0.len()));
    v
}

/// Compound legal-form -> canonical short token. Applied anywhere in name,
/// longest-first to avoid partial overlaps.
static LIST3_COMPOUND_LEGAL: LazyLock<Vec<(Vec<&'static str>, &'static str)>> =
    LazyLock::new(|| {
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
            // Russian (transliterated). Multiple Latin schemes: BGN/PCGN uses `'iu`
            // (apostrophe-dropped to `iu`), others use `yu`. Soft-sign Ь after Т in
            // отвественностью becomes either, so we accept both.
            (
                &["obshchestvo", "s", "ogranichennoi", "otvetstvennostyu"],
                "ooo",
            ),
            (
                &["obshchestvo", "s", "ogranichennoi", "otvetstvennostiu"],
                "ooo",
            ),
            (&["aktsionernoe", "obshchestvo"], "ao"),
            (&["publichnoe", "aktsionernoe", "obshchestvo"], "pjsc"),
            (&["zakrytoe", "aktsionernoe", "obshchestvo"], "cjsc"),
            (&["otkrytoe", "aktsionernoe", "obshchestvo"], "ojsc"),
            // Polish (post-atomic-Latin map: ł->l, NFKD strips accents)
            // spółka z ograniczoną odpowiedzialnością = LLC = "sp. z o.o."
            (
                &["spolka", "z", "ograniczona", "odpowiedzialnoscia"],
                "spzoo",
            ),
            // Abbreviated forms after period-drop:
            //   "Sp. z o.o."    -> "sp z oo"
            //   "Sp Z O O"      -> "sp z o o"
            (&["sp", "z", "oo"], "spzoo"),
            (&["sp", "z", "o", "o"], "spzoo"),
            (&["spolka", "akcyjna"], "sa"),
            (&["prosta", "spolka", "akcyjna"], "psa"),
            // Komandytowo-akcyjna: hyphen-drop concatenates, or source is spaced.
            (&["spolka", "komandytowoakcyjna"], "ska"),
            (&["spolka", "komandytowo", "akcyjna"], "ska"),
            (&["spolka", "komandytowa"], "spk"),
            (&["spolka", "jawna"], "spj"),
            (&["spolka", "partnerska"], "spp"),
            (&["spolka", "cywilna"], "sc"),
            (&["spolka", "europejska"], "se"),
            // Spanish / French
            (&["sociedad", "anonima"], "sa"),
            (&["sociedad", "limitada"], "sl"),
            (&["societe", "anonyme"], "sa"),
            // French SARL appearing as "S.A R.L." (period-drop: spaces leak between)
            (&["sa", "rl"], "sarl"),
            // Japanese (transliterated)
            (&["kabushiki", "kaisha"], "kk"),
            // German UG entrepreneurial co: "UG (haftungsbeschränkt)" or full form
            (&["ug", "haftungsbeschrankt"], "ug"),
            (&["unternehmergesellschaft", "haftungsbeschrankt"], "ug"),
            // Hungarian: long forms collapse to short legal-form codes (then List 1
            // strips the short code in the next pass). All entries are post-NFKD
            // (diacritics stripped: ő->o, ű->u, é->e, etc.).
            (&["korlatolt", "felelossegu", "tarsasag"], "kft"),
            (&["zartkoruen", "mukodo", "reszvenytarsasag"], "zrt"),
            (&["nyilvanosan", "mukodo", "reszvenytarsasag"], "nyrt"),
            (&["beteti", "tarsasag"], "bt"),
            (&["kozkereseti", "tarsasag"], "kkt"),
            // Chinese (Pinyin transliteration of the most-common legal forms).
            // 有限公司      = "Limited Company"      -> ltd
            // 有限责任公司   = "Limited Liability Co" -> ltd
            // 股份有限公司   = "Joint-Stock Limited"  -> jsc
            (&["you", "xian", "gong", "si"], "ltd"),
            (&["you", "xian", "ze", "ren", "gong", "si"], "ltd"),
            (&["gu", "fen", "you", "xian", "gong", "si"], "jsc"),
        ])
    });

/// Single-token descriptor abbreviation -> canonical short form.
/// Empty-string targets are stripped by the final empty-token filter (used
/// for pure connectives like "and" — see comment on the entry below).
static LIST2_CANONICAL: LazyLock<HashMap<&'static str, &'static str>> = LazyLock::new(|| {
    [
        ("manufacturing", "mfg"),
        ("manufact", "mfg"),
        ("manuf", "mfg"),
        ("manufac", "mfg"),
        ("mfr", "mfg"),
        ("mftg", "mfg"),
        ("international", "intl"),
        ("int", "intl"),
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
        ("service", "svc"),
        ("serv", "svc"),
        ("ser", "svc"),
        ("srv", "svc"),
        ("srvc", "svc"),
        ("svcs", "svc"),
        ("solutions", "sln"),
        ("trading", "trd"),
        ("trade", "trd"),
        ("technology", "tech"),
        ("technologies", "tech"),
        ("techs", "tech"),
        ("development", "dev"),
        ("associates", "assoc"),
        ("associate", "assoc"),
        ("management", "mgmt"),
        ("mgt", "mgmt"),
        ("information", "info"),
        ("department", "dept"),
        // Pure connective. Both `&` (expanded to " and " upstream) and the
        // literal word "and" canonicalize away to the same shape, so
        // "Smith & Jones", "Smith and Jones", and "Smith Jones" collide.
        // Edge case: a name reducing to JUST "and" after legal-form strip
        // (rare brand "And Company") becomes empty -> cluster_id=null.
        ("and", ""),
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
    //   '&'                  -> " and "
    //   '.' '\'' '-' '_' '+' -> drop entirely
    //   a-z / 0-9            -> keep
    //   non-decomposable Latin-extension letters: explicit ASCII fallback
    //     (NFKD doesn't decompose these — they're "atomic" precomposed forms,
    //      so without an explicit map the catchall replaces them with space
    //      and splits Polish/German/Danish/French/Icelandic words mid-token)
    //   anything else        -> ' ' (commas, parens, whitespace, non-ASCII)
    let mut s = String::with_capacity(decomposed.len() + 8);
    for c in decomposed.chars() {
        let lower = c.to_ascii_lowercase();
        match lower {
            '&' => s.push_str(" and "),
            '.' | '\'' | '-' | '_' | '+' => {}
            '0'..='9' | 'a'..='z' => s.push(lower),
            'Ł' | 'ł' => s.push('l'),             // Polish, Wendish
            'Ø' | 'ø' => s.push('o'),             // Danish, Norwegian, Faroese
            'Æ' | 'æ' => s.push_str("ae"),        // Old English, Nordic, Icelandic
            'Œ' | 'œ' => s.push_str("oe"),        // French
            'ß' => s.push_str("ss"),              // German eszett
            'Þ' | 'þ' => s.push_str("th"),        // Icelandic, Old English thorn
            'Ð' | 'ð' | 'Đ' | 'đ' => s.push('d'), // Icelandic eth, Croatian d-stroke
            'ı' => s.push('i'),                   // Turkish dotless i
            'İ' => s.push('i'),                   // Turkish dotted capital I
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
        if tokens.len() > 1 && LIST1_SUFFIXES_SINGLE.contains(tokens.last().unwrap().as_str()) {
            tokens.pop();
            continue;
        }

        // single-word head (Slavic forms only — see LIST1_HEAD_STRIPPABLE)
        if tokens.len() > 1 && LIST1_HEAD_STRIPPABLE.contains(tokens.first().unwrap().as_str()) {
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
    if after_prefix
        .first()
        .is_none_or(|&b| !(b as char).is_whitespace())
    {
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
fn apply_compound(tokens: &[String], map: &[(Vec<&'static str>, &'static str)]) -> Vec<String> {
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
        assert_eq!(norm("Apple Incorporated"), "apple");
        assert_eq!(norm("Procter & Gamble"), "procter gamble");
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
        // "&" expands to " and " then "and" is stripped via List 2, so the
        // trailing "and Co" collapses to nothing after "co" is stripped too.
        assert_eq!(norm("Smith, Jones & Co"), "smith jones");
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
    fn unambiguous_legal_forms_still_head_strip() {
        // >2 chars: not plausibly a company's initials, so prefix-form names
        // (routine in Russian/Slavic usage) must still collapse onto the
        // suffix-form spelling of the same entity.
        assert_eq!(norm("LLC RUSSKOYE VREMYA"), norm("RUSSKOYE VREMYA LLC"));
        assert_eq!(norm("LLC VEB CAPITAL"), norm("VEB CAPITAL"));
        assert_eq!(
            norm("LIMITED LIABILITY COMPANY SBERBANK CAPITAL"),
            norm("SBERBANK CAPITAL LLC")
        );
        assert_eq!(norm("GMBH ACME"), "acme");
    }

    #[test]
    fn western_legal_forms_are_tail_only() {
        // Tail position: genuine legal form, strip it.
        assert_eq!(norm("Volvo AB"), "volvo");
        assert_eq!(norm("Philips NV"), "philips");
        assert_eq!(norm("Roche AG"), "roche");
        // Head position: almost certainly the company's initials. Keeping the
        // token is what stops every "<2-letter> International" in a corpus
        // from collapsing onto the bare descriptor "intl".
        assert_eq!(norm("AB International"), "ab intl");
        assert_eq!(norm("AG International"), "ag intl");
        assert_eq!(norm("SA International"), "sa intl");
        assert_eq!(norm("BV Trading"), "bv trd");
        // Distinct heads must stay distinct (the magnet-cluster regression).
        let heads = ["AB", "AG", "AO", "AS", "BT", "BV", "CB", "CO", "NV", "SA"];
        let normed: HashSet<String> = heads
            .iter()
            .filter(|h| **h != "AO") // "ao" is Slavic head-strippable by design
            .map(|h| norm(&format!("{} International", h)))
            .collect();
        assert_eq!(
            normed.len(),
            heads.len() - 1,
            "each head should yield a distinct normalized form, got {:?}",
            normed
        );
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
    fn list2_abbrev_variants_share_canonical() {
        // All manufacturing variants -> mfg
        for v in ["Manufacturing", "Mfg", "Manuf", "Manufac", "MFR", "MFTG"] {
            assert_eq!(norm(&format!("Acme {} Inc", v)), "acme mfg");
        }
        // International + abbrev
        assert_eq!(norm("Intl Acme"), "intl acme");
        assert_eq!(norm("Int Acme"), "intl acme");
        assert_eq!(norm("International Acme"), "intl acme");
        // Service singular + plural + abbrevs
        for v in ["Services", "Service", "Serv", "Ser", "Srv", "SRVC"] {
            assert_eq!(norm(&format!("Acme {} Inc", v)), "acme svc");
        }
        // Associates singular
        assert_eq!(norm("Acme Associates"), "acme assoc");
        assert_eq!(norm("Acme Associate"), "acme assoc");
        // Management + abbrev
        assert_eq!(norm("Acme Management"), "acme mgmt");
        assert_eq!(norm("Acme MGT"), "acme mgmt");
        // Information + Department (no plural-collapse rule for "systems" yet)
        assert_eq!(norm("Acme Information Systems"), "acme info systems");
        assert_eq!(norm("Acme Department"), "acme dept");
    }

    #[test]
    fn and_strip_unifies_ampersand_word_and_blank() {
        assert_eq!(norm("Smith & Jones"), "smith jones");
        assert_eq!(norm("Smith and Jones"), "smith jones");
        assert_eq!(norm("Smith Jones"), "smith jones");
        assert_eq!(norm("Procter & Gamble"), "procter gamble");
        assert_eq!(norm("S&P 500"), "s p 500");
        assert_eq!(norm("P & G Co"), "p g");
        assert_eq!(norm("Black and Decker Ltd"), "black decker");
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

    #[test]
    fn cjk_pinyin_compound() {
        // 有限公司 = Limited Co -> after List 3 collapse + List 1 strip
        assert_eq!(
            norm("BEIJING XYZ TECH YOU XIAN GONG SI"),
            "beijing xyz tech"
        );
        // 有限责任公司 = Limited Liability Co
        assert_eq!(norm("ZHEJIANG XYZ YOU XIAN ZE REN GONG SI"), "zhejiang xyz");
        // 股份有限公司 = Joint-Stock Limited
        assert_eq!(norm("XYZ GU FEN YOU XIAN GONG SI"), "xyz");
    }

    #[test]
    fn hungarian_long_form_compounds() {
        // Korlátolt Felelősségű Társaság = LLC
        assert_eq!(norm("XYZ KORLATOLT FELELOSSEGU TARSASAG"), "xyz");
        // Zártkörűen Működő Részvénytársaság = private joint-stock co
        assert_eq!(norm("XYZ ZARTKORUEN MUKODO RESZVENYTARSASAG"), "xyz");
        // Standalone Részvénytársaság (post-NFKD) -> rt -> strip
        assert_eq!(norm("XYZ RESZVENYTARSASAG"), "xyz");
        assert_eq!(norm("XYZ RT"), "xyz");
    }

    #[test]
    fn estonian_czech_swedish() {
        // Estonian OÜ -> NFKD: ouml + combining diaeresis -> stripped to "ou"
        assert_eq!(norm("ELBRE OÜ"), "elbre");
        // Czech S.R.O. -> period drop -> "sro" -> strip
        assert_eq!(norm("XYZ S.R.O."), "xyz");
        // Swedish full word "Aktiebolag"
        assert_eq!(norm("VOLVO AKTIEBOLAG"), "volvo");
    }

    #[test]
    fn german_ug_haftungsbeschrankt() {
        // "UG (haftungsbeschränkt)" -> parens drop -> "ug haftungsbeschrankt"
        // -> List 3 -> "ug" -> List 1 strip
        assert_eq!(norm("XYZ UG (haftungsbeschränkt)"), "xyz");
        assert_eq!(
            norm("XYZ UNTERNEHMERGESELLSCHAFT (haftungsbeschränkt)"),
            "xyz"
        );
    }

    #[test]
    fn french_sarl_spaced() {
        // "S.A R.L." (with one space) -> period drop + collapse -> "sa rl"
        // -> List 3 -> "sarl" -> List 1 strip
        assert_eq!(norm("XYZ S.A R.L."), "xyz");
    }

    #[test]
    fn russian_long_form_variants() {
        // BGN-style transliteration with apostrophe (post-drop -> ...iu)
        assert_eq!(
            norm("Obshchestvo s ogranichennoi otvetstvennost'iu MATADOR"),
            "matador"
        );
        // Older transliteration with `yu`
        assert_eq!(
            norm("OBSHCHESTVO S OGRANICHENNOI OTVETSTVENNOSTYU MATADOR"),
            "matador"
        );
        // Russian PJSC long form (Публичное Акционерное Общество)
        assert_eq!(
            norm("PUBLICHNOE AKTSIONERNOE OBSHCHESTVO SURGUTNEFTEGAZ"),
            "surgutneftegaz"
        );
        // CJSC and OJSC long forms
        assert_eq!(norm("ZAKRYTOE AKTSIONERNOE OBSHCHESTVO XYZ"), "xyz");
        assert_eq!(norm("OTKRYTOE AKTSIONERNOE OBSHCHESTVO XYZ"), "xyz");
    }
}
