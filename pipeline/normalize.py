"""Title normalization shared by stage 01 (SKU -> film dedupe) and stage 02
(TMDB matching). Catalog titles look like:

    "Alien (DVD)"
    "Fighter, The (1952) (DVD-R)"
    "Sopranos, The: Season 3 (DVD)"
    "Evil Dead 2 (VHS) (Out of Print)"

i.e. natural-sort comma inversion ("X, The"), an optional embedded year, and
one or more trailing parenthetical tags for format/availability.
"""

import re
import unicodedata

# Trailing parenthetical tags that are inventory noise, not part of the title.
# Format names mirror the catalog's `format` taxonomy; availability flags and
# disc counts show up ad hoc. Applied repeatedly, innermost-last.
_TAG_RE = re.compile(
    r"""\s*\(\s*(?:
        4k(?:\s*uhd)?| blu[\s-]?ray(?:\s*3d)? | blu | dvd(?:-r)? | vhs | book | laserdisc |
        out\s*of\s*print | oop | o\.o\.p\.? |
        \d+\s*disc[s]? | disc\s*\d+ |
        widescreen | full\s*screen | criterion
    )\s*\)\s*$""",
    re.IGNORECASE | re.VERBOSE,
)

_YEAR_RE = re.compile(r"\s*\((\d{4})\)\s*$")

# Leading articles the catalog inverts to the end ("Fighter, The"). English
# plus the common foreign-film articles that alphabetized video shelves use.
# The inverted article can sit mid-title before a subtitle separator:
# "Conqueror, The: Hollywood Fallout" -> "The Conqueror: Hollywood Fallout".
_ARTICLES = {"the", "a", "an", "la", "le", "les", "el", "los", "las", "il", "der", "die", "das", "una", "un", "une"}
_INVERSION_RE = re.compile(
    r"^(.*?),\s*(" + "|".join(_ARTICLES) + r")(?=$|\s*[:\-–(])(.*)$",
    re.IGNORECASE,
)

_NUM_WORDS = "one|two|three|four|five|six|seven|eight|nine|ten"
_SEASON_RE = re.compile(
    r"""[:\-–,]?\s*\(?(?:
        (?:the\s+)?(?:complete\s+)?(?:seasons?|series|volumes?|vol\.?|parts?|books?|sets?|discs?|chapters?)
            \s*\d+[a-z]{0,2}(?:\s*(?:&|and|\+|-|–|to)\s*\d+[a-z]{0,2})? |
        seasons?\s+(?:NUM)(?:\s*(?:&|and)\s*(?:NUM))? |
        (?:the\s+)?\d+[a-z]{0,2}\s+season |
        (?:the\s+)?(?:complete|final|mini|entire)[\s-]?(?:seasons?|series|collection) |
        (?:the\s+)?complete\s+(?:\d+[a-z]{0,2}|first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth)\s+season
    )\)?\s*$""".replace("NUM", _NUM_WORDS),
    re.IGNORECASE | re.VERBOSE,
)

# A season suffix that actually implies television (vs. "Part 2" / "Vol. 3",
# which feature films and compilations use too).
TV_SEASON_RE = re.compile(r"season|series", re.IGNORECASE)


_ANY_PAREN_RE = re.compile(r"\s*\([^()]*\)\s*$")


def strip_tags(title: str) -> tuple[str, int | None, list[str]]:
    """Remove trailing parentheticals and extract an embedded (YYYY) year.
    Beyond known format tags, the catalog suffixes arbitrary shelf qualifiers
    ("(Latin American)", "(Criterion)"); a trailing parenthetical is essentially
    never part of the title itself, so all are stripped (kept as qualifiers).
    Returns (clean_title, year_or_None, qualifiers)."""
    t = title.strip()
    year = None
    qualifiers: list[str] = []
    while True:
        m = _YEAR_RE.search(t)
        if m:
            year = int(m.group(1))
            t = t[: m.start()].rstrip()
            continue
        m = _ANY_PAREN_RE.search(t)
        if m and t[: m.start()].strip(" -–:"):  # never strip down to nothing
            if not _TAG_RE.search(t):  # known tags are noise; the rest are qualifiers
                qualifiers.insert(0, t[m.start() :].strip().strip("()"))
            t = t[: m.start()].rstrip()
            continue
        # The catalog truncates long titles mid-tag ("... Discs 5 (Dv"): an
        # UNCLOSED trailing parenthetical is truncation junk, never title.
        m = re.search(r"\s*\([^()]*$", t)
        if m and t[: m.start()].strip(" -–:"):
            t = t[: m.start()].rstrip()
            continue
        break
    return t.strip(" -–:,"), year, qualifiers


def uninvert(title: str) -> str:
    """'Fighter, The' -> 'The Fighter'; 'Conqueror, The: Hollywood Fallout' ->
    'The Conqueror: Hollywood Fallout'. Leaves 'New York, New York' alone (only
    listed articles invert, and only at end-of-title or before a subtitle)."""
    m = _INVERSION_RE.match(title.strip())
    if m:
        return f"{m.group(2)} {m.group(1)}{m.group(3)}".strip()
    return title.strip()


def split_season(title: str) -> tuple[str, str | None]:
    """'The Sopranos: Season 3' -> ('The Sopranos', 'Season 3'). Returns the
    base title and the season/volume suffix (None for plain features)."""
    m = _SEASON_RE.search(title)
    if m and m.start() > 0:
        return title[: m.start()].strip(" -–:,"), title[m.start() :].strip(" -–:,()")
    return title, None


def norm_key(title: str) -> str:
    """Casefolded, accent-stripped, punctuation-free key for grouping and
    matching. NOT for display."""
    t = unicodedata.normalize("NFKD", title)
    t = "".join(c for c in t if not unicodedata.combining(c))
    t = t.casefold()
    t = re.sub(r"&", " and ", t)
    t = re.sub(r"[^a-z0-9]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def parse_catalog_title(raw_title: str) -> dict:
    """Full parse of a catalog SKU title into the fields the pipeline keys on."""
    stripped, title_year, qualifiers = strip_tags(raw_title)
    display = uninvert(stripped)
    # Peel compound suffixes repeatedly: "X: Season 6 - The Final Season" or
    # "X: Complete 1st Season Disc 4" reduce to the bare show title.
    base, season = split_season(display)
    suffixes: list[str] = []
    while season:
        suffixes.insert(0, season)
        base, season = split_season(base)
    season = " · ".join(suffixes) if suffixes else None
    return {
        "title_display": display,  # "The Sopranos: Season 3"
        "title_base": base,  # "The Sopranos"  (TMDB search query)
        "season": season,  # "Season 3" or None
        "title_year": title_year,  # year embedded in the title, if any
        "qualifiers": qualifiers,  # stripped shelf tags, e.g. ["Latin American"]
        "film_key": norm_key(display),  # dedupe key (season kept distinct)
    }
