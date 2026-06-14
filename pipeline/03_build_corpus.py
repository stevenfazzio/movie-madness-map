"""Stage 03: assemble the film corpus — display fields + the two embed-text variants.

Reads  data/films_base.parquet + data/tmdb.parquet (optional: absent/partial TMDB
       degrades gracefully so the spine can be smoke-tested without a key)
Writes data/films.parquet — one row per film, ready for stages 04-07.

Synopsis precedence: the store's own catalog copy when present, else the TMDB
overview ("synopsis_source" records which). The two embed variants differ only
in whether the store's curated sections are part of the text (see config.VARIANTS).
The shelf variant embeds BOTH the coarse genre ("Horror") and the fine section
("Stalker Films") as one deduped "Categories:" line. MAX_FILMS subsetting happens
here so every downstream stage sees the same subset.
"""

import html
import re
from urllib.parse import quote

import numpy as np
import pandas as pd
from config import (
    CATALOG_RAW_PARQUET,
    FILMS_BASE_PARQUET,
    FILMS_PARQUET,
    MAX_FILMS,
    MM_BASE_URL,
    SUBSET_SEED,
    TMDB_IMAGE_BASE,
    TMDB_PARQUET,
)

# The store's `genre` taxonomy mixes real genres with inventory/shelf states;
# these are noise in the genre colormap and embedding, so drop them.
GENRE_STOPLIST = {
    "4K UHD", "4K UHD New Release", "Blu-Ray New Release", "DVD New Release",
    "Head Cleaner", "Storage", "Library", "Staff Picks", "Curated",
    "Customer Recommendations", "VHS Spotlight",
}  # fmt: skip
GENRE_RELABEL = {"Lgbtq+": "LGBTQ+", "Foreign A.a.": "Foreign (Academy Award)"}
# Store-internal shorthand qualifiers (acquisition codes etc.), not descriptive.
QUAL_STOPLIST = {"Si", "G/L", "Hk", "Br", "A&e", "Btv", "Aka"}


def clean_genres(genres) -> list[str]:
    seq = genres if isinstance(genres, (list, np.ndarray)) else []
    return [GENRE_RELABEL.get(g, g) for g in seq if g not in GENRE_STOPLIST]


def clean_text(s) -> str:
    """Unescape entities and collapse whitespace (TMDB overviews carry raw \\r,
    \\n, doubled spaces; catalog copy can be double-encoded)."""
    return re.sub(r"\s+", " ", html.unescape(s or "")).strip()


def shelf_categories(genres: list[str], sections: list[str], quals) -> str:
    """One deduped curatorial line, coarse genre -> fine section -> qualifier.
    Drops a coarse genre whose name already appears as a section (48% of films:
    'Comedy'/'Comedy'), and store-shorthand qualifiers."""
    sec_lower = {s.lower() for s in sections}
    quals = quals if isinstance(quals, (list, np.ndarray)) else []
    cats = [g for g in genres if g.lower() not in sec_lower]
    cats += list(sections)
    cats += [q for q in quals if q not in QUAL_STOPLIST and q not in sections]
    return "; ".join(cats)


TMDB_COLS = [
    "film_id",
    "matched",
    "media_type",
    "overview",
    "poster_path",
    "tmdb_genres",
    "director",
    "cast_top",
    "runtime",
    "vote_average",
    "vote_count",
    "popularity",
    "original_language",
]


_T_STOP = {"the", "a", "an", "of", "and", "de", "la", "le", "el", "il", "part", "vol",
           "los", "las", "un", "une", "season", "series", "disc", "collection", "complete"}
_E_STOP = set(
    "the a an of and or to in is are was were on for with at by from as it its his her their this "
    "that these those who what which into out up down over under after before during while when where "
    "why how he she they we you i him them us not no new film movie story series season about all two one".split()
)
_FMT_STOP = _T_STOP | {"blu", "ray", "dvdr", "dvd", "vhs", "4k", "uhd", "r", "edition", "set"}


def _norm(s):
    return re.sub(r"[^a-z0-9]+", " ", str(s or "").lower()).strip()


def _ttok(t):
    return {w for w in _norm(t).split() if w and w not in _T_STOP and not w.isdigit()}


def _stoks(t):
    return {w for w in _norm(t).split() if w not in _FMT_STOP and not w.isdigit() and len(w) > 1}


def _cw(s):
    return {w for w in _norm(s).split() if len(w) > 2 and w not in _E_STOP}


def _jac(a, b):
    return len(a & b) / len(a | b) if (a and b) else 0.0


def _strip_html(s):
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", str(s or "")))).strip()


def _verbatim_owner_flags(films: pd.DataFrame) -> set:
    """Catalog synopsis is byte-shared with an unrelated title, and a sibling
    sharing it has a matching TMDB overview (so the text demonstrably belongs to
    that sibling). Catches recycled bodies whose prior occupant we can name."""
    df = films[["film_id", "title", "synopsis_catalog", "overview", "matched"]].copy()
    df["sn"] = df["synopsis_catalog"].map(_norm)
    cat = df[df["sn"].str.len() >= 60]
    flagged: set = set()
    for _, g in cat.groupby("sn"):
        if g["film_id"].nunique() < 2:
            continue
        members = list(g.itertuples())
        synt = _cw(members[0].sn)
        ov = {m.film_id: (_jac(synt, _cw(m.overview)) if m.matched else float("nan")) for m in members}
        freq: dict[str, int] = {}
        for m in members:
            for t in _ttok(m.title):
                freq[t] = freq.get(t, 0) + 1
        dominant = {t for t, c in freq.items() if c >= 2}  # franchise signature -> legit
        for m in members:
            if _ttok(m.title) & dominant:
                continue
            v = ov[m.film_id]
            if not (m.matched and v == v and v < 0.06):  # own overview must disagree
                continue
            sib = [ov[o.film_id] for o in members if o.film_id != m.film_id and ov[o.film_id] == ov[o.film_id]]
            if sib and max(sib) >= 0.12:  # a sibling owns the synopsis
                flagged.add(m.film_id)
    return flagged


def _recycled_slug_flags(films: pd.DataFrame) -> set:
    """The store reuses WordPress posts: a recycled rental keeps the previous
    occupant's URL slug and body text. Flag a matched film whose ingested
    synopsis came from a SKU whose slug names a DIFFERENT title (guarded against
    numeric/short titles that tokenize to nothing) and whose text disagrees with
    the film's own TMDB overview. Catches recycled bodies whose prior occupant
    was never TMDB-matched, so _verbatim_owner_flags misses them."""
    if not CATALOG_RAW_PARQUET.exists():
        return set()
    raw = pd.read_parquet(CATALOG_RAW_PARQUET, columns=["id", "slug", "content_html"])
    body = {i: _norm(_strip_html(c))[:200] for i, c in zip(raw["id"], raw["content_html"])}
    slug = dict(zip(raw["id"], raw["slug"]))
    flagged: set = set()
    for r in films.itertuples():
        if not r.matched:
            continue
        sc = _norm(r.synopsis_catalog)
        if len(sc) < 60:
            continue
        tt = _stoks(r.title)
        if not tt:  # numeric/short title -> slug test is meaningless
            continue
        skus = list(r.sku_ids) if r.sku_ids is not None else []
        src = next((s for s in skus if body.get(s, "") == sc[:200]), None)  # the SKU we ingested
        if src is None:
            continue
        st = _stoks(slug.get(src, ""))
        if len(st) >= 2 and not (st & tt) and _jac(_cw(r.synopsis_catalog), _cw(r.overview)) < 0.06:
            flagged.add(r.film_id)
    return flagged


def flag_contaminated_catalog_synopsis(films: pd.DataFrame) -> np.ndarray:
    """Films whose catalog synopsis is provably NOT about that film, via either
    route below. Both are the same root cause -- the store recycles WordPress
    rental posts and leaves the previous title's slug and body behind (e.g. a
    post now titled "The Imitation Game" still lives at /silicon-valley-season-6
    and carries Silicon Valley's plot). Both are gated on the synopsis disagreeing
    with the film's own TMDB overview, so a correct synopsis paired with a bad
    TMDB match is left alone. ~3,300 films; see analysis/. These get the
    TMDB-overview fallback instead of the stale catalog copy."""
    flagged = _verbatim_owner_flags(films) | _recycled_slug_flags(films)
    return films["film_id"].isin(flagged).to_numpy()


def main() -> None:
    films = pd.read_parquet(FILMS_BASE_PARQUET)

    if TMDB_PARQUET.exists():
        tmdb = pd.read_parquet(TMDB_PARQUET)[TMDB_COLS]
        films = films.merge(tmdb, on="film_id", how="left")
        print(f"TMDB data joined: {films['matched'].fillna(False).sum():.0f}/{len(films)} films matched")
    else:
        print("no tmdb.parquet — building catalog-only corpus (run stage 02 for enrichment)")
        for col in TMDB_COLS[1:]:
            films[col] = None
        films["overview"] = ""
    films["matched"] = films["matched"].fillna(False).astype(bool)
    for col in ("overview", "director"):
        films[col] = films[col].fillna("")
    for col in ("tmdb_genres", "cast_top"):
        films[col] = films[col].map(lambda v: list(v) if isinstance(v, (list, np.ndarray)) else [])

    # --- synopsis: store copy first, TMDB overview as gap-fill (then hygiene) ---
    # ...except where the store's copy is provably a different film's synopsis
    # (CMS contamination); there the TMDB overview wins. See
    # flag_contaminated_catalog_synopsis + analysis/.
    contaminated = flag_contaminated_catalog_synopsis(films)
    use_catalog = (films["synopsis_catalog"] != "") & ~contaminated
    films["synopsis"] = np.where(use_catalog, films["synopsis_catalog"], films["overview"])
    films["synopsis"] = films["synopsis"].map(clean_text)
    films["synopsis_source"] = np.select(
        [use_catalog, films["overview"] != ""],
        ["catalog", "tmdb"],
        default="none",
    )
    films["synopsis_overridden"] = contaminated
    print(f"synopsis override: {int(contaminated.sum())} contaminated catalog synopses replaced with TMDB overview")

    # --- coarse genre (cleaned) — used by embed text, hover, and colormap ---
    films["genres_coarse"] = films["genres_mm"].map(clean_genres)
    films["genre_coarse_str"] = films["genres_coarse"].map(lambda gs: "; ".join(gs))

    # --- display fields ---
    year_str = films["year"].map(lambda y: str(int(y)) if pd.notna(y) else "")
    films["decade"] = films["year"].map(lambda y: f"{int(y) // 10 * 10}s" if pd.notna(y) else "Unknown")
    films["section_primary"] = films["sections"].map(lambda s: s[0] if len(s) else "Unshelved")
    films["formats_str"] = films["formats"].map(", ".join)
    films["vhs_only"] = films["formats"].map(lambda f: list(f) == ["VHS"])
    films["cast_str"] = films["cast_top"].map(lambda c: ", ".join(c[:4]))
    films["tmdb_genre_primary"] = films["tmdb_genres"].map(lambda g: g[0] if len(g) else "—")
    films["poster_url"] = films["poster_path"].map(lambda p: f"{TMDB_IMAGE_BASE}{p}" if pd.notna(p) and p else "")
    films["mm_url"] = [f"{MM_BASE_URL}/search/?query={quote(t)}&search_by=Title" for t in films["title"]]

    # --- embed texts (the variant fork; see CLAUDE.md) ---
    title_year = films["title"] + np.where(year_str != "", " (" + year_str + ")", "")
    # Prepend the season to the embedded synopsis: a show's seasons share a
    # byte-identical series blurb (25.7% of rows), so the season spreads them
    # along a gradient instead of stacking on one point.
    embed_syn = [
        f"{season}. {syn}" if (isinstance(season, str) and season and syn) else syn
        for season, syn in zip(films["season"], films["synopsis"])
    ]
    # Shelf variant: one deduped coarse->fine "Categories:" line before the synopsis.
    categories = [
        shelf_categories(g, s, q) for g, s, q in zip(films["genres_coarse"], films["sections"], films["qualifiers"])
    ]
    films["embed_text_synopsis"] = pd.Series(
        [f"{ty}\n{syn}".strip() for ty, syn in zip(title_year, embed_syn)], index=films.index
    )
    films["embed_text_shelf"] = pd.Series(
        [
            "\n".join(part for part in (ty, f"Categories: {cat}" if cat else "", syn) if part).strip()
            for ty, cat, syn in zip(title_year, categories, embed_syn)
        ],
        index=films.index,
    )

    if MAX_FILMS is not None and len(films) > MAX_FILMS:
        films = films.sample(n=MAX_FILMS, random_state=SUBSET_SEED).sort_values("film_id")
        print(f"MAX_FILMS subset: {len(films)} films (seed {SUBSET_SEED})")

    n_syn = (films["synopsis"] != "").sum()
    src_counts = films["synopsis_source"].value_counts().to_dict()
    print(f"{len(films)} films; synopsis coverage {n_syn / len(films):.0%} (sources: {src_counts})")

    tmp = FILMS_PARQUET.with_suffix(".parquet.tmp")
    films.to_parquet(tmp, index=False)
    assert len(pd.read_parquet(tmp)) == len(films)
    tmp.replace(FILMS_PARQUET)
    print(f"wrote {FILMS_PARQUET}")


if __name__ == "__main__":
    main()
