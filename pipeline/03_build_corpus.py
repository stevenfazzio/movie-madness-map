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
    films["synopsis"] = np.where(films["synopsis_catalog"] != "", films["synopsis_catalog"], films["overview"])
    films["synopsis"] = films["synopsis"].map(clean_text)
    films["synopsis_source"] = np.select(
        [films["synopsis_catalog"] != "", films["overview"] != ""],
        ["catalog", "tmdb"],
        default="none",
    )

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
