"""Stage 03: assemble the film corpus — display fields + the two embed-text variants.

Reads  data/films_base.parquet + data/tmdb.parquet (optional: absent/partial TMDB
       degrades gracefully so the spine can be smoke-tested without a key)
Writes data/films.parquet — one row per film, ready for stages 04-07.

Synopsis precedence: the store's own catalog copy when present, else the TMDB
overview ("synopsis_source" records which). The two embed variants differ only
in whether the store's shelf sections are part of the text (see config.VARIANTS).
MAX_FILMS subsetting happens here so every downstream stage sees the same subset.
"""

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

    # --- synopsis: store copy first, TMDB overview as gap-fill ---
    films["synopsis"] = np.where(films["synopsis_catalog"] != "", films["synopsis_catalog"], films["overview"])
    films["synopsis_source"] = np.select(
        [films["synopsis_catalog"] != "", films["overview"] != ""],
        ["catalog", "tmdb"],
        default="none",
    )

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
    shelf_bits = [
        "; ".join(list(s) + [q for q in qs if q not in s]) for s, qs in zip(films["sections"], films["qualifiers"])
    ]
    films["embed_text_synopsis"] = (title_year + "\n" + films["synopsis"]).str.strip()
    films["embed_text_shelf"] = (
        title_year + "\n" + ["Shelf: " + b + "\n" if b else "" for b in shelf_bits] + films["synopsis"]
    ).str.strip()

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
