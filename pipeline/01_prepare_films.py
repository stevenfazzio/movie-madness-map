"""Stage 01: collapse rental SKUs into films and join taxonomy term names.

Reads  data/catalog_raw.parquet + data/taxonomy_terms.parquet
Writes data/films_base.parquet — one row per film (normalized title + year),
       with formats/sections/etc. as lists and the best catalog synopsis.

Grouping key: norm_key(title) + year (from the `date` taxonomy, falling back to
a year embedded in the title). TV seasons stay separate films because the season
suffix is part of the title. Same-title-same-year remakes collide; accepted.
"""

import html as html_mod
import re

import pandas as pd
from config import CATALOG_RAW_PARQUET, FILMS_BASE_PARQUET, RENTAL_TAXONOMIES, TAXONOMY_TERMS_PARQUET
from normalize import parse_catalog_title

TAG_RE = re.compile(r"<[^>]+>")


def strip_html(html: str) -> str:
    text = TAG_RE.sub(" ", html or "")
    text = html_mod.unescape(text).replace("\xa0", " ")
    return re.sub(r"\s+", " ", text).strip()


def pick_display_title(titles: pd.Series) -> str:
    """SKUs of the same film differ in casing ('ALIEN' vs 'Alien'). Prefer a
    mixed-case variant; fall back to title-casing an all-caps one."""
    candidates = sorted(set(titles), key=lambda t: (t.isupper(), len(t)))
    best = candidates[0]
    return best.title() if best.isupper() else best


def main() -> None:
    skus = pd.read_parquet(CATALOG_RAW_PARQUET)
    terms = pd.read_parquet(TAXONOMY_TERMS_PARQUET)
    # Location/section names carry a trailing "*" house-mark on most specialty
    # shelves; strip it for display/embed (the genre-vs-location split already
    # encodes the hierarchy). Term names also carry HTML entities.
    name_of = {i: html_mod.unescape(n).rstrip("*").strip() for i, n in zip(terms["id"], terms["name"])}

    # Term-id lists -> name lists (ids are unique across WP taxonomies).
    for tax in RENTAL_TAXONOMIES:
        skus[tax] = skus[tax].map(lambda ids: [name_of[i] for i in (ids if ids is not None else []) if i in name_of])

    # Titles arrive with WP HTML entities ("&#8211;", "&amp;") — unescape first.
    skus["title_raw"] = skus["title_raw"].map(html_mod.unescape)
    parsed = pd.DataFrame([parse_catalog_title(t) for t in skus["title_raw"]], index=skus.index)
    skus = pd.concat([skus, parsed], axis=1)

    # Year: `date` taxonomy first (term names are years), title-embedded fallback.
    # The catalog has typo years ("1000", "7079"); out-of-range -> missing.
    def year_of(row) -> int | None:
        candidates = [int(n) for n in row["date"] if n.isdigit() and len(n) == 4]
        candidates.append(row["title_year"])
        for y in candidates:
            if y is not None and not pd.isna(y) and 1888 <= y <= 2027:
                return int(y)
        return None

    skus["year"] = skus.apply(year_of, axis=1).astype("Int64")
    skus["synopsis_catalog"] = skus["content_html"].map(strip_html)
    skus["group_key"] = skus["film_key"] + "|" + skus["year"].astype(str)

    def collapse(g: pd.DataFrame) -> pd.Series:
        synopses = [s for s in g["synopsis_catalog"] if s]
        return pd.Series(
            {
                "title": pick_display_title(g["title_display"]),
                "title_base": g["title_base"].iloc[0],
                "season": g["season"].iloc[0],
                "year": g["year"].iloc[0],
                "synopsis_catalog": max(synopses, key=len) if synopses else "",
                "formats": sorted({f for fs in g["format"] for f in fs}),
                "sections": sorted({s for ss in g["location"] for s in ss}),
                "qualifiers": sorted({q for qs in g["qualifiers"] for q in qs}),
                "genres_mm": sorted({x for xs in g["genre"] for x in xs}),
                "rating": next((r for rs in g["rating"] for r in rs), None),
                "languages": sorted({x for xs in g["language"] for x in xs}),
                "sku_count": len(g),
                "sku_ids": list(g["id"]),
                "sku_titles": list(g["title_raw"]),
            }
        )

    films = skus.groupby("group_key", sort=True).apply(collapse, include_groups=False).reset_index()
    # film_id is the content-derived group key ("the fighter|1952"): stable across
    # catalog refetches, so the stage-02 TMDB cache keyed on it stays valid.
    films = films.rename(columns={"group_key": "film_id"})
    films["year"] = films["year"].astype("Int64")

    assert films["film_id"].is_unique
    assert len(films) > 0
    n_with_syn = (films["synopsis_catalog"] != "").sum()
    print(f"{len(skus)} SKUs -> {len(films)} films")
    print(f"catalog synopsis coverage at film level: {n_with_syn}/{len(films)} ({n_with_syn / len(films):.0%})")
    print(f"multi-SKU films: {(films['sku_count'] > 1).sum()}")

    tmp = FILMS_BASE_PARQUET.with_suffix(".parquet.tmp")
    films.to_parquet(tmp)
    assert len(pd.read_parquet(tmp)) == len(films)
    tmp.replace(FILMS_BASE_PARQUET)
    print(f"wrote {FILMS_BASE_PARQUET}")


if __name__ == "__main__":
    main()
