"""Stage 07: render the interactive DataMapPlot, one HTML per variant.

Layout: one point per film, positioned by stage 05's UMAP of that variant's
embeddings. Hover shows poster (when TMDB matched), title/year, director, the
synopsis, the store shelf section, formats, and rating; clicking a point opens
the film's search page on moviemadness.org ("go rent it"). A colormap dropdown
covers decade, shelf section, format, rating, TMDB genre/popularity, runtime,
and how many editions the store stocks. Toponymy region names float on top.

Inputs:  data/umap_coords_<variant>.npz, data/films.parquet,
         [optional] data/toponymy_labels_<variant>.parquet
Output:  data/movie_map_<variant>.html  (+ copied to docs/<variant>.html)
"""

from __future__ import annotations

import re
from collections import Counter
from html import escape

import datamapplot
import glasbey
import numpy as np
import pandas as pd
from config import (
    FILMS_PARQUET,
    PROJECT_NAME,
    PROJECT_TAGLINE,
    VARIANTS,
    docs_html,
    labels_parquet,
    map_html,
    umap_npz,
)
from matplotlib.colors import to_hex

VARIANT_BLURB = {
    "synopsis": "layout: what the films are about",
    "shelf": "layout: synopsis + the store's shelf sections",
}

FORMAT_COLORS = {
    "VHS only": "#c0392b",
    "4K UHD": "#8e44ad",
    "Blu-Ray": "#2471a3",
    "DVD": "#7f8c8d",
    "Other": "#b7950b",
}

RATING_COLORS = {
    "G": "#27ae60",
    "PG": "#82e0aa",
    "PG-13": "#f4d03f",
    "R": "#e67e22",
    "NC-17": "#cb4335",
    "X": "#78281f",
    "N/R": "#95a5a6",
    "Unrated": "#95a5a6",
}

ATTRIBUTION_HTML = """
<div id="mm-attribution" style="position:fixed;bottom:8px;left:10px;z-index:10;font-size:11px;
     font-family:Roboto,sans-serif;opacity:.75;pointer-events:auto;max-width:60vw">
  Catalog: <a href="https://www.moviemadness.org/" target="_blank" rel="noopener">Movie Madness</a>,
  a nonprofit video store in Portland, OR (Hollywood Theatre) &nbsp;&middot;&nbsp;
  Film metadata &amp; posters: <a href="https://www.themoviedb.org/" target="_blank" rel="noopener">TMDB</a>
  (this product uses the TMDB API but is not endorsed or certified by TMDB) &nbsp;&middot;&nbsp;
  Click a film to find it at the store
</div>
"""


# DataMapPlot hardcodes the deck.gl scroll-zoom at speed 0.01 (datamap.js),
# which feels sluggish; bump it. 0.04 = 4x the default — snappy but still smooth.
ZOOM_SPEED = 0.04


def postprocess_html(html: str) -> str:
    """Apply our post-render patches to a DataMapPlot HTML string: attribution
    footer + faster scroll-zoom. Shared so already-rendered files can be patched
    in place without a full re-render."""
    html, n = re.subn(r"</body>", ATTRIBUTION_HTML + "</body>", html, count=1)
    assert n == 1, "no </body> found for attribution injection"
    # Minified controller config: scrollZoom:{speed:0.01,smooth:true}
    html, n = re.subn(r"(scrollZoom:\{speed:)[0-9.]+", rf"\g<1>{ZOOM_SPEED}", html, count=1)
    assert n == 1, "scroll-zoom speed token not found (datamapplot internals changed?)"
    return html


def categorical_color_mapping(values, default=None, default_color="#c9ccd1"):
    uniques = sorted(set(map(str, values)))
    others = [v for v in uniques if v != default]
    palette = glasbey.create_palette(palette_size=max(len(others), 1))
    mapping = {v: to_hex(palette[i]) for i, v in enumerate(others)}
    if default is not None and default in uniques:
        mapping[default] = default_color
    return mapping


def _fill_nonfinite(a):
    a = np.asarray(a, dtype=float)
    finite = a[np.isfinite(a)]
    fill = np.median(finite) if finite.size else 0.0  # all-NaN (e.g. tiny smoke subset) -> 0
    return np.where(np.isfinite(a), a, fill)


def pick_coarse_genre(genre_lists: pd.Series) -> pd.Series:
    """One coarse genre per film for the colormap. Films carry 0-4 genres (99.4%
    have 1); when >1, pick the GLOBALLY RAREST so a film isn't swallowed by the
    huge 'Foreign' bucket (a Foreign+Horror film colors as Horror). Empty -> Other."""
    counts = Counter(g for gs in genre_lists for g in (gs if isinstance(gs, (list, np.ndarray)) else []))

    def pick(gs):
        gs = list(gs) if isinstance(gs, (list, np.ndarray)) else []
        return min(gs, key=lambda g: (counts[g], g)) if gs else "Other"

    return genre_lists.map(pick)


def format_bucket(formats: list) -> str:
    fset = set(formats)
    if fset == {"VHS"}:
        return "VHS only"
    if "4K UHD" in fset:
        return "4K UHD"
    if any("Blu-Ray" in f for f in fset):
        return "Blu-Ray"
    if any(f.startswith("DVD") for f in fset):
        return "DVD"
    return "Other"


def bucket_top_n(values: pd.Series, n: int, other="Other") -> pd.Series:
    top = values.value_counts().nlargest(n).index
    return values.where(values.isin(top), other)


def render_variant(variant: str, films: pd.DataFrame) -> None:
    crd = np.load(umap_npz(variant), allow_pickle=True)
    coords = crd["coords"].astype(np.float32)
    layout = pd.DataFrame({"film_id": crd["film_id"], "x": coords[:, 0], "y": coords[:, 1]})

    df = layout.merge(films, on="film_id", how="left")
    assert len(df) == len(layout), "join changed row count — film_id not unique"
    print(f"[{variant}] map points: {len(df):,}")
    coords_xy = df[["x", "y"]].to_numpy()

    label_layers = []
    lp = labels_parquet(variant)
    if lp.exists():
        topo = pd.read_parquet(lp)
        merged = df[["film_id"]].merge(topo, on="film_id", how="left")
        layer_cols = sorted(
            (c for c in topo.columns if c.startswith("label_layer_")),
            key=lambda s: int(s.rsplit("_", 1)[1]),
        )
        label_layers = [merged[c].fillna("Unlabelled").to_numpy() for c in layer_cols]
        print(f"  [{variant}] using {len(label_layers)} Toponymy label layer(s)")
    else:
        print(f"  [{variant}] no Toponymy labels; rendering without region names")

    # --- hover fields (raw HTML is interpolated into the template; pre-escape) ---
    year_str = df["year"].map(lambda y: str(int(y)) if pd.notna(y) else "?")
    poster_html = [
        f'<img src="{p}" style="width:92px;float:left;margin:0 10px 6px 0;border-radius:3px" loading="lazy">'
        if p
        else ""
        for p in df["poster_url"].fillna("")
    ]
    synopsis = df["synopsis"].fillna("").map(lambda s: escape(s if len(s) <= 320 else s[:317] + "…"))
    byline = [
        " · ".join(
            x
            for x in (
                escape(d) if d else "",
                f"{int(r)} min" if pd.notna(r) and r else "",
            )
            if x
        )
        for d, r in zip(df["director"].fillna(""), df["runtime"])
    ]
    shelf_str = df["sections"].map(lambda s: escape("; ".join(s)) if len(s) else "—")
    cast_html = [
        f'<div style="margin-top:4px;font-size:11px;opacity:.65">{escape(c)}</div>' if c else ""
        for c in df["cast_str"].fillna("")
    ]
    # Coarse-genre hover line, deduped against the fine shelf sections (drop a
    # coarse token already shown as a section, e.g. "Comedy"); omit if nothing left.
    genre_html = []
    for genres, sections in zip(df["genres_coarse"], df["sections"]):
        sec_lower = {s.lower() for s in sections}
        coarse = [g for g in (genres if isinstance(genres, (list, np.ndarray)) else []) if g.lower() not in sec_lower]
        genre_html.append(
            f'<div style="margin-top:6px;font-size:11px;color:#7a9a6a">Genre: {escape("; ".join(coarse))}</div>'
            if coarse
            else ""
        )
    rating_clean = df["rating"].fillna("N/R").replace({"": "N/R", "N/": "N/R"})

    extra = pd.DataFrame(
        {
            "poster": poster_html,
            "title": [escape(t) for t in df["title"].fillna("")],
            "year": year_str.to_numpy(),
            "byline": byline,
            "synopsis": synopsis.to_numpy(),
            "shelf": shelf_str.to_numpy(),
            "formats": [escape(f) for f in df["formats_str"].fillna("")],
            "rating": [escape(r) for r in rating_clean],
            "mm_url": df["mm_url"].fillna("").to_numpy(),
        }
    )

    hover_template = (
        '<div style="max-width:360px;overflow:hidden">'
        "{poster}"
        '<div style="font-weight:700;font-size:14px;line-height:1.3">{title} '
        '<span style="font-weight:400;opacity:.7">({year})</span></div>'
        '<div style="margin-top:3px;font-size:11px;opacity:.8">{byline}</div>'
        '<div style="margin-top:5px;font-size:12px;line-height:1.4">{synopsis}</div>'
        "{genre}"
        '<div style="margin-top:4px;font-size:11px;color:#9a6a00">Shelf: {shelf}</div>'
        '<div style="margin-top:3px;font-size:11px;opacity:.7">{formats} &nbsp;·&nbsp; {rating}</div>'
        "{cast}"
        "</div>"
    )
    extra["cast"] = cast_html
    extra["genre"] = genre_html

    # Search corpus (DataMapPlot substring-matches this): title + people + the
    # store's shelf sections, so "noir"/"kung fu"/"criterion" find the right
    # shelf. Synopsis is deliberately excluded — its common words ("love",
    # "war") would flood the substring match and dilute title/person lookup.
    section_search = df["sections"].map(lambda s: " ".join(s) if len(s) else "")
    qual_search = df["qualifiers"].map(lambda q: " ".join(q) if len(q) else "")  # Criterion, A24, Arrow...
    hover_text = (
        df["title"].fillna("")
        + " " + df["director"].fillna("")
        + " " + df["cast_str"].fillna("")
        + " " + section_search
        + " " + qual_search
    ).to_numpy()

    # --- colormaps ---
    year_num = _fill_nonfinite(df["year"].to_numpy(dtype=float))
    # Coarse store genre (75 values) bucketed to top-15 + "Other" — the store's
    # top-level shelf family ("Horror" of "Horror > Stalker Films").
    genre_cat = bucket_top_n(pick_coarse_genre(df["genres_coarse"]), 15).to_numpy()
    fmt_cat = df["formats"].map(format_bucket).to_numpy()
    rating_cat = rating_clean.to_numpy()
    editions = df["sku_count"].to_numpy(dtype=float).clip(1, 10)

    rawdata = [year_num, genre_cat, fmt_cat, rating_cat, editions]
    metadata = [
        {"field": "year", "description": "Release year", "kind": "continuous", "cmap": "viridis"},
        {
            "field": "store_genre",
            "description": "Genre (store, coarse)",
            "kind": "categorical",
            "color_mapping": categorical_color_mapping(genre_cat, default="Other"),
        },
        {"field": "format", "description": "Format", "kind": "categorical", "color_mapping": FORMAT_COLORS},
        {
            "field": "rating",
            "description": "MPAA rating",
            "kind": "categorical",
            # RATING_COLORS spread LAST so the hand-picked green->red palette wins
            # over the glasbey fallback (which only needs to cover stray values).
            "color_mapping": {**categorical_color_mapping(rating_cat, default="N/R"), **RATING_COLORS},
        },
        {"field": "editions", "description": "Editions in stock (1-10+)", "kind": "continuous", "cmap": "YlGnBu"},
    ]

    # TMDB-derived colormaps only exist on an enriched corpus (stage 02 run).
    if df["matched"].fillna(False).any():
        tmdb_genre_cat = df["tmdb_genre_primary"].fillna("—").to_numpy()
        pop = df["popularity"].to_numpy(dtype=float)
        pop_log = np.log10(np.where(np.isfinite(pop) & (pop > 0), pop, np.nan))
        finite_pop = pop_log[np.isfinite(pop_log)]
        pop_log = np.where(np.isfinite(pop_log), pop_log, finite_pop.min() if finite_pop.size else 0.0)
        runtime_num = _fill_nonfinite(df["runtime"].to_numpy(dtype=float)).clip(40, 240)
        rawdata += [tmdb_genre_cat, pop_log, runtime_num]
        metadata += [
            {
                "field": "tmdb_genre",
                "description": "Genre (TMDB)",
                "kind": "categorical",
                "color_mapping": categorical_color_mapping(tmdb_genre_cat, default="—"),
            },
            {"field": "popularity", "description": "TMDB popularity (log)", "kind": "continuous", "cmap": "inferno"},
            {"field": "runtime", "description": "Runtime (min)", "kind": "continuous", "cmap": "cividis"},
        ]

    print(f"  [{variant}] rendering DataMapPlot...")
    fig = datamapplot.create_interactive_plot(
        coords_xy,
        *label_layers,
        hover_text=hover_text,
        hover_text_html_template=hover_template,
        extra_point_data=extra,
        on_click="window.open(`{mm_url}`, '_blank')",
        colormap_rawdata=rawdata,
        colormap_metadata=metadata,
        title=PROJECT_NAME,
        sub_title=f"{PROJECT_TAGLINE} · {VARIANT_BLURB[variant]}",
        enable_search=True,
        font_family="IBM Plex Sans",
        tooltip_font_family="IBM Plex Sans",
        darkmode=True,
    )
    out = map_html(variant)
    fig.save(str(out))

    # Post-render: attribution footer + faster scroll-zoom (see postprocess_html).
    out.write_text(postprocess_html(out.read_text()))
    print(f"  [{variant}] saved {out} ({out.stat().st_size / 1e6:.1f} MB)")

    dh = docs_html(variant)
    dh.parent.mkdir(exist_ok=True)
    dh.write_bytes(out.read_bytes())
    print(f"  [{variant}] copied to {dh}")


def main():
    films = pd.read_parquet(FILMS_PARQUET)
    for variant in VARIANTS:
        render_variant(variant, films)


if __name__ == "__main__":
    main()
