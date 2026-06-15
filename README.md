# Movie Madness Map

An interactive 2D map of the rental catalog of [Movie Madness](https://www.moviemadness.org/) —
the Portland, Oregon nonprofit video store (run by the [Hollywood Theatre](https://hollywoodtheatre.org/))
whose collection of ~90,000 titles on DVD, Blu-ray, 4K, and VHS is one of the largest physical-media
libraries in the world.

**One dot = one film** (or one TV season, or the occasional book). Films are embedded from their
title, year, and synopsis, laid out with UMAP so similar films sit together, and the regions are
named by [Toponymy](https://github.com/TutteInstitute/toponymy). Hovering shows the poster, director,
synopsis, and — most importantly — *which shelf it lives on*, because Movie Madness's hand-curated
section system ("Hollywood Directors", "60s Go Go Chicks", "A Better America", …) is the soul of
the store. Clicking a film opens its page in the store's own catalog search.

**Live:** <https://stevenfazzio.github.io/movie-madness-map/>

The map is built in **two layout variants** to answer a design question empirically:

- `shelf` — embeds the store's shelf section alongside title + year + synopsis, letting the store's
  own curatorial logic partially shape the neighborhoods. **This is the published map**
  (`docs/index.html`); its national-cinema and director regions preserve the store's personality.
- `synopsis` — embeds title + year + synopsis only: the map organizes purely by what films are
  about. Kept for comparison at [`/synopsis.html`](https://stevenfazzio.github.io/movie-madness-map/synopsis.html).

## Pipeline

Eight stages, each a standalone script reading the previous stage's output (see
`pipeline/config.py` for all knobs):

```
00_fetch_catalog.py   WP REST API crawl (gentle)        -> catalog_raw.parquet + taxonomy_terms.parquet
01_prepare_films.py   SKU -> film dedupe + term joins   -> films_base.parquet
02_fetch_tmdb.py      TMDB match + enrich (needs key)   -> tmdb.parquet
03_build_corpus.py    synopsis precedence + embed texts -> films.parquet
04_embed_films.py     Cohere embed-v4.0, per variant    -> embeddings_<variant>.npz
05_reduce_umap.py     UMAP 2D layout, per variant       -> umap_coords_<variant>.npz
06_label_topics.py    Toponymy + Claude region names    -> toponymy_labels_<variant>.parquet
07_visualize.py       DataMapPlot interactive HTML      -> movie_map_<variant>.html + docs/
```

```bash
make install                    # uv sync --extra dev
cp .env.example .env            # then fill in CO_API_KEY / ANTHROPIC_API_KEY / TMDB_API_KEY
uv run python pipeline/00_fetch_catalog.py    # ... through 07, in order
```

## Data sources & attribution

- **Catalog** (titles, formats, years, ratings, shelf sections, many synopses): the public
  [Movie Madness](https://www.moviemadness.org/) rental catalog, fetched politely (throttled,
  resumable) from their public WordPress REST API. If you love this data, go rent a movie —
  they ship memberships too.
- **Film metadata** (posters, director/cast, synopsis gap-fill, genres): [TMDB](https://www.themoviedb.org/).
  This product uses the TMDB API but is not endorsed or certified by TMDB.

## License

The **code** in this repository is licensed under the [MIT License](LICENSE).

The **map data** embedded in the rendered pages under `docs/` (catalog text, synopses, genres,
cast/crew, ratings, poster references) is *not* covered by that grant — it is derived from Movie
Madness and TMDB and remains subject to their respective terms (see Data sources & attribution
above). Reuse the code freely; clearing reuse of the embedded data is your own responsibility.
